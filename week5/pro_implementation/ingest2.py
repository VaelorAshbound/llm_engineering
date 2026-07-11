# 1 Do Imports
# 2 Do Settings
# 3 Create Classes (done)
# 4 Fetch Documents (done)
# 5 Make Prompt (done)
# 6 Make Messages (done)
# 7 Process Documents
# 8 Create Chunks
# 9 Create Embeddings

# %%
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion, embedding
from litellm.types.utils import Choices, ModelResponse
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_not_exception_type, wait_exponential
from tqdm import tqdm

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

try:
    KNOWLEDGE_BASE = Path(__file__).parent.parent / "knowledge-base"
    DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
except NameError:
    KNOWLEDGE_BASE = Path.cwd().parent / "knowledge-base"
    DB_NAME = str(Path.cwd().parent / "preprocessed_db")

COLLECTION_NAME = "docs"

MODEL = "openrouter/deepseek/deepseek-v4-flash"
EMBEDDING_MODEL = "openai/text-embedding-3-large"

TARGET_WORDS = 200  # aim for ~200-word chunks
OVERLAP_WORDS = 50  # ~25% overlap between adjacent chunks
MAX_OUTPUT_TOKENS = 32000

WORKERS = 4

WAIT = wait_exponential(multiplier=1, min=10, max=240)

SYSTEM_PROMPT = """
    ## GOAL
    You are a precision document analyst for InsureLLM, an insurance technology company. Your sole purpose is to transform a single structured document into a set of semantically coherent chunks — each enriched with a headline and a summary — to power downstream search, retrieval, and knowledge-base indexing. Every chunk you produce must be accurate, self-contained, and faithful to the source text.

    ---

    ## INFORMATION

    ### About InsureLLM
    InsureLLM is an insurance technology company. Documents you will process belong to the InsureLLM knowledge corpus and may cover products, contracts, employees etc.

    Document type context matters. A contract should be chunked by clause. A product doc by feature or section. An employee profile by area (bio, role, skills, etc.). Apply judgment based on `type`.

    ### Input Document Structure
    You will receive the document in the user message containing:
    - `type`: the document category
    - `source`: the file path the document belongs to
    - `text`: the full raw content to be chunked, with every line prefixed by its 1-indexed line number in the form `<n>: <line content>`

    ### Chunking Rules
    1. **Semantic boundaries first**: Split at natural boundaries — sections, paragraphs, clauses, bullet groups, topics — never mid-sentence or mid-clause.
    2. **Self-contained meaning**: Each chunk must make sense in isolation without requiring other chunks for basic comprehension.
    3. **No fabrication**: Headlines and summaries must derive entirely from the original text. Do not infer, embellish, or add external knowledge.
    4. **Report line numbers, never copy text**: You identify each chunk by the first and last source line it spans (`start_line` and `end_line`, 1-indexed and inclusive). You must NOT reproduce, copy, paraphrase, or normalize the source text anywhere in your output — the original text is reconstructed downstream directly from the line numbers you report. The `<n>:` prefixes are reference markers only, not part of the content.
    5. **Target ~{target_words} words per chunk**: Size each chunk at roughly {target_words} words, always cutting at natural boundaries (sections, paragraphs, clauses) rather than at an exact word count. This document should yield at least {min_chunks} chunks — produce more if its structure warrants, never fewer. Do not pad or fragment mid-thought to hit a number; the word target is guidance, the natural boundary wins.
    6. **Minimum chunk size**: A chunk must contain at least one meaningful, complete sentence or clause. Do not create single-word or trivial chunks.
    7. **Overlap between chunks**: Adjacent chunks must overlap by about {overlap_words} words (~25% of a chunk). Set each chunk's `start_line` a few lines before the previous chunk's `end_line` so the boundary line(s) appear in both chunks — the same text in two chunks improves retrieval.
    8. **Exhaustive coverage**: Every source line must be covered by at least one chunk. The first chunk must start at line 1, the last chunk must end at the final line, and consecutive chunks must not leave gaps (chunk N's `start_line` ≤ chunk N-1's `end_line` + 1).

    ---

    ## ACTION

    Follow this exact process for every document:

    **Step 1 — Understand the document**
    Read the full `text`. Identify the document `type`, its overall topic, and its internal structure (headings, sections, numbered clauses, paragraph breaks, lists, etc.).

    **Step 2 — Plan chunk boundaries**
    Before writing output, mentally identify where the natural semantic breaks are. For each candidate chunk, confirm it:
    - Is self-contained
    - Does not split a logical unit (a clause, a list, a paragraph)
    - Would not benefit from being merged with an adjacent chunk (avoid micro-chunks)

    **Step 3 — Generate each chunk**
    For each planned chunk, produce:
    - `headline`: A concise title (5–10 words) that identifies what this chunk is about. Written in title case. Must be informative, not generic (e.g., not "Section 1" or "Introduction").
    - `summary`: 1–3 sentences that distill the key information in the chunk. Written for a reader who needs to decide if this chunk is relevant to their query. Do not copy the original text verbatim here — rephrase concisely.
    - `start_line`: The 1-indexed number of the first source line this chunk covers (inclusive).
    - `end_line`: The 1-indexed number of the last source line this chunk covers (inclusive). Must be ≥ `start_line`.

    **Step 4 — Validate before output**
    Before finalizing, verify:
    - Each chunk is roughly {target_words} words (a natural boundary always overrides exact size)
    - Total chunk count is at least {min_chunks}
    - Adjacent chunks overlap by about {overlap_words} words (next `start_line` sits a few lines before the previous `end_line`)
    - No source line has been omitted — the chunks in order span from line 1 to the final line with no gaps (overlaps expected)

"""

USER_PROMPT = """
    Process the following InsureLLM document and return chunks. Follow all rules defined in your instructions exactly.

    Document:
    - `type`: {type}
    - `source`: {source}
    - `text` (each line prefixed with its 1-indexed line number as `<n>: `):
{text}
"""


class Documents(BaseModel):
    source: str = Field(description="The file path the document belongs to")
    type: str = Field(description="The document category")
    text: str = Field(description="The full raw content of the document")


class Metadata(BaseModel):
    source: str = Field(description="The file path the document belongs to")
    type: str = Field(description="The document category")


class Result(BaseModel):
    page_content: str = Field(
        description="Contains headline, summary, and original text of the chunk"
    )
    metadata: Metadata


class Chunk(BaseModel):
    headline: str = Field(
        description="A short, specific title (max 10 words) capturing the chunk's main topic"
    )
    summary: str = Field(
        description="A 2–3 sentence description of what the chunk covers, written in third-person"
    )
    start_line: int = Field(
        description="1-indexed number of the first source line this chunk covers (inclusive)"
    )
    end_line: int = Field(
        description="1-indexed number of the last source line this chunk covers (inclusive)"
    )

    def to_result(self, document: Documents):
        # Reconstruct the verbatim text in code by slicing the source lines,
        # so the stored content is byte-identical to the source — the model
        # only ever reports line numbers, never copies text.
        lines = document.text.splitlines()
        start = max(1, self.start_line)
        end = min(len(lines), self.end_line)
        original_text = "\n".join(lines[start - 1 : end])

        metadata: Metadata = Metadata(source=document.source, type=document.type)
        result: Result = Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + original_text,
            metadata=metadata,
        )
        return result


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents():
    documents: list[Documents] = []

    for folder in KNOWLEDGE_BASE.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            source = file.as_posix()
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(Documents(type=doc_type, source=source, text=text))

    return documents


def make_messages(document: Documents):
    words = len(document.text.split())  # total words
    chunk_size = TARGET_WORDS  # words per chunk
    overlap = OVERLAP_WORDS  # words shared between neighbours
    step = chunk_size - overlap  # new words per chunk

    if words <= chunk_size:
        num_chunks = 1  # whole doc fits in one chunk
    else:
        # first is different as it has no overlap
        leftover = words - chunk_size  # words after the first chunk
        extra_chunks = math.ceil(leftover / step)  # round UP to cover them all
        num_chunks = 1 + extra_chunks  # +1 for that first chunk
    system_prompt = SYSTEM_PROMPT.format(
        target_words=TARGET_WORDS,
        overlap_words=OVERLAP_WORDS,
        min_chunks=num_chunks,
    )
    numbered_text = "\n".join(
        f"{i}: {line}" for i, line in enumerate(document.text.splitlines(), start=1)
    )
    user_prompt = USER_PROMPT.format(
        type=document.type, source=document.source, text=numbered_text
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@retry(wait=WAIT, retry=retry_if_not_exception_type(RuntimeError))
def process_document(document):
    message = make_messages(document)
    response = cast(
        ModelResponse,
        completion(
            model=MODEL,
            response_format=Chunks,
            messages=message,
            temperature=0.1,
            max_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    choice = cast(Choices, response.choices[0])
    reply = choice.message.content or ""
    chunks = Chunks.model_validate_json(reply).chunks
    result = [chunk.to_result(document) for chunk in chunks]
    print(result)
    return result


def process_documents(documents):
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for result in tqdm(pool.map(process_document, documents), total=len(documents)):
            results.extend(result)
    return results


def create_embeddings(chunks):
    chroma_client = PersistentClient(path=DB_NAME)
    if COLLECTION_NAME in [
        collection.name for collection in chroma_client.list_collections()
    ]:
        chroma_client.delete_collection(COLLECTION_NAME)

    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    ids = [str(chunk) for chunk in range(len(chunks))]
    page_contents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata.model_dump() for chunk in chunks]

    response = embedding(
        model=EMBEDDING_MODEL,
        input=page_contents,
        api_base=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    vectors = [data["embedding"] for data in response.data]

    collection.add(
        ids=ids, embeddings=vectors, metadatas=metadatas, documents=page_contents
    )


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = process_documents(documents)
    create_embeddings(chunks)
