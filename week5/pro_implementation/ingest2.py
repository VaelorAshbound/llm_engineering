from pathlib import Path
from typing import cast

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion
from litellm.types.utils import Choices, ModelResponse
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(override=True)
OPENAI = OpenAI()
MODEL = "gpt-4.1-mini"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
DB_NAME = "knowledge_base"
COLLECTION_NAME = "docs"
AVG_CHUNK_SIZE = 500


class Results(BaseModel):
    page_content: str = Field(
        description="""
        A chunk of text based on the provided document, that is most likely to be surfaced in a query.
        It includes a headline, a summary, and the original text of the chunk.
        """
    )
    metadata: dict = Field(
        description="Metadata about the chunk, including the source and type of the original document"
    )


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Results(
            page_content=self.headline
            + "\n\n"
            + self.summary
            + "\n\n"
            + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk] = Field(
        description="A list of chunks extracted from the provided document"
    )


def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""

    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            file_path = file.as_posix()
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(
                    {"type": doc_type, "source": file_path, "text": content}
                )
    return documents


def make_prompt(document):
    how_many = (len(document["text"]) // AVG_CHUNK_SIZE) + 1
    return f"""
    You are an expert document splitter. Your duty is to take a document and split it into {how_many} chunks for a KnowledgeBase.

    Here is the context so that you can understand the document:
    1. The document is a markdown file that contains information about a specific topic related to Insurellm.
    2. The document is from the shared drive of a company called Insurellm.
    3. The document is of type: {document["type"]}
    4. The document has been retrieved from: {document["source"]}

    The process is as follows:
    1. Read the document and understand its content.
    2. Split the document into {how_many} overlapping chunks, where each chunk is approximately {AVG_CHUNK_SIZE} characters long.
    3. For each chunk, create a headline that captures the main topic of the chunk, typically a few words, that is most likely to be surfaced in a query.
    4. For each chunk, write a summary of a few sentences summarizing the content of the chunk to answer common questions.
    5. For each chunk, include the original text of the chunk from the provided document, exactly as is, not changed in any way.

    The rules are as follows:
    1. The chunks should be overlapping, meaning that the end of one chunk should overlap with the
        beginning of the next chunk to ensure that important information is not lost between chunks.
    2. The headline should be concise and informative, capturing the essence of the chunk's content.
    3. The summary should be clear and concise,
        providing a quick overview of the chunk's content to help users understand what information
        is contained in the chunk without having to read the entire original text.
    4. The original text should be included in its entirety for each chunk, without any modifications,
        to ensure that the full context is preserved for users who want to read the original content.

    Finally, here is the document that you need to split into chunks:
    {document["text"]}

    """


def generate_messages(document):
    user_prompt = make_prompt(document)
    return [
        {"role": "user", "content": user_prompt},
    ]


def process_document(document):
    messages = generate_messages(document)
    response = cast(
        ModelResponse,
        completion(model=MODEL, messages=messages, response_format=Chunks),
    )
    result = cast(Choices, response.choices[0]).message.content or ""
    assert result is not None, "Expected non-None content from completion"
    doc_as_chunks = Chunks.model_validate_json(result).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


def create_chunks(documents):
    all_chunks = []
    for document in documents:
        chunks = process_document(document)
        all_chunks.extend(chunks)
    return all_chunks


def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(name=COLLECTION_NAME)

    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]
    texts = [chunk.page_content for chunk in chunks]
    embeddings = OPENAI.embeddings.create(model=MODEL, input=texts).data
    vectors = [e.embedding for e in embeddings]
    collection.add(ids=ids, metadatas=metas, documents=texts, embeddings=vectors)  # type: ignore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
