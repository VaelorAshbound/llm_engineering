# %%

# #1 Import necessary libraries and set up environment variables

from pathlib import Path
from typing import List, cast

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion
from litellm.types.utils import Choices, ModelResponse
from openai import OpenAI
from pydantic import BaseModel, Field
from week8.day4 import collection

load_dotenv(override=True)
OPENAI = OpenAI()
MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
DB_NAME = "knowledge_bases"
KNOWLEDGE_BASE_PATH = Path("knowledge_base")
COLLECTION_NAME = "docs"
AVG_CHUNK_SIZE = 500
RETRIEVAL_K = 10

# #2 Creates pydantic models for the document and the knowledge base


# Represents the structure of the final chunked documents
# That will be stored in vector database.
class Document(BaseModel):
    page_content: str = Field(..., description="The content of the document.")
    metadata: dict = Field(..., description="Metadata associated with the document.")


# Represents the structure of the chunked documents.
# LLM will create it from the original knowledge base documents.
# Each chunk contains a headline, a summary, and the original text.
class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query"
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    # Convert Chunk (headline + summary + original_text) into Document (page_content + metadata)
    def as_document(self, document):
        metadata = {"source": document["source"], "type": document["type"]}
        return Document(
            page_content=self.headline
            + "\n\n"
            + self.summary
            + "\n\n"
            + self.original_text,
            metadata=metadata,
        )


# Represents a collection of chunks.
class Chunks(BaseModel):
    chunks: List[Chunk] = Field(description="A list of chunks")


# #3 First, Load all knowledge base documents
# Second, For each document, create chunks using the LLM.
# Third, Store the chunks in a vector database for later retrieval.


# Directory Loader Function
def load_documents_from_directory() -> List[dict]:
    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():  # Iterate through each Directory.
        doc_type = folder.name
        # recursively find all files matching a pattern in a directory and all its subdirectories
        for file in folder.rglob("*.md"):
            source = file.as_posix()  # Normalizes a path to always use forward slashes
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(
                    {"source": source, "type": doc_type, "content": content}
                )
    return documents


# Chunk Creation Prompt Function
def create_prompt_for_chunk_creation(document: dict) -> str:
    number_of_chunks = (len(document["content"]) // AVG_CHUNK_SIZE) + 1
    prompt = f"""
    You are an expert document splitter that splits documents into chunks.
    You take a document and you split the document into overlapping chunks for a KnowledgeBase.

    The document is from the shared drive of a company called Insurellm.
    The document is of type: {document["type"]}
    The document has been retrieved from: {document["source"]}

    A chatbot will use these chunks to answer questions about the company.
    You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
    This document should probably be split into {number_of_chunks} chunks, but you can have more or less as appropriate.
    There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

    For each chunk, you should provide a headline, a summary, and the original text of the chunk.
    Together your chunks should represent the entire document with overlap.

    Here is the document:

    {document["content"]}

    Respond with the chunks.
    """

    return prompt


# Message Generation Function
def create_messages_for_chunk_creation(document: dict) -> List[dict]:
    return [
        {"role": "system", "content": create_prompt_for_chunk_creation(document)},
    ]


# Process Document Function
def process_document(document: dict) -> List[Document]:
    messages = create_messages_for_chunk_creation(document)
    response = cast(
        ModelResponse,
        completion(
            model=MODEL,
            messages=messages,
            response_format=Chunks,
        ),
    )
    result = cast(Choices, response.choices[0]).message.content or ""
    valid_chunks = Chunks.model_validate_json(result).chunks
    return [chunk.as_document(document) for chunk in valid_chunks]


# Create Chunks For Each Document Function
def create_chunks(documents: List[dict]) -> List[Document]:
    all_chunks = []
    for document in documents:
        chunks = process_document(document)
        all_chunks.extend(chunks)
    return all_chunks


# 4. Create Embedding, Create Vector Database, Store Chunks in Vector Database


# Create Embeddings Function
def create_embeddings(chunks: List[Document]):
    chroma = PersistentClient(
        path=DB_NAME
    )  # ChromaDB client, Set the path to the DB dir.

    # If the collection already exists, delete it to start fresh.
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)

    ids = [str(i) for i in range(len(chunks))]
    page_content = [chunk.page_content for chunk in chunks]
    metas = [chunk.metadata for chunk in chunks]
    emb = OPENAI.embeddings.create(model=EMBEDDING_MODEL, input=page_content).data
    vectors = [e.embedding for e in emb]

    global collection

    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    collection.add(ids=ids, embeddings=vectors, documents=page_content, metadatas=metas)  # type: ignore[arg-type]


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


# Rerank Chunks Function


def rerank_chunks(question, chunks):
    system_prompt = """
    You are a document re-ranker.
    You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
    The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
    You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
    Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
    """
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = cast(
        ModelResponse,
        completion(model=MODEL, messages=messages, response_format=RankOrder),
    )
    reply = cast(Choices, response.choices[0]).message.content or ""
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


def fetch_context_unranked(question):
    query = (
        OPENAI.embeddings.create(model=EMBEDDING_MODEL, input=[question])
        .data[0]
        .embedding
    )
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    docs = results["documents"]
    metas = results["metadatas"]
    assert docs is not None and metas is not None
    for result in zip(docs[0], metas[0]):
        chunks.append(Document(page_content=result[0], metadata=dict(result[1])))
    return chunks


def fetch_context(question):
    chunks = fetch_context_unranked(question)
    return rerank_chunks(question, chunks)


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}"
        for chunk in chunks
    )
    SYSTEM_PROMPT = f"""
    You are a knowledgeable, friendly assistant representing the company Insurellm.
    You are chatting with a user about Insurellm.
    Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
    If you don't know the answer, say so.
    For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
    {context}

    With this context, please answer the user's question. Be accurate, relevant and complete.
    """

    system_prompt = SYSTEM_PROMPT
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = cast(
        ModelResponse,
        completion(model=MODEL, messages=[{"role": "system", "content": message}]),
    )
    return cast(Choices, response.choices[0]).message.content


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    query = rewrite_query(question, history)
    print(query)
    chunks = fetch_context(query)
    messages = make_rag_messages(question, history, chunks)
    response = cast(ModelResponse, completion(model=MODEL, messages=messages))
    return cast(Choices, response.choices[0]).message.content or "", chunks
