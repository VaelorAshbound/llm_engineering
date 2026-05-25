from typing import Any, cast

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion
from litellm.types.utils import Choices
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(override=True)
MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI = OpenAI()
DB_NAME = "knowledge_base"
COLLECTION_NAME = "docs"
RETRIEVAL_K = 10
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)


class Result(BaseModel):
    page_content: str = Field(
        description="""
        A chunk of text based on the provided document, that is most likely to be surfaced in a query.
        It includes a headline, a summary, and the original text of the chunk.
        """
    )
    metadata: dict = Field(
        description="Metadata about the chunk, including the source and type of the original document"
    )


def rewrite_query(question, history):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
    You are a knowledge retrieval assistant for a company called Insurellm.
    Your duty is to take the user's question and rewrite it into a more specific query
    that can be used to search a Knowledge Base for relevant information about Insurellm.

    Here is the context to help you rewrite the question:
    1. The Knowledge Base contains documents about Insurellm,
        including information about the company's products, employees, contracts, and other relevant details.
    2. The user's question may be vague or broad,
        and your task is to refine it into a concise and specific query that is more likely to surface relevant content in the Knowledge Base.

    The process for rewriting the question is as follows:
    1. Read the user's question and the history of the conversation to understand the context and the user's intent.
    2. Identify the key details and specific information that the user is seeking about Insurellm.
    3. Rewrite the question to be more specific and focused on the details of the user's question,
        while retaining the original meaning and intent.
    4. Ensure that the rewritten question is concise and clear, making it easier to search the Knowledge Base for relevant information.

    The rules for rewriting the question are as follows:
    1. The rewritten question should be specific and focused on the details of the user's question.
    2. The rewritten question should retain the original meaning and intent of the user's question.
    3. The rewritten question should be concise and clear, avoiding unnecessary words or ambiguity.

    This is the history of your conversation so far with the user:
    {history}
    And this is the user's current question:
    {question}

    IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
    """
    response = cast(
        Any, completion(model=MODEL, messages=[{"role": "system", "content": message}])
    )
    content = cast(Choices, response.choices[0]).message.content
    assert content is not None, "Expected non-None content from completion"
    return content


def fetch_context_unranked(question):
    # One input → one result → grab it directly with `.data[0].embedding`.
    # Result is a **single vector**.
    query = (
        OPENAI.embeddings.create(model=EMBEDDING_MODEL, input=[question])
        .data[0]
        .embedding
    )
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    documents = results["documents"] or []
    metadatas = results["metadatas"] or []
    for result in zip(documents[0], metadatas[0]):
        chunks.append(Result(page_content=result[0], metadata=dict(result[1])))  # type: ignore[arg-type]
    return chunks


def merge_chunks(chunks1, chunks2):
    merged = [chunk.pagecontent for chunk in chunks1]
    for chunk in chunks2:
        if chunk.page_content not in merged:
            merged.append(chunk.page_content)
    return merged


def rerank_chunks(question, chunks):
    system_prompt = """
    You are a document re-ranking assistant for a company called Insurellm.
    Your duty is to take a list of chunks of text and re-rank them based on
    their relevance to the user's question about Insurellm.

    The process for re-ranking the chunks is as follows:
    1. Read the user's question and the list of chunks to understand the context
        and the user's intent.
    2. Evaluate the relevance of each chunk to the user's question, considering
        factors such as the presence of key details, the specificity of the
        information, and the overall relevance to the user's intent.
    3. Re-rank the chunks in order of relevance, with the most relevant chunks
        appearing at the top of the list.

    The rules for re-ranking the chunks are as follows:
    1. The re-ranked list of chunks should be ordered from most relevant to
        least relevant based on the user's question.
    2. The relevance of each chunk should be evaluated based on the presence
        of key details, the specificity of the information, and the overall relevance
        to the user's intent.
    3. The re-ranked list of chunks should provide the most relevant information to
        the user based on their question about Insurellm.

    The re-ranked list of chunks should be returned in a list of ranked chunk ids,
    where the chunk id corresponds to the original index of the chunk in the input list.
    """

    user_prompt = f"""
    Here is the user's question:
    {question}
    Order the following chunks from most relevant to least relevant to the user's question.
    Respond with a list of ranked chunk ids, where the chunk id corresponds to the original index of the chunk in the input list.
    """

    for index, chunk in enumerate(chunks):
        user_prompt += f"\nChunk ID {index + 1}: \n\n{chunk.page_content}\n\n"
    user_prompt += "Repl"


def fetch_context(question, history):
    refined_query = rewrite_query(question, history)
    chunks1 = fetch_context_unranked(question)
    chunks2 = fetch_context_unranked(refined_query)
    merged_chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank_chunks(question, merged_chunks)


def answer_question(question, history):
    context = fetch_context(question, history)
