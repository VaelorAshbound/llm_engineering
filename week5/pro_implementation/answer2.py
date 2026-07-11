# %%
import os
from pathlib import Path
from typing import Literal, cast

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion, embedding
from litellm.types.utils import Choices, ModelResponse
from pydantic import BaseModel, Field

load_dotenv()

try:
    DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
except NameError:
    DB_NAME = str(Path.cwd().parent / "preprocessed_db")

COLLECTION_NAME = "docs"

chroma_client = PersistentClient(path=DB_NAME)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

MODEL = "openrouter/openai/gpt-4.1-nano"
EMBEDDING_MODEL = "openai/text-embedding-3-large"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
K_RETRIEVAL = 20
K_FINAL = 10

SYSTEM_PROMPT = """
    You are a retrieval query optimizer for InsureLLM, an insurance technology company.

    GOAL
    Rewrite the user's latest query into a single, self-contained search query that maximizes relevant chunk retrieval from the InsureLLM knowledge base.

    KNOWLEDGE BASE DOMAINS
    The knowledge base contains four domains. Use this to guide phrasing:
    - Company     → corporate information, mission, office locations, company history
    - Contracts   → client contracts, terms and conditions, clauses, renewals, SLAs, pricing agreements
    - Employees   → staff profiles, roles, departments, team structures, points of contact
    - Products    → insurance products, policy types, coverage details, premiums, features, eligibility

    RULES
    - Resolve all pronouns and vague references using the conversation history (e.g. "it", "that plan", "her", "the contract").
    - Identify which KB domain the query most likely targets and use domain-appropriate terminology in the rewrite.
    - Expand insurance-specific shorthand when clear from context (e.g. "E&O" → "errors and omissions liability coverage").
    - Preserve the user's intent exactly — do not broaden, narrow, or assume unstated scope.
    - Prefer specific, noun-rich phrasing over conversational language.
    - Keep the rewritten query to one or two sentences maximum.
    - Output the rewritten query only — no explanation, no preamble, no quotes.
"""

USER_PROMPT = """
    CONVERSATION HISTORY
    {conversation_history}

    LATEST USER QUERY
    {user_query}

    TASK
    Rewrite the latest user query into an optimized InsureLLM knowledge base search query, resolving any references from the conversation history above.
"""

RERANK_SYSTEM_PROMPT = """
    You are a context reranker for InsureLLM, an insurance technology company.

    GOAL
    Given a user query and a set of retrieved knowledge base chunks, rank the chunks from most to least useful for answering the query.

    KNOWLEDGE BASE DOMAINS
    - Company     → corporate information, mission, office locations, company history
    - Contracts   → client contracts, terms and conditions, clauses, renewals, SLAs, pricing agreements
    - Employees   → staff profiles, roles, departments, team structures, points of contact
    - Products    → insurance products, policy types, coverage details, premiums, features, eligibility

    SCORING CRITERIA
    Assign each chunk one of three relevance scores:
    - high   → directly addresses the query; contains specific facts, names, values, or clauses the user is asking about
    - medium → provides useful background or partial information; same domain but not a direct answer
    - low    → topically adjacent but unlikely to contribute a meaningful answer

    RULES
    - Evaluate each chunk independently against the query before ranking.
    - Prefer domain match: a chunk from the wrong KB domain should score lower even if superficially related.
    - Prefer specificity: a chunk with exact figures, dates, names, or clause text outranks a general summary.
    - Do not rewrite, summarize, or alter chunk content.
    - Output valid JSON only — no explanation, no markdown, no preamble.
"""

RERANK_USER_PROMPT = """
    USER QUERY
    {query}

    RETRIEVED CHUNKS
    {chunks}

    TASK
    Rank the chunks above from most to least relevant for answering the query.
    Return a JSON object whose "reranks" array is ordered by score:

    {{
      "reranks": [
        {{ "id": 1, "score": "high" | "medium" | "low" }},
        ...
      ]
    }}
"""

RAG_SYSTEM_PROMPT = """
    You are a knowledge base assistant for InsureLLM, an insurance technology company.
    You answer questions from employees and authorized users using the retrieved context below.

    KNOWLEDGE BASE DOMAINS
    - Company     → corporate information, mission, office locations, company history
    - Contracts   → client contracts, terms and conditions, clauses, renewals, SLAs, pricing agreements
    - Employees   → staff profiles, roles, departments, team structures, points of contact
    - Products    → insurance products, policy types, coverage details, premiums, features, eligibility

    RULES
    - Ground every factual claim in the retrieved context. Do not invent names, figures, dates, clauses, or policy details that are not present in the context.
    - Answer the question as fully and completely as the context allows — include every relevant fact from the context that helps address the question. Do not omit details that are present.
    - When the context contains specific facts — names, numbers, dates, or clause text — quote them precisely.
    - Only state that the information is unavailable if the context genuinely contains nothing relevant to the question. If the context is partially relevant, answer with what is available rather than abstaining.
    - Cite the source of your answer when helpful, using the CHUNK SOURCE values from the context.
    - If the context is conflicting or ambiguous, point that out instead of silently choosing one version.
    - Use the conversation history to resolve references, but ground every factual claim in the retrieved context.

    RETRIEVED CONTEXT
    Each chunk below is labelled with its type and source.
    {context}
"""


class Metadata(BaseModel):
    type: str = Field(description="")
    source: str = Field(description="")


class Results(BaseModel):
    page_content: str = Field(description="")
    metadata: Metadata = Field(description="")


class Rerank(BaseModel):
    id: int = Field(description="Chunk ID")
    score: Literal["high", "medium", "low"] = Field(description="Relevance score")


class Reranks(BaseModel):
    reranks: list[Rerank]


def fetch_context_unranked(question: str):
    """
    EmbeddingResponse
    │
    ├── .data ──────────► [  ← a LIST (one entry per input string)
    │                        {
    │                          "object":    "embedding",
    │                          "index":     0,
    │                          "embedding": [0.0123, -0.0456, 0.0789, ... ]   ← ~3072 floats
    │                        }
    │                      ]
    │
    ├── .model   = "text-embedding-3-large"
    ├── .object  = "list"
    └── .usage   = { "prompt_tokens": 7, "total_tokens": 7 }

    embedding(...).data[0]["embedding"]
                  └──┬─┘└┬┘└────┬─────┘
                     │   │      └── pull the float list out of that dict
                     │   └───────── first (and only) result, since you sent 1 string
                     └───────────── the list of results


    """
    query = embedding(
        model=EMBEDDING_MODEL,
        input=question,
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_BASE_URL,
    ).data[0]["embedding"]

    response = collection.query(query_embeddings=[query], n_results=K_RETRIEVAL)
    documents = (response["documents"] or [[]])[0]
    metadatas = (response["metadatas"] or [[]])[0]

    return [
        Results(page_content=document, metadata=Metadata.model_validate(metadata))
        for document, metadata in zip(documents, metadatas)
    ]


def fetch_context(question: str, history: list[str] = []):
    rewritten_question = rewrite_query(question, history)
    context_1 = fetch_context_unranked(question)
    context_2 = fetch_context_unranked(rewritten_question)
    merged_chunks = merge_chunks(context_1, context_2)
    reranked = rerank(question, merged_chunks)
    top_chunks = [
        merged_chunks[r.id - 1] for r in reranked if 0 < r.id <= len(merged_chunks)
    ]
    return top_chunks[:K_FINAL]


def rewrite_query(question: str, history: list[str]):
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT.format(conversation_history=history, user_query=question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = cast(
        ModelResponse, completion(model=MODEL, messages=messages, temperature=0)
    )
    content = cast(Choices, response.choices[0]).message.content or ""
    return content


def rerank(question: str, chunks):
    chunk_content = "\n"
    for index, chunk in enumerate(chunks):
        chunk_content += (
            f"\n\n CHUNK ID: {index + 1}\n\nCHUNK CONTENT:\n{chunk.page_content}\n\n"
        )
    user_prompt = RERANK_USER_PROMPT.format(query=question, chunks=chunk_content)
    messages = [
        {"role": "system", "content": RERANK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response = cast(
        ModelResponse,
        completion(
            model=MODEL, messages=messages, response_format=Reranks, temperature=0
        ),
    )
    content = cast(Choices, response.choices[0]).message.content or ""
    valid_content = Reranks.model_validate_json(content).reranks

    high = []
    medium = []
    low = []
    for rerank in valid_content:
        if rerank.score == "high":
            high.append(rerank)
        elif rerank.score == "medium":
            medium.append(rerank)
        else:
            low.append(rerank)
    # Keep low chunks as a fallback tail so we always have enough to fill K_FINAL.
    new_ranks = high + medium + low
    return new_ranks


def merge_chunks(chunks1, chunks2):
    new_chunks = []
    seen_content = set()
    for chunk in chunks1 + chunks2:
        if chunk.page_content not in seen_content:
            seen_content.add(chunk.page_content)
            new_chunks.append(chunk)
    return new_chunks


def make_rag_message(question, history, chunks):
    context = "\n"
    for chunk in chunks:
        context += f"\n\n CHUNK TYPE: {chunk.metadata.type}\nCHUNK SOURCE: {chunk.metadata.source}\nCHUNK TEXT:\n{chunk.page_content}"
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


def answer_question(question: str, history: list[str] = []):
    top_chunks = fetch_context(question, history)
    messages = make_rag_message(question, history, top_chunks)
    response = cast(
        ModelResponse, completion(model=MODEL, messages=messages, temperature=0)
    )
    content = cast(Choices, response.choices[0]).message.content or ""
    return content, top_chunks
