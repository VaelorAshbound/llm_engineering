#!/usr/bin/env python
# coding: utf-8

# ## RAG Day 3
#
# ### Expert Question Answerer for InsureLLM
#
# LangChain 1.0 implementation of a RAG pipeline.
#
# Using the VectorStore we created last time (with HuggingFace `all-MiniLM-L6-v2`)

# %%


import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# %%


MODEL = "gpt-4.1-nano"
DB_NAME = "vector_db"
load_dotenv(override=True)


# ### Connect to Chroma; use Hugging Face all-MiniLM-L6-v2

# %%


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)


# ### Set up the 2 key LangChain objects: retriever and llm
#
# #### A sidebar on "temperature":
# - Controls how diverse the output is
# - A temperature of 0 means that the output should be predictable
# - Higher temperature for more variety in answers
#
# Some people describe temperature as being like 'creativity' but that's not quite right
# - It actually controls which tokens get selected during inference
# - temperature=0 means: always select the token with highest probability
# - temperature=1 usually means: a token with 10% probability should be picked 10% of the time
#
# Note: a temperature of 0 doesn't mean outputs will always be reproducible. You also need to set a random seed. We will do that in weeks 6-8. (Even then, it's not always reproducible.)
#
# Note 2: if you want creativity, use the System Prompt!

# %%


retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model=MODEL)


# ### These LangChain objects implement the method `invoke()`

# %%


retriever.invoke("Who is Avery?")


# %%


llm.invoke("Who is Avery?")


# ## Time to put this together!

# %%


SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""


# %%


def answer_question(question: str, history):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    )
    return response.content


# %%


answer_question("Who is Averi Lancaster?", [])


# ## What could possibly come next? 😂

# %%


gr.ChatInterface(answer_question).launch()


# ## Admit it - you thought RAG would be more complicated than that!!
