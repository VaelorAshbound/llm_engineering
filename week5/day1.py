#!/usr/bin/env python
# coding: utf-8

# # Welcome to RAG week!!
#
# ## Expert Knowledge Worker
#
# ### A question answering Assistant that is an expert knowledge worker
# ### To be used by employees of Insurellm, an Insurance Tech company
# ### The AI assistant needs to be accurate and the solution should be low cost.
#
# This project will use RAG (Retrieval Augmented Generation) to ensure our question/answering assistant has high accuracy.
#
# This first implementation will use a simplistic, brute-force type of RAG..
#
# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/business.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#181;">Business applications of this week's projects</h2>
#             <span style="color:#181;">RAG is perhaps the most immediately applicable technique of anything that we cover in the course! In fact, there are commercial products that do precisely what we build this week: nuanced querying across large databases of information, such as company contracts or product specs. RAG gives you a quick-to-market, low cost mechanism for adapting an LLM to your business area.</span>
#         </td>
#     </tr>
# </table>

# %%


import glob
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h1 style="color:#900;">Important Note</h1>
#             <span style="color:#900;">
#             This lab, and all the labs for Week 5, has been updated to use LangChain 1.0. This is intended to be reviewed with the new video series as of November 2025. If you're reviewing the older video series, then please consider doing <code>git checkout original</code> to revert to the prior code, then later <code>git checkout main</code> to get back to the new code. I have a really exciting week ahead, with evals and Advanced RAG!
#             </span>
#         </td>
#     </tr>
# </table>

# %%


# Setting up

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4.1-nano"
openai = OpenAI()


# ### Let's read in all employee data into a dictionary

# %%


knowledge = {}

filenames = glob.glob("knowledge-base/employees/*")

for filename in filenames:
    name = Path(filename).stem.split(" ")[-1]
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()


# %%


_ = knowledge


# %%


knowledge["lancaster"]


# %%


filenames = glob.glob("knowledge-base/products/*")

for filename in filenames:
    name = Path(filename).stem
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()


# %%


knowledge.keys()


# %%


SYSTEM_PREFIX = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""


# %%


def get_relevant_context_simple(message):
    text = "".join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    relevant_context = []
    for word in words:
        if word in knowledge:
            relevant_context.append(knowledge[word])
    return relevant_context


# ## But a more pythonic way:

# %%


def get_relevant_context(message):
    text = "".join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    return [knowledge[word] for word in words if word in knowledge]


# %%


get_relevant_context("Who is lancaster?")


# %%


get_relevant_context("Who is Lancaster and what is carllm?")


# %%


def additional_context(message):
    relevant_context = get_relevant_context(message)
    if not relevant_context:
        result = "There is no additional context relevant to the user's question."
    else:
        result = "The following additional context might be relevant in answering the user's question:\n\n"
        result += "\n\n".join(relevant_context)
    return result


# %%


print(additional_context("Who is Alex Lancaster?"))


# %%


def chat(message, history):
    system_message = SYSTEM_PREFIX + additional_context(message)
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


# ## Now we will bring this up in Gradio using the Chat interface -
#
# A quick and easy way to prototype a chat with an LLM

# %%


view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)


# %%
