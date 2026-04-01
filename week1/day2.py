#!/usr/bin/env python
# coding: utf-8

# # Welcome to the Day 2 Lab!
#

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/resources.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#f71;">Just before we get started --</h2>
#             <span style="color:#f71;">I thought I'd take a second to point you at this page of useful resources for the course. This includes links to all the slides.<br/>
#             <a href="https://edwarddonner.com/2024/11/13/llm-engineering-resources/">https://edwarddonner.com/2024/11/13/llm-engineering-resources/</a><br/>
#             Please keep this bookmarked, and I'll continue to add more useful links there over time.
#             </span>
#         </td>
#     </tr>
# </table>

# ## First - let's talk about the Chat Completions API
#
# 1. The simplest way to call an LLM
# 2. It's called Chat Completions because it's saying: "here is a conversation, please predict what should come next"
# 3. The Chat Completions API was invented by OpenAI, but it's so popular that everybody uses it!
#
# ### We will start by calling OpenAI again - but don't worry non-OpenAI people, your time is coming!
#

# %%


import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print(
        "No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!"
    )
elif not api_key.startswith("sk-proj-"):
    print(
        "An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook"
    )
else:
    print("API key found and looks good so far!")


# ## Do you know what an Endpoint is?
#
# If not, please review the Technical Foundations guide in the guides folder
#
# And, here is an endpoint that might interest you...

# %%


headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

payload = {
    "model": "gpt-5-nano",
    "messages": [{"role": "user", "content": "Tell me a fun fact"}],
}

print(payload)


# %%


response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
)

response.json()


# %%


response.json()["choices"][0]["message"]["content"]


# # What is the openai package?
#
# It's known as a Python Client Library.
#
# It's nothing more than a wrapper around making this exact call to the http endpoint.
#
# It just allows you to work with nice Python code instead of messing around with janky json objects.
#
# But that's it. It's open-source and lightweight. Some people think it contains OpenAI model code - it doesn't!
#

# %%


# Create OpenAI client

openai = OpenAI()

response = openai.chat.completions.create(
    model="gpt-5-nano", messages=[{"role": "user", "content": "Tell me a fun fact"}]
)

response.choices[0].message.content


# ## And then this great thing happened:
#
# OpenAI's Chat Completions API was so popular, that the other model providers created endpoints that are identical.
#
# They are known as the "OpenAI Compatible Endpoints".
#
# For example, google made one here: https://generativelanguage.googleapis.com/v1beta/openai/
#
# And OpenAI decided to be kind: they said, hey, you can just use the same client library that we made for GPT. We'll allow you to specify a different endpoint URL and a different key, to use another provider.
#
# So you can use:
#
# ```python
# gemini = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key="AIz....")
# gemini.chat.completions.create(...)
# ```
#
# And to be clear - even though OpenAI is in the code, we're only using this lightweight python client library to call the endpoint - there's no OpenAI model involved here.
#
# If you're confused, please review Guide 9 in the Guides folder!
#
# And now let's try it!
#
# ## THIS IS OPTIONAL - but if you wish to try out Google Gemini, please visit:
#
# https://aistudio.google.com/
#
# And set up your API key at
#
# https://aistudio.google.com/api-keys
#
# And then add your key to the `.env` file, being sure to Save the .env file after you change it:
#
# `GOOGLE_API_KEY=AIz...`
#

# %%


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

load_dotenv(override=True)

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print(
        "No API key was found - please be sure to add your key to the .env file, and save the file! Or you can skip the next 2 cells if you don't want to use Gemini"
    )
elif not google_api_key.startswith("AIz"):
    print("An API key was found, but it doesn't start AIz")
else:
    print("API key found and looks good so far!")


# %%


gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)

response = gemini.chat.completions.create(
    model="gemini-2.5-flash-lite",
    messages=[{"role": "user", "content": "Tell me a fun fact"}],
)

response.choices[0].message.content


# ## And Ollama also gives an OpenAI compatible endpoint
#
# ...and it's on your local machine!
#
# If the next cell doesn't print "Ollama is running" then please open a terminal and run `ollama serve`

# %%


requests.get("http://localhost:11434").content


# ### Download llama3.2 from meta
#
# Change this to llama3.2:1b if your computer is smaller.
#
# Don't use llama3.3 or llama4! They are too big for your computer..

# %%


os.system("ollama pull llama3.2")


# %%


OLLAMA_BASE_URL = "http://localhost:11434/v1"

ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


# %%


# Get a fun fact

response = ollama.chat.completions.create(
    model="llama3.2", messages=[{"role": "user", "content": "Tell me a fun fact"}]
)

response.choices[0].message.content


# %%


# Now let's try deepseek-r1:1.5b - this is DeepSeek "distilled" into Qwen from Alibaba Cloud

os.system("ollama pull deepseek-r1:1.5b")


# %%


response = ollama.chat.completions.create(
    model="deepseek-r1:1.5b",
    messages=[{"role": "user", "content": "Tell me a fun fact"}],
)

response.choices[0].message.content


# # HOMEWORK EXERCISE ASSIGNMENT
#
# Upgrade the day 1 project to summarize a webpage to use an Open Source model running locally via Ollama rather than OpenAI
#
# You'll be able to use this technique for all subsequent projects if you'd prefer not to use paid APIs.
#
# **Benefits:**
# 1. No API charges - open-source
# 2. Data doesn't leave your box
#
# **Disadvantages:**
# 1. Significantly less power than Frontier Model
#
# ## Recap on installation of Ollama
#
# Simply visit [ollama.com](https://ollama.com) and install!
#
# Once complete, the ollama server should already be running locally.
# If you visit:
# [http://localhost:11434/](http://localhost:11434/)
#
# You should see the message `Ollama is running`.
#
# If not, bring up a new Terminal (Mac) or Powershell (Windows) and enter `ollama serve`
# And in another Terminal (Mac) or Powershell (Windows), enter `ollama pull llama3.2`
# Then try [http://localhost:11434/](http://localhost:11434/) again.
#
# If Ollama is slow on your machine, try using `llama3.2:1b` as an alternative. Run `ollama pull llama3.2:1b` from a Terminal or Powershell, and change the code from `MODEL = "llama3.2"` to `MODEL = "llama3.2:1b"`

# %%
