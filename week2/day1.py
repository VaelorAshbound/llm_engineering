#!/usr/bin/env python
# coding: utf-8

# # Welcome to Week 2!
#
# ## Frontier Model APIs
#
# In Week 1, we used multiple Frontier LLMs through their Chat UI, and we connected with the OpenAI's API.
#
# Today we'll connect with them through their APIs..

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#900;">Important Note - Please read me</h2>
#             <span style="color:#900;">I'm continually improving these labs, adding more examples and exercises.
#             At the start of each week, it's worth checking you have the latest code.<br/>
#             First do a git pull and merge your changes as needed</a>. Check out the GitHub guide for instructions. Any problems? Try asking ChatGPT to clarify how to merge - or contact me!<br/>
#             </span>
#         </td>
#     </tr>
# </table>
# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/resources.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#f71;">Reminder about the resources page</h2>
#             <span style="color:#f71;">Here's a link to resources for the course. This includes links to all the slides.<br/>
#             <a href="https://edwarddonner.com/2024/11/13/llm-engineering-resources/">https://edwarddonner.com/2024/11/13/llm-engineering-resources/</a><br/>
#             Please keep this bookmarked, and I'll continue to add more useful links there over time.
#             </span>
#         </td>
#     </tr>
# </table>

# ## Setting up your keys - OPTIONAL!
#
# We're now going to try asking a bunch of models some questions!
#
# This is totally optional. If you have keys to Anthropic, Gemini or others, then you can add them in.
#
# If you'd rather not spend the extra, then just watch me do it!
#
# For OpenAI, visit https://openai.com/api/
# For Anthropic, visit https://console.anthropic.com/
# For Google, visit https://aistudio.google.com/
# For DeepSeek, visit https://platform.deepseek.com/
# For Groq, visit https://console.groq.com/
# For Grok, visit https://console.x.ai/
#
#
# You can also use OpenRouter as your one-stop-shop for many of these! OpenRouter is "the unified interface for LLMs":
#
# For OpenRouter, visit https://openrouter.ai/
#
#
# With each of the above, you typically have to navigate to:
# 1. Their billing page to add the minimum top-up (except Gemini, Groq, Google, OpenRouter may have free tiers)
# 2. Their API key page to collect your API key
#
# ### Adding API keys to your .env file
#
# When you get your API keys, you need to set them as environment variables by adding them to your `.env` file.
#
# ```
# OPENAI_API_KEY=xxxx
# ANTHROPIC_API_KEY=xxxx
# GOOGLE_API_KEY=xxxx
# DEEPSEEK_API_KEY=xxxx
# GROQ_API_KEY=xxxx
# GROK_API_KEY=xxxx
# OPENROUTER_API_KEY=xxxx
# ```
#
# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#900;">Any time you change your .env file</h2>
#             <span style="color:#900;">Remember to Save it! And also rerun load_dotenv(override=True)<br/>
#             </span>
#         </td>
#     </tr>
# </table>

# %%


# imports

import os
from typing import Any

import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from google import genai
from IPython.display import Markdown, display
from langchain_openai import ChatOpenAI
from litellm import completion
from openai import OpenAI

# %%


load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
grok_api_key = os.getenv("GROK_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set (and this is optional)")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:2]}")
else:
    print("Google API Key not set (and this is optional)")

if deepseek_api_key:
    print(f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
else:
    print("DeepSeek API Key not set (and this is optional)")

if groq_api_key:
    print(f"Groq API Key exists and begins {groq_api_key[:4]}")
else:
    print("Groq API Key not set (and this is optional)")

if grok_api_key:
    print(f"Grok API Key exists and begins {grok_api_key[:4]}")
else:
    print("Grok API Key not set (and this is optional)")

if openrouter_api_key:
    print(f"OpenRouter API Key exists and begins {openrouter_api_key[:3]}")
else:
    print("OpenRouter API Key not set (and this is optional)")


# %%


# Connect to OpenAI client library
# A thin wrapper around calls to HTTP endpoints

openai = OpenAI()

# For Gemini, DeepSeek and Groq, we can use the OpenAI python client
# Because Google and DeepSeek have endpoints compatible with OpenAI
# And OpenAI allows you to change the base_url

anthropic_url = "https://api.anthropic.com/v1/"
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
deepseek_url = "https://api.deepseek.com"
groq_url = "https://api.groq.com/openai/v1"
grok_url = "https://api.x.ai/v1"
openrouter_url = "https://openrouter.ai/api/v1"
ollama_url = "http://localhost:11434/v1"

anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)
deepseek = OpenAI(api_key=deepseek_api_key, base_url=deepseek_url)
groq = OpenAI(api_key=groq_api_key, base_url=groq_url)
grok = OpenAI(api_key=grok_api_key, base_url=grok_url)
openrouter = OpenAI(base_url=openrouter_url, api_key=openrouter_api_key)
ollama = OpenAI(api_key="ollama", base_url=ollama_url)


# %%


tell_a_joke = [
    {
        "role": "user",
        "content": "Tell a joke for a student on the journey to becoming an expert in LLM Engineering",
    },
]


# %%


response = openai.chat.completions.create(model="gpt-4.1-mini", messages=tell_a_joke)  # type: ignore[arg-type]
display(Markdown(response.choices[0].message.content))


# %%


response = anthropic.chat.completions.create(
    model="claude-sonnet-4-5-20250929",
    messages=tell_a_joke,  # type: ignore[arg-type]
)
display(Markdown(response.choices[0].message.content))


# ## Training vs Inference time scaling

# %%


easy_puzzle = [
    {
        "role": "user",
        "content": "You toss 2 coins. One of them is heads. What's the probability the other is tails? Answer with the probability only.",
    },
]


# %%


# %%


response = openai.chat.completions.create(
    model="gpt-5-nano",
    messages=easy_puzzle,  # type: ignore[arg-type]
    reasoning_effort="minimal",
)
display(Markdown(response.choices[0].message.content))


# %%


response = openai.chat.completions.create(
    model="gpt-5-nano",
    messages=easy_puzzle,  # type: ignore[arg-type]
    reasoning_effort="low",
)
display(Markdown(response.choices[0].message.content))


# %%


response = openai.chat.completions.create(
    model="gpt-5-mini",
    messages=easy_puzzle,  # type: ignore[arg-type]
    reasoning_effort="minimal",
)
display(Markdown(response.choices[0].message.content))


# ## Testing out the best models on the planet

# %%


hard = """
On a bookshelf, two volumes of Pushkin stand side by side: the first and the second.
The pages of each volume together have a thickness of 2 cm, and each cover is 2 mm thick.
A worm gnawed (perpendicular to the pages) from the first page of the first volume to the last page of the second volume.
What distance did it gnaw through?
"""
hard_puzzle = [{"role": "user", "content": hard}]


# %%


response = openai.chat.completions.create(
    model="gpt-5-nano",
    messages=hard_puzzle,  # type: ignore[arg-type]
    reasoning_effort="minimal",
)
display(Markdown(response.choices[0].message.content))


# %%


response = anthropic.chat.completions.create(
    model="claude-sonnet-4-5-20250929",
    messages=hard_puzzle,  # type: ignore[arg-type]
)
display(Markdown(response.choices[0].message.content))


# %%


response = openai.chat.completions.create(model="gpt-5", messages=hard_puzzle)  # type: ignore[arg-type]
display(Markdown(response.choices[0].message.content))


# %%


response = gemini.chat.completions.create(model="gemini-2.5-pro", messages=hard_puzzle)  # type: ignore[arg-type]
display(Markdown(response.choices[0].message.content))


# ## A spicy challenge to test the competitive spirit

# %%


dilemma_prompt = """
You and a partner are contestants on a game show. You're each taken to separate rooms and given a choice:
Cooperate: Choose "Share" — if both of you choose this, you each win $1,000.
Defect: Choose "Steal" — if one steals and the other shares, the stealer gets $2,000 and the sharer gets nothing.
If both steal, you both get nothing.
Do you choose to Steal or Share? Pick one.
"""

dilemma = [
    {"role": "user", "content": dilemma_prompt},
]


# %%


response = anthropic.chat.completions.create(
    model="claude-sonnet-4-5-20250929",
    messages=dilemma,  # type: ignore[arg-type]
)
display(Markdown(response.choices[0].message.content))


# %%


response = groq.chat.completions.create(model="openai/gpt-oss-120b", messages=dilemma)  # type: ignore[arg-type]
display(Markdown(response.choices[0].message.content))


# %%


response = deepseek.chat.completions.create(model="deepseek-reasoner", messages=dilemma)  # type: ignore[arg-type]
display(Markdown(response.choices[0].message.content))


# %%


response = grok.chat.completions.create(model="grok-4", messages=dilemma)  # type: ignore[arg-type]
display(Markdown(response.choices[0].message.content))


# ## Going local
#
# Just use the OpenAI library pointed to localhost:11434/v1

# %%


requests.get("http://localhost:11434/").content

# If not running, run ollama serve at a command line


# %%


os.system("ollama pull llama3.2")


# %%


# Only do this if you have a large machine - at least 16GB RAM

os.system("ollama pull gpt-oss:20b")


# %%


response = ollama.chat.completions.create(model="llama3.2", messages=easy_puzzle)  # type: ignore[arg-type]
display(Markdown(response.choices[0].message.content))


# %%


response = ollama.chat.completions.create(model="gpt-oss:20b", messages=easy_puzzle)  # type: ignore[arg-type]
display(Markdown(response.choices[0].message.content))


# ## Gemini and Anthropic Client Library
#
# We're going via the OpenAI Python Client Library, but the other providers have their libraries too

# %%


client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Describe the color Blue to someone who's never been able to see in 1 sentence",
)
print(response.text)


# %%


client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    messages=[
        {
            "role": "user",
            "content": "Describe the color Blue to someone who's never been able to see in 1 sentence",
        }
    ],
    max_tokens=100,
)
print(response.content[0].text)  # type: ignore[union-attr]


# ## Routers and Abtraction Layers
#
# Starting with the wonderful OpenRouter.ai - it can connect to all the models above!
#
# Visit openrouter.ai and browse the models.
#
# Here's one we haven't seen yet: GLM 4.5 from Chinese startup z.ai

# %%


response = openrouter.chat.completions.create(
    model="z-ai/glm-4.5",
    messages=tell_a_joke,  # type: ignore[arg-type]
)
display(Markdown(response.choices[0].message.content))


# ## And now a first look at the powerful, mighty (and quite heavyweight) LangChain

# %%


llm = ChatOpenAI(model="gpt-5-mini")
response = llm.invoke(tell_a_joke)

display(Markdown(response.content))


# ## Finally - my personal fave - the wonderfully lightweight LiteLLM

# %%


response: Any = completion(model="openai/gpt-4.1", messages=tell_a_joke)  # type: ignore[assignment]
reply = response.choices[0].message.content
display(Markdown(reply))


# %%


print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
print(f"Total cost: {response._hidden_params['response_cost'] * 100:.4f} cents")


# ## Now - let's use LiteLLM to illustrate a Pro-feature: prompt caching

# %%


with open("hamlet.txt", "r", encoding="utf-8") as f:
    hamlet = f.read()

loc = hamlet.find("Speak, man")
print(hamlet[loc : loc + 100])


# %%


question = [
    {
        "role": "user",
        "content": "In Hamlet, when Laertes asks 'Where is my father?' what is the reply?",
    }
]


# %%


response = completion(model="gemini/gemini-2.5-flash-lite", messages=question)  # type: ignore[assignment]
display(Markdown(response.choices[0].message.content))


# %%


print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
print(f"Total cost: {response._hidden_params['response_cost'] * 100:.4f} cents")


# %%


question[0]["content"] += (
    "\n\nFor context, here is the entire text of Hamlet:\n\n" + hamlet
)


# %%


response = completion(model="gemini/gemini-2.5-flash-lite", messages=question)  # type: ignore[assignment]
display(Markdown(response.choices[0].message.content))


# %%


print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
print(f"Total cost: {response._hidden_params['response_cost'] * 100:.4f} cents")


# %%


response = completion(model="gemini/gemini-2.5-flash-lite", messages=question)  # type: ignore[assignment]
display(Markdown(response.choices[0].message.content))


# %%


print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
print(f"Total cost: {response._hidden_params['response_cost'] * 100:.4f} cents")


# ## Prompt Caching with OpenAI
#
# For OpenAI:
#
# https://platform.openai.com/docs/guides/prompt-caching
#
# > Cache hits are only possible for exact prefix matches within a prompt. To realize caching benefits, place static content like instructions and examples at the beginning of your prompt, and put variable content, such as user-specific information, at the end. This also applies to images and tools, which must be identical between requests.
#
#
# Cached input is 4X cheaper
#
# https://openai.com/api/pricing/

# ## Prompt Caching with Anthropic
#
# https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
#
# You have to tell Claude what you are caching
#
# You pay 25% MORE to "prime" the cache
#
# Then you pay 10X less to reuse from the cache with inputs.
#
# https://www.anthropic.com/pricing#api

# ## Gemini supports both 'implicit' and 'explicit' prompt caching
#
# https://ai.google.dev/gemini-api/docs/caching?lang=python

# ## And now for some fun - an adversarial conversation between Chatbots..
#
# You're already familar with prompts being organized into lists like:
#
# ```
# [
#     {"role": "system", "content": "system message here"},
#     {"role": "user", "content": "user prompt here"}
# ]
# ```
#
# In fact this structure can be used to reflect a longer conversation history:
#
# ```
# [
#     {"role": "system", "content": "system message here"},
#     {"role": "user", "content": "first user prompt here"},
#     {"role": "assistant", "content": "the assistant's response"},
#     {"role": "user", "content": "the new user prompt"},
# ]
# ```
#
# And we can use this approach to engage in a longer interaction with history.

# %%


# Let's make a conversation between GPT-4.1-mini and Claude-haiku-4.5
# We're using cheap versions of models so the costs will be minimal

gpt_model = "gpt-4.1-mini"
claude_model = "claude-haiku-4-5"

gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

claude_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

gpt_messages = ["Hi there"]
claude_messages = ["Hi"]


# %%


def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, claude in zip(gpt_messages, claude_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": claude})
    response = openai.chat.completions.create(model=gpt_model, messages=messages)  # type: ignore[arg-type]
    return response.choices[0].message.content


# %%


call_gpt()


# %%


def call_claude():
    messages = [{"role": "system", "content": claude_system}]
    for gpt, claude_message in zip(gpt_messages, claude_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": claude_message})
    messages.append({"role": "user", "content": gpt_messages[-1]})
    response = anthropic.chat.completions.create(model=claude_model, messages=messages)  # type: ignore[arg-type]
    return response.choices[0].message.content


# %%


call_claude()


# %%


call_gpt()


# %%


gpt_messages = ["Hi there"]
claude_messages = ["Hi"]

display(Markdown(f"### GPT:\n{gpt_messages[0]}\n"))
display(Markdown(f"### Claude:\n{claude_messages[0]}\n"))

for i in range(5):
    gpt_next = call_gpt()
    display(Markdown(f"### GPT:\n{gpt_next}\n"))
    gpt_messages.append(gpt_next or "")

    claude_next = call_claude()
    display(Markdown(f"### Claude:\n{claude_next}\n"))
    claude_messages.append(claude_next or "")


# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#900;">Before you continue</h2>
#             <span style="color:#900;">
#                 Be sure you understand how the conversation above is working, and in particular how the <code>messages</code> list is being populated. Add print statements as needed. Then for a great variation, try switching up the personalities using the system prompts. Perhaps one can be pessimistic, and one optimistic?<br/>
#             </span>
#         </td>
#     </tr>
# </table>

# # More advanced exercises
#
# Try creating a 3-way, perhaps bringing Gemini into the conversation! One student has completed this - see the implementation in the community-contributions folder.
#
# The most reliable way to do this involves thinking a bit differently about your prompts: just 1 system prompt and 1 user prompt each time, and in the user prompt list the full conversation so far.
#
# Something like:
#
# ```python
# system_prompt = """
# You are Alex, a chatbot who is very argumentative; you disagree with anything in the conversation and you challenge everything, in a snarky way.
# You are in a conversation with Blake and Charlie.
# """
#
# user_prompt = f"""
# You are Alex, in conversation with Blake and Charlie.
# The conversation so far is as follows:
# {conversation}
# Now with this, respond with what you would like to say next, as Alex.
# """
# ```
#
# Try doing this yourself before you look at the solutions. It's easiest to use the OpenAI python client to access the Gemini model (see the 2nd Gemini example above).
#
# ## Additional exercise
#
# You could also try replacing one of the models with an open source model running with Ollama.

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/business.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#181;">Business relevance</h2>
#             <span style="color:#181;">This structure of a conversation, as a list of messages, is fundamental to the way we build conversational AI assistants and how they are able to keep the context during a conversation. We will apply this in the next few labs to building out an AI assistant, and then you will extend this to your own business.</span>
#         </td>
#     </tr>
# </table>

# %%
