#!/usr/bin/env python
# coding: utf-8

# # Special Extra!!
#
# Getting the best models on the planet to generate an SVG
#
# Inspired by the legendary Simon Willison's Pelican riding a bike test
#
# Key point: this is a very different task to image generation! The model needs to describe the image with lines and shapes.
#
# ### This uses OpenRouter.ai so that we easily access the latest models

# %%


import os
import time
from datetime import datetime

from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI
from revealer import reveal

load_dotenv(override=True)


# %%


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if OPENROUTER_API_KEY and OPENROUTER_API_KEY.startswith("sk-or-"):
    print("OPENROUTER_API_KEY looks good so far")
else:
    print("OPENROUTER_API_KEY doesn't seem right")


# %%


openrouter = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


# %%


challenge = "a panda rollerblading to work"
prompt = f"Generate an SVG of {challenge}. Respond with the SVG only, no code blocks."
messages = [{"role": "user", "content": prompt}]


# %%


def artist(model, effort=None):
    try:
        start = datetime.now()
        response = openrouter.chat.completions.create(
            model=model, messages=messages, reasoning_effort=effort
        )
        result = response.choices[0].message.content
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        heading = (
            f"### {model}\n**Time:** {elapsed // 60:.0f} min {elapsed % 60:.0f} s\n\n"
        )
    except Exception as e:
        print(f"Model {model} failed: {e}")
        heading = f"### {model}\n**Error:** {e}\n\n"
        return heading, None
    return heading, result


# %%


results = [
    artist("openai/gpt-oss-120b"),
    artist("openai/gpt-5-nano", effort="low"),
    artist("deepseek/deepseek-v3.2"),
    artist("moonshotai/kimi-k2-thinking"),
    artist("x-ai/grok-4.1-fast"),
    artist("anthropic/claude-opus-4.5"),
    artist("openai/gpt-5.2", effort="high"),
    artist("google/gemini-3-pro-preview"),
]


# %%


for result in results:
    try:
        display(Markdown(result[0]))
        reveal(result[1])
        time.sleep(12)
    except Exception as e:
        print(f"Error displaying result: {e}")


# ## In Week 4 we will have more scientific ways to compare models..
#
# but this was quite fun.
