# %%

import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are a helpful assistant that provides information about Earth.
If the user talks about Mars, make jokes about mars and how it is just a barren wasteland and not as good as Earth.
If the user talks about the moon, make jokes about how the moon is just a rock and not as good as Earth.
"""


def chat(message, history):
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": message}]
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )
    return response.choices[0].message.content


view = gr.ChatInterface(fn=chat, type="messages")
view.launch()
