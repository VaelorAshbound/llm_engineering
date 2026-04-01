#!/usr/bin/env python
# coding: utf-8

# # Project - Airline AI Assistant
#
# We'll now bring together what we've learned to make an AI Customer Support assistant for an Airline

# %%


# imports

import base64
import json
import os
import sqlite3
from io import BytesIO

import gradio as gr
from dotenv import load_dotenv
from IPython.display import display
from openai import OpenAI
from PIL import Image

# %%


# Initialization

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4.1-mini"
openai = OpenAI()

DB = "prices.db"


# %%


system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""


# %%


def get_ticket_price(city):
    print(f"DATABASE TOOL CALLED: Getting price for {city}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT price FROM prices WHERE city = ?", (city.lower(),))
        result = cursor.fetchone()
        return (
            f"Ticket price to {city} is ${result[0]}"
            if result
            else "No price data available for this city"
        )


# %%


get_ticket_price("Paris")


# %%


price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}
tools = [{"type": "function", "function": price_function}]
tools  # type: ignore # noqa


# %%


def chat(message, history):  # type: ignore[reportRedefinition]
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )
    response = openai.chat.completions.create(model=MODEL, messages=messages)  # type: ignore[arg-type]
    return response.choices[0].message.content


gr.ChatInterface(fn=chat, type="messages").launch()


# %%


def chat(message, history):  # type: ignore[reportRedefinition]
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,  # type: ignore[arg-type]
        tools=tools,  # type: ignore[arg-type]
    )

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses = handle_tool_calls(message)
        messages.append(message)  # type: ignore[arg-type]
        messages.extend(responses)
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
        )

    return response.choices[0].message.content


# %%


def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get("destination_city")
            price_details = get_ticket_price(city)
            responses.append(
                {"role": "tool", "content": price_details, "tool_call_id": tool_call.id}
            )
    return responses


# %%


gr.ChatInterface(fn=chat, type="messages").launch()


# ## A bit more about what Gradio actually does:
#
# 1. Gradio constructs a frontend Svelte app based on our Python description of the UI
# 2. Gradio starts a server built upon the Starlette web framework listening on a free port that serves this React app
# 3. Gradio creates backend routes for our callbacks, like chat(), which calls our functions
#
# And of course when Gradio generates the frontend app, it ensures that the the Submit button calls the right backend route.
#
# That's it!
#
# It's simple, and it has a result that feels magical.

#

# # Let's go multi-modal!!
#
# We can use DALL-E-3, the image generation model behind GPT-4o, to make us some images
#
# Let's put this in a function called artist.
#
# ### Price alert: each time I generate an image it costs about 4 cents - don't go crazy with images!

# %%


def artist(city):
    image_response = openai.images.generate(
        model="dall-e-3",
        prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    assert image_response.data is not None
    image_base64 = image_response.data[0].b64_json
    assert image_base64 is not None
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


# %%


image = artist("New York City")
display(image)


# %%


def talker(message):
    response = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",  # Also, try replacing onyx with alloy or coral
        input=message,
    )
    return response.content


# ## Let's bring this home:
#
# 1. A multi-modal AI assistant with image and audio generation
# 2. Tool callling with database lookup
# 3. A step towards an Agentic workflow
#

# %%


def chat(history):  # type: ignore[reportRedefinition]
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,  # type: ignore[arg-type]
        tools=tools,  # type: ignore[arg-type]
    )
    cities = []
    image = None

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses, cities = handle_tool_calls_and_return_cities(message)
        messages.append(message)  # type: ignore[arg-type]
        messages.extend(responses)
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
        )

    reply = response.choices[0].message.content
    history += [{"role": "assistant", "content": reply}]

    voice = talker(reply)

    if cities:
        image = artist(cities[0])

    return history, voice, image


# %%


def handle_tool_calls_and_return_cities(message):
    responses = []
    cities = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get("destination_city")
            cities.append(city)
            price_details = get_ticket_price(city)
            responses.append(
                {"role": "tool", "content": price_details, "tool_call_id": tool_call.id}
            )
    return responses, cities


# ## The 3 types of Gradio UI
#
# `gr.Interface` is for standard, simple UIs
#
# `gr.ChatInterface` is for standard ChatBot UIs
#
# `gr.Blocks` is for custom UIs where you control the components and the callbacks

# %%


# Callbacks (along with the chat() function above)


def put_message_in_chatbot(message, history):
    return "", history + [{"role": "user", "content": message}]


# UI definition

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500, interactive=False)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        message = gr.Textbox(label="Chat with our AI Assistant:")

    # Hooking up events to callbacks

    message.submit(
        put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
    ).then(chat, inputs=chatbot, outputs=[chatbot, audio_output, image_output])

ui.launch(inbrowser=True, auth=("ed", "bananas"))


# # Exercises and Business Applications
#
# Add in more tools - perhaps to simulate actually booking a flight. A student has done this and provided their example in the community contributions folder.
#
# Next: take this and apply it to your business. Make a multi-modal AI assistant with tools that could carry out an activity for your work. A customer support assistant? New employee onboarding assistant? So many possibilities! Also, see the week2 end of week Exercise in the separate Notebook.

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/thankyou.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#090;">I have a special request for you</h2>
#             <span style="color:#090;">
#                 My editor tells me that it makes a HUGE difference when students rate this course on Udemy - it's one of the main ways that Udemy decides whether to show it to others. If you're able to take a minute to rate this, I'd be so very grateful! And regardless - always please reach out to me at ed@edwarddonner.com if I can help at any point.
#             </span>
#         </td>
#     </tr>
# </table>
