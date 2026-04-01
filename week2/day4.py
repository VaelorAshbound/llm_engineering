#!/usr/bin/env python
# coding: utf-8

# # Project - Airline AI Assistant
#
# We'll now bring together what we've learned to make an AI Customer Support assistant for an Airline

# %%


import json
import os
import sqlite3

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

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

# As an alternative, if you'd like to use Ollama instead of OpenAI
# Check that Ollama is running for you locally (see week1/day2 exercise) then uncomment these next 2 lines
# MODEL = "llama3.2"
# openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')


# %%


system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""


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


# ## Tools
#
# Tools are an incredibly powerful feature provided by the frontier LLMs.
#
# With tools, you can write a function, and have the LLM call that function as part of its response.
#
# Sounds almost spooky.. we're giving it the power to run code on our machine?
#
# Well, kinda.

# %%


# Let's start by making a useful function

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}


def get_ticket_price(destination_city):  # type: ignore[reportRedefinition]
    print(f"Tool called for city {destination_city}")
    price = ticket_prices.get(destination_city.lower(), "Unknown ticket price")
    return f"The price of a ticket to {destination_city} is {price}"


# %%


get_ticket_price("London")


# %%


# There's a particular dictionary structure that's required to describe our function:

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


# %%


# And this is included in a list of tools:

tools = [{"type": "function", "function": price_function}]


# %%


tools  # type: ignore


# ## Getting OpenAI to use our Tool
#
# There's some fiddly stuff to allow OpenAI "to call our tool"
#
# What we actually do is give the LLM the opportunity to inform us that it wants us to run the tool.
#
# Here's how the new chat function looks:

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

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response = handle_tool_call(message)
        messages.append(message)  # type: ignore[arg-type]
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)  # type: ignore[arg-type]

    return response.choices[0].message.content


# %%


# We have to write that function handle_tool_call:


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    response: dict = {}
    if tool_call.function.name == "get_ticket_price":
        arguments = json.loads(tool_call.function.arguments)
        city = arguments.get("destination_city")
        price_details = get_ticket_price(city)
        response = {
            "role": "tool",
            "content": price_details,
            "tool_call_id": tool_call.id,
        }
    return response


# %%


gr.ChatInterface(fn=chat, type="messages").launch()


# ## Let's make a couple of improvements
#
# Handling multiple tool calls in 1 response
#
# Handling multiple tool calls 1 after another

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

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses = handle_tool_calls(message)
        messages.append(message)  # type: ignore[arg-type]
        messages.extend(responses)
        response = openai.chat.completions.create(model=MODEL, messages=messages)  # type: ignore[arg-type]

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


# %%


DB = "prices.db"

with sqlite3.connect(DB) as conn:
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)"
    )
    conn.commit()


# %%


def get_ticket_price(city):  # type: ignore[reportRedefinition]
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


get_ticket_price("London")


# %%


def set_ticket_price(city, price):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO prices (city, price) VALUES (?, ?) ON CONFLICT(city) DO UPDATE SET price = ?",
            (city.lower(), price, price),
        )
        conn.commit()


# %%


ticket_prices = {"london": 799, "paris": 899, "tokyo": 1420, "sydney": 2999}
for city, price in ticket_prices.items():
    set_ticket_price(city, price)


# %%


get_ticket_price("Tokyo")


# %%


gr.ChatInterface(fn=chat, type="messages").launch()


# ## Exercise
#
# Add a tool to set the price of a ticket!

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/business.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#181;">Business Applications</h2>
#             <span style="color:#181;">Hopefully this hardly needs to be stated! You now have the ability to give actions to your LLMs. This Airline Assistant can now do more than answer questions - it could interact with booking APIs to make bookings!</span>
#         </td>
#     </tr>
# </table>
