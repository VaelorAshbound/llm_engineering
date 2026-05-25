# %%
import json
import os
import sqlite3

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=api_key)


DB = "flights.db"

with sqlite3.connect(DB) as conn:
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS prices (destination TEXT PRIMARY KEY, price REAL)"
    )
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS bookings (name TEXT, destination TEXT, date TEXT, amount REAL)"
    )
    conn.commit()


SYSTEM_PROMPT = """
You are a flight assistant for an Airline called FlightAI in New York.

You can perform the following actions:
1. Check ticket prices: Given a route, return the current ticket prices for that route by calling the `check_ticket_prices` function.
2. Book a flight: Given a route, user's name, and date, book a flight for the user by calling the `book_flight` function and return the booking confirmation.

Here are the rules you must follow:
1. If the user asks about ticket prices, call the `check_ticket_prices` function with the appropriate parameters and return the result to the user.
2. If the user wants to book a flight, call the `book_flight` function with the appropriate parameters and return the booking confirmation to the user.
3. Always be accurate and provide clear and concise information to the user. If you don't know the answer to a question, say so.
"""

ticket_prices = {"london": 799, "paris": 899, "tokyo": 1420, "sydney": 2999}


def check_ticket_prices(destination):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT price FROM prices WHERE destination = ?", (destination.lower(),)
        )
        result = cursor.fetchone()
        return result[0] if result else "No price data available for this destination"


def set_ticket_price(destination, price):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO prices (destination, price) VALUES (?, ?) ON CONFLICT(destination) DO UPDATE SET price = ?",
            (destination.lower(), price, price),
        )
        conn.commit()


for destination, price in ticket_prices.items():
    set_ticket_price(destination, price)


def book_flight(name, destination, date):
    amount = check_ticket_prices(destination)
    if not isinstance(amount, (int, float)):
        return f"Sorry, we cannot book a flight to {destination} as we do not operate there."

    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, destination, date FROM bookings WHERE name = ? AND destination = ? AND date = ?",
            (name, destination, date),
        )
        if cursor.fetchone():
            return (
                f"Sorry {name}, You already have a booking for {destination} on {date}."
            )
        cursor.execute(
            "INSERT INTO bookings (name, destination, date, amount) VALUES (?, ?, ?, ?)",
            (name, destination, date, amount),
        )
        conn.commit()

    return f"Flight booked for {name} to {destination} on {date}."


price_function = {
    "name": "check_ticket_prices",
    "description": "Check ticket prices for a given destination",
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": "The destination to check ticket prices for",
            }
        },
        "required": ["destination"],
        "additionalProperties": False,
    },
}

booking_function = {
    "name": "book_flight",
    "description": "Book a flight for a user to a given destination on a given date",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the user booking the flight",
            },
            "destination": {
                "type": "string",
                "description": "The destination to book the flight to",
            },
            "date": {
                "type": "string",
                "description": "The date of the flight in YYYY-MM-DD format",
            },
        },
        "required": ["name", "destination", "date"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": price_function},
    {"type": "function", "function": booking_function},
]


def handle_tool_calls(message):
    response = []
    for tool_call in message.tool_calls:
        function = tool_call.function
        function_name = function.name
        function_arguments = function.arguments
        parsed_arguments = json.loads(function_arguments)
        if function_name == "check_ticket_prices":
            result = check_ticket_prices(**parsed_arguments)
            response.append(
                {"role": "tool", "content": str(result), "tool_call_id": tool_call.id}
            )
        elif function_name == "book_flight":
            result = book_flight(**parsed_arguments)
            response.append(
                {"role": "tool", "content": str(result), "tool_call_id": tool_call.id}
            )
    return response


def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": message}]
    )
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,  # type: ignore[arg-type]
        tools=tools,  # type: ignore[arg-type]
    )

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        messages.append(message)  # type: ignore[arg-type]
        responses = handle_tool_calls(message)
        messages.extend(responses)
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
        )

    return response.choices[0].message.content


gr.ChatInterface(fn=chat, type="messages").launch()
