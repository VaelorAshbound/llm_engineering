import json
import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a customer support assistant for an airline called FlightAI.
You are tasked with supporting customers with their inquiries about ticket prices.
You have access to a tool that can get the price of a ticket to a city.
The tool is called get_ticket_price and it takes in a city name as an argument and returns the price of a ticket to that city.
When you receive a message from the user,
you should first determine if you need to use the tool to answer the user's question.
If you do, you should call the tool with the appropriate arguments and use the result to formulate your response to the user.
Always be courteous and concise in your responses. If you don't know the answer to a question, say so.
"""

ticket_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a ticket to a city",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            }
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}

tools = [{"type": "function", "function": ticket_function}]

system_message = {"role": "system", "content": SYSTEM_PROMPT}

ticket_prices = {
    "paris": 120,
    "london": 100,
    "new york": 200,
    "tokyo": 300,
}


def get_ticket_prices(destination_city):
    city = destination_city.lower()
    price = ticket_prices.get(city)
    if price:
        return f"The price of a ticket to {destination_city} is ${price}."
    else:
        return f"Sorry, we don't have ticket price information for {destination_city}."


def handle_tool_calls(message):
    responses = []
    cities = []
    for tool_call in message.tool_calls:
        tool_call_function = tool_call.function
        function_name = tool_call_function.name
        function_args = tool_call.arguments
        parsed_args = json.loads(function_args)
        city = parsed_args.get("destination_city")
        cities.append(city)
        if function_name == "get_ticket_price":
            ticket_price = get_ticket_prices(**parsed_args)
            response = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": ticket_price,
            }
            responses.append(response)
    return responses, cities


def artist(city):
    # This is a placeholder function that simulates generating an image based on the city name.
    # In a real implementation, this could be replaced with a call to an image generation model or API.
    return f"Image of {city}"


def chat(history):
    messages = [system_message] + history
    response = OPENAI.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
    )

    cities = []
    image = None

    while response.choices[0].finish_reason == "tool_call":
        message = response.choices[0].message
        messages.append(message)
        responses, cities = handle_tool_calls(message)
        messages.extend(responses)
        response = OPENAI.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools,
        )

    reply = response.choices[0].message.content

    history = history + [{"role": "assistant", "content": reply}]

    if cities:
        image = artist(cities[0])

    return history, None, image


def put_message_in_chatbot(message, chatbot):
    return "", chatbot + [{"role": "user", "content": message}]


with gr.Blocks() as demo:
    with gr.Row():
        chatbot = gr.Chatbot(label="Chat with FlightAI", height=400, type="messages")
    with gr.Row():
        image_output = gr.Image(label="Image Output", height=400)
    with gr.Row():
        audio_output = gr.Audio(label="Audio Output", autoplay=True)
    with gr.Row():
        message = gr.Textbox(label="Enter your message here...")

    message.submit(
        fn=put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
    ).then(chat, inputs=chatbot, outputs=[chatbot, audio_output, image_output])

demo.launch(inbrowser=True, auth=("ed", "bananas"))
