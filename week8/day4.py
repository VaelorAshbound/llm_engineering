#!/usr/bin/env python
# coding: utf-8

# # The Price is Right
#
# ## Week 8 Order of Play
#
# Day 1: Modal.com and SpecialistAgent
# Day 2: RAG, FrontierAgent, Ensemble Agent
# Day 3: ScannerAgent, MessengerAgent
# Day 4: AutonomousPlannerAgent
# Day 5: The Price Is Right Finale
#
#
# Now it's time for the Planning Agent

# %%


import json
import logging

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from agents.scanner_agent import ScannerAgent

load_dotenv(override=True)
openai = OpenAI()
MODEL = "gpt-5.1"


# ## Start with some test data

# %%


test_results = ScannerAgent().test_scan()
test_results


# ## Now let's create 3 pretend functions..

# %%


def scan_the_internet_for_bargains() -> str:
    """This tool scans the internet for great deals and gets a curated list of promising deals"""
    print("Fake function to scan the internet - this returns a hardcoded set of deals")
    return test_results.model_dump_json()


# %%


def estimate_true_value(description: str) -> str:
    """
    This tool estimates the true value of a product based on a text description of it
    """
    print(
        f"Fake function to estimating true value of {description[:20]}... - this always returns $300"
    )
    return f"Product {description} has an estimated true value of $300"


# %%


def notify_user_of_deal(
    description: str, deal_price: float, estimated_true_value: float, url: str
) -> str:
    """
    This tool notifies the user of a great deal, given a description of it, the price of the deal, and the estimated true value
    """
    print(
        f"Fake function to notify user of {description} which costs {deal_price} and estimate is {estimated_true_value}"
    )
    return "notification sent ok"


# ### Let's try them out

# %%


notify_user_of_deal("a new iphone", 100, 1000, "https://www.apple.com/iphone")


# ### OK now a big block of JSON

# %%


scan_function = {
    "name": "scan_the_internet_for_bargains",
    "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}

estimate_function = {
    "name": "estimate_true_value",
    "description": "Given the description of an item, estimate how much it is actually worth",
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The description of the item to be estimated",
            },
        },
        "required": ["description"],
        "additionalProperties": False,
    },
}

notify_function = {
    "name": "notify_user_of_deal",
    "description": "Send the user a push notification about the single most compelling deal; only call this one time",
    "parameters": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "The description of the item itself scraped from the internet",
            },
            "deal_price": {
                "type": "number",
                "description": "The price offered by this deal scraped from the internet",
            },
            "estimated_true_value": {
                "type": "number",
                "description": "The estimated actual value that this is worth",
            },
            "url": {
                "type": "string",
                "description": "The URL of this deal as scraped from the internet",
            },
        },
        "required": ["description", "deal_price", "estimated_true_value", "url"],
        "additionalProperties": False,
    },
}


# %%


tools = [
    {"type": "function", "function": scan_function},
    {"type": "function", "function": estimate_function},
    {"type": "function", "function": notify_function},
]


# %%


tools


# %%


def handle_tool_call(message):
    """
    Actually call the tools associated with this message
    """
    results = []
    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append(
            {
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id,
            }
        )
    return results


# %%


system_message = "You find great deals on bargain products using your tools, and notify the user of the best bargain."
user_message = """
First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
Then just reply OK to indicate success.
"""
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},
]


# %%


messages


# %%


done = False
while not done:
    response = openai.chat.completions.create(
        model=MODEL, messages=messages, tools=tools
    )
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        results = handle_tool_call(message)
        messages.append(message)
        messages.extend(results)
    else:
        done = True
response.choices[0].message.content


# ## And now.. into an Autonomous Planning Agent
#
# And switching the fake functions for REAL functions!

# %%


root = logging.getLogger()
root.setLevel(logging.INFO)


# %%


DB = "products_vectorstore"
client = chromadb.PersistentClient(path=DB)
collection = client.get_or_create_collection("products")


# %%


from agents.autonomous_planning_agent import AutonomousPlanningAgent

agent = AutonomousPlanningAgent(collection)


# %%


agent.plan()


# %%
