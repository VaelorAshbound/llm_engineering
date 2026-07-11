#!/usr/bin/env python
# coding: utf-8

# # The Price is Right
#
# ## Week 8 Order of Play
#
# Day 1: Modal.com and SpecialistAgent
# Day 2: RAG, FrontierAgent, Ensemble Agent
# Day 3: ScannerAgent, MessengerAgent
# Day 4: AutonomousPlannerAgent and DealAgentFramework
# Day 5: The Price Is Right Finale
#
#
# Today we'll build another piece of the puzzle: a ScanningAgent that looks for promising deals by subscribing to RSS feeds.

# %%


import logging
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

from agents.deals import DealSelection, ScrapedDeal

load_dotenv(override=True)
openai = OpenAI()
MODEL = "gpt-5-mini"


# %%


deals = ScrapedDeal.fetch(show_progress=True)


# %%


len(deals)


# %%


deals[10].describe()


# ### We are going to ask GPT-5-mini to summarize deals and identify their price

# %%


SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
"""

USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
You should rephrase the description to be a summary of the product itself, not the terms of the deal.
Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 

Deals:

"""

USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."


# %%


# this makes a suitable user prompt given scraped deals


def make_user_prompt(scraped):
    user_prompt = USER_PROMPT_PREFIX
    user_prompt += "\n\n".join([scrape.describe() for scrape in scraped])
    user_prompt += USER_PROMPT_SUFFIX
    return user_prompt


# %%


# Let's create a user prompt for the deals we just scraped, and look at how it begins

user_prompt = make_user_prompt(deals)
print(user_prompt[:2000])
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_prompt},
]


# %%


response = openai.chat.completions.parse(
    model=MODEL,
    messages=messages,
    response_format=DealSelection,
    reasoning_effort="minimal",
)
results = response.choices[0].message.parsed
results


# %%


for deal in results.deals:
    print(deal.product_description)
    print(deal.price)
    print(deal.url)
    print()


# %%


root = logging.getLogger()
root.setLevel(logging.INFO)


# %%


from agents.scanner_agent import ScannerAgent

# %%


agent = ScannerAgent()
result = agent.scan()


# %%


result


# ### Introducing Pushover
#
# Pushover is a nifty tool for sending Push Notifications to your phone.
#
# It's super easy to set up and install!
#
# Simply visit https://pushover.net/ and click 'Login or Signup' on the top right to sign up for a free account, and create your API keys.
#
# Once you've signed up, on the home screen, click "Create an Application/API Token", and give it any name (like AIEngineer) and click Create Application.
#
# Then add 2 lines to your `.env` file:
#
# PUSHOVER_USER=_put the key that's on the top right of your Pushover home screen and probably starts with a u_
# PUSHOVER_TOKEN=_put the key when you click into your new application called Agents (or whatever) and probably starts with an a_
#
# Remember to save your `.env` file, and run `load_dotenv(override=True)` after saving, to set your environment variables.
#
# Finally, click "Add Phone, Tablet or Desktop" to install on your phone.

# %%


load_dotenv(override=True)


# %%


pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


# %%


if pushover_user:
    print(f"Pushover user found and starts with {pushover_user[0]}")
else:
    print("Pushover user not found")

if pushover_token:
    print(f"Pushover token found and starts with {pushover_token[0]}")
else:
    print("Pushover token not found")


# %%


def push(message):
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)


# %%


push("MASSIVE DEAL!!")


# %%


from agents.messaging_agent import MessagingAgent

agent = MessagingAgent()
agent.push("SUCH A MASSIVE DEAL!!")


# %%


agent.notify(
    "A special deal on Sumsung 60 inch LED TV going at a great bargain",
    300,
    1000,
    "www.samsung.com",
)


# %%
