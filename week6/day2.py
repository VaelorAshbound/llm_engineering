#!/usr/bin/env python
# coding: utf-8

# # "THE PRICE IS RIGHT" Capstone Project
#
# This week - build a model that predicts how much something costs from a description, based on a scrape of Amazon data
#
#
# A model that can estimate how much something costs, from its description.
#
# # Order of play
#
# DAY 1: Data Curation
# DAY 2: Data Pre-processing
# DAY 3: Evaluation, Baselines, Traditional ML
# DAY 4: Deep Learning and LLMs
# DAY 5: Fine-tuning a Frontier Model
#
# ## DAY 2: Data Pre-processing
#
# Today we'll rewrite the products into a standard format.
# LLMs are great at this!
#

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/business.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#181;">Business value of Data Pre-processing / Re-writing</h2>
#             <span style="color:#181;">LLMs have made it simple to do something that was considered impossible only a few years ago.
#             This approach can be applied to almost any business vertical, and it's similar to the advanced techniques
#             we used on Week 5.</span>
#         </td>
#     </tr>
# </table>

# %%


import json
import os
from typing import Any, cast

from dotenv import load_dotenv
from groq import Groq
from litellm import completion
from pricer.batch import Batch
from pricer.items import Item

load_dotenv(override=True)


# # The next cell is where you choose Dataset
#
# Use `LITE_MODE = True` for the free, fast version with training data size of 20,000
#
# USe `LITE_MODE =  False` for the powerful, full version with training data size of 800,000
#
# ## For this lab
#
# You can skip altogether and load the dataset from HuggingFace: $0
#
# You can run pre-processing for the lite dataset: under $1
#
# You can run pre-processing for the full dataset: $30

# %%


LITE_MODE = True


# %%


username = "ed-donner"
dataset = f"{username}/items_raw_lite" if LITE_MODE else f"{username}/items_raw_full"

train, val, test = Item.from_hub(dataset)

items = train + val + test

print(f"Loaded {len(items):,} items")
print(items[0])


# %%


items[2].id


# %%


# Give every item an id

for index, item in enumerate(items):
    item.id = index


# %%


SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


# %%


print(items[0].full)


# %%


messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": items[0].full},
]
response = cast(
    Any,
    completion(
        messages=messages, model="groq/openai/gpt-oss-20b", reasoning_effort="low"
    ),
)

print(response.choices[0].message.content)
print()
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Cost: {response._hidden_params['response_cost'] * 100:.3f} cents")


# %%


messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": items[0].full},
]
response = cast(
    Any,
    completion(
        messages=messages, model="ollama/llama3.2", api_base="http://localhost:11434"
    ),
)
print(response.choices[0].message.content)
print()
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Cost: {response._hidden_params['response_cost'] * 100:.3f} cents")


# %%


MODEL = "openai/gpt-oss-20b"


# %%


def make_jsonl(item):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.full},
        ],
        "reasoning_effort": "low",
    }
    line = {
        "custom_id": str(item.id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
    return json.dumps(line)


# %%


items[0]


# %%


make_jsonl(items[0])


# %%


def make_file(start, end, filename):
    batch_file = filename
    with open(batch_file, "w") as f:
        for i in range(start, end):
            f.write(make_jsonl(items[i]))
            f.write("\n")


# %%


make_file(0, 1000, "jsonl/0_1000.jsonl")


# %%


groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# %%


with open("jsonl/0_1000.jsonl", "rb") as f:
    response = groq.files.create(file=f, purpose="batch")
print(response)


# %%


file_id = response.id
assert file_id is not None, "File ID must not be None"
print(file_id)


# %%


response = cast(
    Any,
    groq.batches.create(
        completion_window="24h", endpoint="/v1/chat/completions", input_file_id=file_id
    ),
)
print(response)


# %%


result = groq.batches.retrieve(response.id)
print(result)


# %%


assert result.output_file_id is not None, "Output file ID must not be None"
response = groq.files.content(result.output_file_id)
response.write_to_file("jsonl/batch_results.jsonl")


# %%


with open("jsonl/batch_results.jsonl", "r") as f:
    for line in f:
        json_line = json.loads(line)
        id = int(json_line["custom_id"])
        summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
        items[id].summary = summary


# %%


print(items[0].full)


# %%


print(items[1000].summary)


# ## I've put exactly this logic into a Batch class
#
# - Divides items into groups of 1,000
# - Kicks off batches for each
# - Allows us to monitor and collect the results when complete
#
# ## COSTS
#
# Using Groq, for me - this cost under $1 for the Lite dataset and under $30 for the big dataset
#
# But you don't need to pay anything! In the next lab, you can load my pre-processed results

# %%


Batch.create(items, LITE_MODE)


# %%


Batch.run()


# %%


Batch.fetch()


# %%


for index, item in enumerate(items):
    if not item.summary:
        print(index)


# %%


print(items[10234].summary)


# %%


# Remove the fields that we don't need in the hub

for item in items:
    item.full = None
    item.id = None


# ## Push the final dataset to the hub
#
# If lite mode, we'll only push the lite dataset
#
# If full mode, we'll push both datasets (in case you decide to use lite later)

# %%


username = "ed-donner"
full = f"{username}/items_full"
lite = f"{username}/items_lite"

if LITE_MODE:
    train = items[:20_000]
    val = items[20_000:21_000]
    test = items[21_000:]
    Item.push_to_hub(lite, train, val, test)
else:
    train = items[:800_000]
    val = items[800_000:810_000]
    test = items[810_000:]
    Item.push_to_hub(full, train, val, test)

    train_lite = train[:20_000]
    val_lite = val[:1_000]
    test_lite = test[:1_000]
    Item.push_to_hub(lite, train_lite, val_lite, test_lite)


# ## And here they are!
#
# https://huggingface.co/datasets/ed-donner/items_lite
#
# https://huggingface.co/datasets/ed-donner/items_full
#

#
