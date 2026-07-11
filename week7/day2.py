#!/usr/bin/env python
# coding: utf-8

# # Week 7 Day 2
#
# # "THE PRICE IS RIGHT" Capstone Project
#
# This week - fine tune an open-source model!
#
# A model that can estimate how much something costs, from its description.
#
# # Order of play
#
# DAY 1: QLoRA
# DAY 2: Prompt data and Base Model
# DAY 3: Train Part 1
# DAY 4: Train Part 2
# DAY 5: Eval
#
# ## First we need to upload the final dataset
#
#

# %%


import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
from pricer.items import Item
from tqdm.notebook import tqdm
from transformers import AutoTokenizer

# %%


LITE_MODE = False

load_dotenv(override=True)
hf_token = os.environ["HF_TOKEN"]
login(hf_token, add_to_git_credential=True)


# %%


username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)
items = train + val + test

print(
    f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
)


# %%


BASE_MODEL = "meta-llama/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


# %%


token_counts = [item.count_tokens(tokenizer) for item in tqdm(items)]


# %%


plt.figure(figsize=(15, 6))
plt.title(
    f"Tokens in Summary: Avg {sum(token_counts) / len(token_counts):,.1f} and highest {max(token_counts):,}\n"
)
plt.xlabel("Number of tokens in summary")
plt.ylabel("Count")
plt.hist(token_counts, rwidth=0.7, color="skyblue", bins=range(0, 200, 10))
plt.show()


# %%


CUTOFF = 110
cut = len([count for count in token_counts if count > CUTOFF])
print(
    f"With this CUTOFF, we will truncate {cut:,} items which is {cut / len(items):.1%}"
)


# %%


print(train[0].summary)


# %%


for item in tqdm(train + val):
    item.make_prompts(tokenizer, CUTOFF, True)
for item in tqdm(test):
    item.make_prompts(tokenizer, CUTOFF, False)


# %%


print("PROMPT:")
print(test[0].prompt)
print("COMPLETION:")
print(test[0].completion)


# %%


prompt_token_counts = [item.count_prompt_tokens(tokenizer) for item in tqdm(items)]


# %%


plt.figure(figsize=(15, 6))
plt.title(
    f"Tokens: Avg {sum(prompt_token_counts) / len(prompt_token_counts):,.1f} and highest {max(prompt_token_counts):,}\n"
)
plt.xlabel("Number of tokens in prompt and the completion")
plt.ylabel("Count")
plt.hist(prompt_token_counts, rwidth=0.7, color="gold", bins=range(0, 200, 10))
plt.show()


# %%


username = "ed-donner"
dataset = (
    f"{username}/items_prompts_lite" if LITE_MODE else f"{username}/items_prompts_full"
)

Item.push_prompts_to_hub(dataset, train, val, test)


# Here are the datasets in HuggingFace:
#
# https://huggingface.co/datasets/ed-donner/items_prompts_lite
#
# https://huggingface.co/datasets/ed-donner/items_prompts_full
#
# Please see this notebook in Google Colab:
#
# https://colab.research.google.com/drive/1wO3lNMrMfprlJZF4X9fSsQ8tYC3SRZbh?usp=sharing

#
