#!/usr/bin/env python
# coding: utf-8

# ## The Big Project begins!!
#
# # "THE PRICE IS RIGHT" Capstone Project
#
# This week - build a model that predicts how much something costs from a description, based on a scrape of Amazon data
#
# # Order of play
#
# DAY 1: Data Curation
# DAY 2: Data Pre-processing
# DAY 3: Evaluation, Baselines, Traditional ML
# DAY 4: Deep Learning and LLMs
# DAY 5: Fine-tuning a Frontier Model
#
# ## DAY 1: Data Curation
#
# Today we'll scrub our dataset and curate our data
#
# The dataset is here:
# https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
#
# And the folder with all the product datasets is here:
# https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/meta_categories

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/business.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#181;">Business value of Data Curation</h2>
#             <span style="color:#181;">Data Curation can be considered the less glamorous work of a Data Scientist. I say that's nonsense!
#             This is where the science happens - what could be more glamorous than that?! R&D with your
#             dataset can often have a greater impact on performance than the fashionable 'hyper-parameter optimization' that we do later.
#             So: prepare for Quality Time with Data Quality.</span>
#         </td>
#     </tr>
# </table>

# %%


# imports

import os
import random
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from pricer.items import Item
from pricer.loaders import ItemLoader
from pricer.parser import parse
from tqdm.notebook import tqdm

load_dotenv(override=True)


# %%


# Log in to HuggingFace - if you get a "Note" about Environment variable being set, ignore it

hf_token = os.environ["HF_TOKEN"]
login(hf_token, add_to_git_credential=True)


# ## Load our dataset
#
# In the next cell, we load in the dataset from huggingface.
#
# If this gives you an error like "trust_remote_code is no longer supported", then please run this command in a new cell: `!uv add --upgrade datasets==3.6.0` and then restart the Kernel, and try again.

# %%


dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Appliances",
    split="full",
    trust_remote_code=True,
)
assert isinstance(dataset, Dataset)


# %%


print(f"Number of Appliances: {len(dataset):,}")


# %%


# Investigate a particular datapoint

dataset[6]


# %%


# What's the most expensive item?

max_price = 0
max_item = None

for datapoint in tqdm(dataset):
    try:
        price = float(datapoint["price"])  # type: ignore[index]
        if price > max_price:
            max_item = datapoint
            max_price = price
    except ValueError:
        pass

if max_item is not None:
    print(
        f"The most expensive item is {max_item['title']} and it costs {max_price:,.2f}"  # type: ignore[index]
    )


# This is the closest I can find - looks like it's going at a bargain price!!
#
# https://www.amazon.com/TurboChef-Electric-Countertop-Microwave-Convection/dp/B01D05U9NO/

# %%


# Load into Item objects if they have a price range $1-$1000 and enough details

items = [parse(datapoint, "Appliances") for datapoint in tqdm(dataset)]
items = [item for item in items if item is not None]
print(f"There are {len(items):,} items from {len(dataset):,} datapoints")


# %%


items[0]


# %%


print(items[0].full)


# %%


prices = [item.price for item in items]
lengths = [len(item.full or "") for item in items]


# %%


# Plot the distribution of lengths

plt.figure(figsize=(15, 6))
plt.title(
    f"Lengths: Avg {sum(lengths) / len(lengths):,.0f} and highest {max(lengths):,}\n"
)
plt.xlabel("Length (chars)")
plt.ylabel("Count")
plt.hist(lengths, rwidth=0.7, color="lightblue", bins=range(0, 6000, 100))
plt.show()


# %%


max_length = max(lengths)
max_length_item = items[lengths.index(max_length)]
print(max_length_item.full)


# %%


# Plot the distribution of prices
plt.figure(figsize=(15, 6))
plt.title(f"Prices: Avg {sum(prices) / len(prices):,.2f} and highest {max(prices):,}\n")
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.hist(prices, rwidth=0.7, color="orange", bins=range(0, 1000, 10))
plt.show()


# %%


print(items[3].full)


# %%


loader = ItemLoader("Appliances")
items = loader.load()


# %%


dataset_names = [
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Appliances",
    "Musical_Instruments",
]


# %%


items = []
for dataset_name in dataset_names:
    loader = ItemLoader(dataset_name)
    items.extend(loader.load())


# %%


print(f"A grand total of {len(items):,} items")


# %%


items[1000]


# %%


random.seed(42)
random.shuffle(items)

seen = set()
items = [x for x in tqdm(items) if not (x.title in seen or seen.add(x.title))]

seen = set()
items = [x for x in tqdm(items) if not (x.full in seen or seen.add(x.full))]

del seen
print(f"After deduplication, we have {len(items):,} items")


# %%


lengths = [len(item.full or "") for item in items]
plt.figure(figsize=(15, 6))
plt.title(
    f"Text length: Avg {sum(lengths) / len(lengths):,.1f} and highest {max(lengths):,}\n"
)
plt.xlabel("Length (characters)")
plt.ylabel("Count")
plt.hist(lengths, rwidth=0.7, color="skyblue", bins=range(0, 4050, 50))
plt.show()


# %%


# Plot the distribution of prices

prices = [item.price for item in items]
plt.figure(figsize=(15, 6))
plt.title(f"Prices: Avg {sum(prices) / len(prices):,.1f} and highest {max(prices):,}\n")
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.hist(prices, rwidth=0.7, color="blueviolet", bins=range(0, 1000, 10))
plt.show()


# %%


category_counts = Counter([item.category for item in items])

categories = list(category_counts.keys())
counts = [category_counts[category] for category in categories]

plt.figure(figsize=(15, 6))
plt.bar(categories, counts, color="goldenrod")
plt.title("How many in each category")
plt.xlabel("Categories")
plt.ylabel("Count")
plt.xticks(rotation=30, ha="right")

for i, v in enumerate(counts):
    plt.text(i, v, f"{v:,}", ha="center", va="bottom")

plt.show()


# %%


np.random.seed(42)

SIZE = 820_000

prices = np.array([it.price for it in items], dtype=float)
categories = np.array([it.category for it in items])
p = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)

w = p**2
w[categories == "Tools_and_Home_Improvement"] *= 0.5
w[categories == "Automotive"] *= 0.05

w = w / w.sum()
idx = np.random.choice(len(items), size=SIZE, replace=False, p=w)
sample = [items[i] for i in idx]


# %%


prices = [item.price for item in sample]
plt.figure(figsize=(15, 6))
plt.title(
    f"Prices: Avg {sum(prices) / len(prices):,.1f} lowest {min(prices):,} and highest {max(prices):,}\n"
)
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.hist(prices, rwidth=0.7, color="blueviolet", bins=range(0, 1000, 10))
plt.show()


# %%


# Just for good measure, let's shuffle the sample again for the final dataset

random.seed(42)
random.shuffle(sample)


# %%


prices = [item.price for item in sample]
plt.figure(figsize=(15, 6))
plt.title(
    f"Prices: Avg {sum(prices) / len(prices):,.1f} lowest {min(prices):,} and highest {max(prices):,}\n"
)
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.hist(prices, rwidth=0.7, color="blueviolet", bins=range(0, 1000, 10))
plt.show()


# %%


category_counts = Counter([item.category for item in sample])

categories = list(category_counts.keys())
counts = [category_counts[category] for category in categories]

# Bar chart by category
plt.figure(figsize=(15, 6))
plt.bar(categories, counts, color="goldenrod")
plt.title("How many in each category")
plt.xlabel("Categories")
plt.ylabel("Count")

plt.xticks(rotation=30, ha="right")

# Add value labels on top of each bar
for i, v in enumerate(counts):
    plt.text(i, v, f"{v:,}", ha="center", va="bottom")

# Display the chart
plt.show()


# %%


# Automotive still in the lead, but improved somewhat
# For another perspective, let's look at a pie

plt.figure(figsize=(12, 10))
plt.pie(counts, labels=categories, autopct="%1.0f%%", startangle=90)

# Add a circle at the center to create a donut chart (optional)
centre_circle = mpatches.Circle((0, 0), 0.70, fc="white")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Categories")

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis("equal")

plt.show()


# %%


# How does the price vary with the character count?

sizes = [len(item.full or "") for item in sample]
prices = [item.price for item in sample]

# Create the scatter plot
plt.figure(figsize=(15, 8))
plt.scatter(sizes, prices, s=0.2, color="red")

# Add labels and title
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("Is there a simple correlation with text length?")

# Display the plot
plt.show()


# %%


# How does the price vary with the weight?

ounces = [item.weight for item in sample]
prices = [item.price for item in sample]

# Create the scatter plot
plt.figure(figsize=(15, 8))
plt.scatter(ounces, prices, s=0.2, color="darkorange")

# Add labels and title
plt.xlabel("Weight (ounces)")
plt.ylabel("Price")
plt.xlim(0, 400)
plt.title("Is there a simple correlation with weight?")

# Display the plot
plt.show()


# ## Now push this dataset to the HuggingFace Hub
#
# Replace the username with your HF username if you've crafted your own dataset
#
# Or, ignore this cell and you can load my dataset tomorrow!

# %%


username = "ed-donner"
full = f"{username}/items_raw_full"
lite = f"{username}/items_raw_lite"

train = sample[:800_000]
val = sample[800_000:810_000]
test = sample[810_000:]

Item.push_to_hub(full, train, val, test)

train_lite = train[:20_000]
val_lite = val[:1_000]
test_lite = test[:1_000]

Item.push_to_hub(lite, train_lite, val_lite, test_lite)


# ## Sidenote
#
# If you like the variety of colors that matplotlib can use in its charts, you should bookmark this:
#
# https://matplotlib.org/stable/gallery/color/named_colors.html
#
