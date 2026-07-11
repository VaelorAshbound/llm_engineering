# %%
import os

import numpy as np
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login
from pricer.loadersb import load_all
from semhash import SemHash

load_dotenv(override=True)
login(os.environ["HF_TOKEN"])

CATEGORIES = [
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

dataset = load_all(CATEGORIES)
print(f"Loaded {len(dataset):,} items")

# %%

# Semantic dedup (replaces the two manual title/full `seen`-set passes) —
# also removes near-duplicates that exact matching misses.
result = SemHash.from_records(dataset.to_list(), columns=["full"]).self_deduplicate()
dataset = Dataset.from_list(result.selected)
print(f"Dropped {result.duplicate_ratio:.1%} duplicates -> {len(dataset):,} items")

# %%

# Weighted subsample: favour higher prices, damp the over-represented categories.
np.random.seed(42)
SIZE = 820_000

prices = np.array(dataset["price"], dtype=float)
categories = np.array(dataset["category"])
p = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)

w = p**2
w[categories == "Tools_and_Home_Improvement"] *= 0.5
w[categories == "Automotive"] *= 0.05
w /= w.sum()

idx = np.random.choice(len(dataset), size=min(SIZE, len(dataset)), replace=False, p=w)
sample = dataset.select(idx).shuffle(seed=42)

# %%

USERNAME = "ed-donner"

train = sample.select(range(800_000))
val = sample.select(range(800_000, 810_000))
test = sample.select(range(810_000, len(sample)))


DatasetDict(
    {
        "train": train.select(range(20_000)),
        "validation": val.select(range(1_000)),
        "test": test.select(range(1_000)),
    }
).push_to_hub(f"{USERNAME}/items_raw_lite")
