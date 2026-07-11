from pricer.batchb import Batch
from pricer.items import Item

LITE_MODE = True

username = "ed-donner"
dataset = f"{username}/items_raw_lite" if LITE_MODE else f"{username}/items_raw_full"
train, val, test = Item.from_hub(dataset)
items = train + val + test

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

# Remove the fields that we don't need in the hub

for item in items:
    item.full = None
    item.id = None

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
