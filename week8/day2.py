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
# ## RAG (Retrieval Augmented Generation) based on a dataset of 800,000 scraped Amazon products
#
# #### For our 2nd agent, we will be asking OpenAI to estimate the price of one of our deals - and we will give it a hand.
#
# We discovered that LLMs are really good at this, out of the box.
#
# And we discovered that we can beat a frontier LLM by fine-tuning an open-source LLM.
#
# Now we are going to try **inference time** techniques instead of training -- by using RAG!

# %%


# imports

import logging
import os
import re

import chromadb
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from huggingface_hub import login
from litellm import completion
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm

from agents.evaluator import evaluate
from agents.items import Item

# %%


# environment

load_dotenv(override=True)
DB = "products_vectorstore"


# %%


# Log in to HuggingFace
# If you don't have a HuggingFace account, you can set one up for free at www.huggingface.co
# And then add the HF_TOKEN to your .env file as explained in the project README

hf_token = os.environ["HF_TOKEN"]
login(token=hf_token, add_to_git_credential=False)


# %%


LITE_MODE = False


# %%


username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

print(
    f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
)


# # Now create a Chroma Datastore
#
# Now we will use the free, open-source Vector database Chroma.
# We will create a Chroma datastore with 400,000 products from our training dataset.

# %%


client = chromadb.PersistentClient(path=DB)


# # Introducing the SentenceTransformer Encoding LLM
#
# The all-MiniLM is a very useful model from HuggingFace that maps sentences & paragraphs to 384 dimensional vectors and is ideal for tasks like semantic search.
#
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#
# It can run pretty quickly locally.
#
# As an alternative, OpenAI provides a closed-source Embeddings model. Benefits compared to OpenAI embeddings:
# 1. It's free and fast!
# 3. We can run it locally, so the data never leaves our box - might be useful if you're building a personal RAG

# %%


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# %%


# Pass in a list of texts, get back a numpy array of vectors

vector = encoder.encode(
    [
        "A proficient AI engineer who has almost reached the finale of AI Engineering Core Track!"
    ]
)[0]
print(vector.shape)
vector


# ## With that background, let's populate our Chroma database
#
# ### By calculating vectors for 800,000 scraped products
#
# This takes 30 minutes on my machine on my GPU - it might take longer for you - feel free to use the Lite dataset!

# %%


# Check if the collection exists; if not, create it

collection_name = "products"
existing_collection_names = [
    collection.name for collection in client.list_collections()
]

if collection_name not in existing_collection_names:
    collection = client.create_collection(collection_name)
    for i in tqdm(range(0, len(train), 1000)):
        documents = [item.summary for item in train[i : i + 1000]]
        vectors = encoder.encode(documents).astype(float).tolist()
        metadatas = [
            {"category": item.category, "price": item.price}
            for item in train[i : i + 1000]
        ]
        ids = [f"doc_{j}" for j in range(i, i + 1000)]
        ids = ids[: len(documents)]
        collection.add(
            ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas
        )

collection = client.get_or_create_collection(collection_name)


# # Let's visualize the vectorized data

# %%


# It is very fun turning this up to 800_000 and seeing the full dataset visualized,
# but it almost crashes my box every time so do that at your own risk!! 10_000 is safe!

MAXIMUM_DATAPOINTS = 10_000


# %%


CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["cyan", "blue", "brown", "orange", "yellow", "green", "purple", "red"]


# %%


# Prework
result = collection.get(
    include=["embeddings", "documents", "metadatas"], limit=MAXIMUM_DATAPOINTS
)
vectors = np.array(result["embeddings"])
documents = result["documents"]
categories = [metadata["category"] for metadata in result["metadatas"]]
colors = [COLORS[CATEGORIES.index(c)] for c in categories]


# %%


# Let's try a 2D chart
# TSNE stands for t-distributed Stochastic Neighbor Embedding - it's a common technique for reducing dimensionality of data

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)


# %%


# Create the 2D scatter plot
fig = go.Figure(
    data=[
        go.Scatter(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            mode="markers",
            marker=dict(size=4, color=colors, opacity=0.7),
            text=[
                f"Category: {c}<br>Text: {d[:50]}..."
                for c, d in zip(categories, documents)
            ],
            hoverinfo="text",
        )
    ]
)

fig.update_layout(
    title="2D Chroma Vectorstore Visualization",
    scene=dict(xaxis_title="x", yaxis_title="y"),
    width=1200,
    height=800,
    margin=dict(r=20, b=10, l=10, t=40),
)

fig.show()


# %%


# Let's try 3D!

tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)


# %%


# Create the 3D scatter plot
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode="markers",
            marker=dict(size=2, color=colors, opacity=0.7),
            text=[
                f"Category: {c}<br>Text: {d[:50]}..."
                for c, d in zip(categories, documents)
            ],
            hoverinfo="text",
        )
    ]
)

fig.update_layout(
    title="3D Chroma Vector Store Visualization",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    width=1200,
    height=800,
    margin=dict(r=20, b=10, l=10, t=40),
)

fig.show()


# %%


test[0]


# %%


def vector(item):
    return encoder.encode(item.summary)


# %%


def find_similars(item):
    vec = vector(item)
    results = collection.query(query_embeddings=vec.astype(float).tolist(), n_results=5)
    documents = results["documents"][0][:]
    prices = [m["price"] for m in results["metadatas"][0][:]]
    return documents, prices


# %%


find_similars(test[0])


# %%


# We need to give some context to GPT-5.1 by selecting 5 products with similar descriptions


def make_context(similars, prices):
    message = "For context, here are some other items that might be similar to the item you need to estimate.\n\n"
    for similar, price in zip(similars, prices):
        message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
    return message


# %%


documents, prices = find_similars(test[0])
print(make_context(documents, prices))


# %%


def messages_for(item, similars, prices):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}\n\n"
    message += make_context(similars, prices)
    return [{"role": "user", "content": message}]


# %%


documents, prices = find_similars(test[0])
print(messages_for(test[0], documents, prices)[0]["content"])


# %%


# The function for gpt-5-mini


def gpt_5__1_rag(item):
    documents, prices = find_similars(item)
    response = completion(
        model="gpt-5.1",
        messages=messages_for(item, documents, prices),
        reasoning_effort="none",
        seed=42,
    )
    return response.choices[0].message.content


# %%


# How much does our favorite distortion pedal cost?

test[0].price


# %%


# Let's do this!!

gpt_5__1_rag(test[0])


# %%


evaluate(gpt_5__1_rag, test)


# %%


import modal

Pricer = modal.Cls.from_name("pricer-service", "Pricer")
pricer = Pricer()


# %%


def specialist(item):
    return pricer.price.remote(item.summary)


# %%


def get_price(reply):
    reply = reply.replace("$", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", reply)
    return float(match.group()) if match else 0


# ## Download the Neural Network weights from Week 6 into this directory
#
# The file `deep_neural_network.pth` here:
#
# https://drive.google.com/drive/folders/1uq5C9edPIZ1973dArZiEO-VE13F7m8MK?usp=drive_link

# %%


from agents.deep_neural_network import DeepNeuralNetworkInference

runner = DeepNeuralNetworkInference()
runner.setup()
runner.load("deep_neural_network.pth")


def deep_neural_network(item):
    return runner.inference(item.summary)


# %%


def ensemble(item):
    price1 = get_price(gpt_5__1_rag(item))
    price2 = specialist(item)
    price3 = deep_neural_network(item)
    return price1 * 0.8 + price2 * 0.1 + price3 * 0.1


# %%


evaluate(ensemble, test)


# %%


root = logging.getLogger()
root.setLevel(logging.INFO)


# %%


from agents.frontier_agent import FrontierAgent

agent = FrontierAgent(collection)
agent.price(
    "Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio"
)


# %%


agent.price("Shure MV7+ professional podcaster microphone with usb-c and XLR outputs")


# %%


from agents.neural_network_agent import NeuralNetworkAgent

agent = NeuralNetworkAgent()


# %%


agent.price("Shure MV7+ professional podcaster microphone with usb-c and XLR outputs")


# %%


from agents.ensemble_agent import EnsembleAgent

agent = EnsembleAgent(collection)


# %%


agent.price("Shure MV7+ professional podcaster microphone with usb-c and XLR outputs")


# %%
