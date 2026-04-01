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
# ## DAY 4: Neural Networks and LLMs
#
# Today we'll work from Traditional ML to Neural Networks to Large Language Models!!

# %%


# imports

import csv
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from huggingface_hub import login
from litellm import completion
from pricer.evaluator import evaluate
from pricer.items import Item
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm

# %%


LITE_MODE = False

load_dotenv(override=True)
hf_token = os.environ["HF_TOKEN"]
login(hf_token, add_to_git_credential=True)


# %%


username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

print(
    f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
)


# # Before we look at the Artificial Neural Networks
#
# ## There is a different kind of Neural Network we could consider

# %%


# Write the test set to a CSV

with open("human_in.csv", "w", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for t in test[:100]:
        writer.writerow([t.summary, 0])


# %%


# Read it back in

human_predictions = []
with open("human_out.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        human_predictions.append(float(row[1]))


# %%


def human_pricer(item):
    idx = test.index(item)
    return human_predictions[idx]


# %%


human = human_pricer(test[0])
actual = test[0].price
print(f"Human predicted {human} for an item that actually costs {actual}")


# %%


evaluate(human_pricer, test, size=100)


# # And now - a vanilla Neural Network
#
# During the remainder of this course we will get deeper into how Neural Networks work, and how to train a neural network.
#
# This is just a sneak preview - let's make our own Neural Network, from scratch, using Pytorch.
#
# Use this to get intuition; it's not important to know all about Neural networks at this point..

# %%


# Prepare our documents and prices

y = np.array([float(item.price) for item in train])
documents = [item.summary for item in train]


# %%


# Use the HashingVectorizer for a Bag of Words model
# Using binary=True with the CountVectorizer makes "one-hot vectors"

np.random.seed(42)
vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
X = vectorizer.fit_transform(documents)


# %%


# Define the neural network - here is Pytorch code to create a 8 layer neural network


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 64)
        self.layer6 = nn.Linear(64, 64)
        self.layer7 = nn.Linear(64, 64)
        self.layer8 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output1 = self.relu(self.layer1(x))
        output2 = self.relu(self.layer2(output1))
        output3 = self.relu(self.layer3(output2))
        output4 = self.relu(self.layer4(output3))
        output5 = self.relu(self.layer5(output4))
        output6 = self.relu(self.layer6(output5))
        output7 = self.relu(self.layer7(output6))
        output8 = self.layer8(output7)
        return output8


# %%


# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X.toarray())  # type: ignore[union-attr]
y_train_tensor = torch.FloatTensor(y).unsqueeze(1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.01, random_state=42
)

# Create the loader
train_dataset = TensorDataset(X_train, y_train)  # type: ignore[arg-type]
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model
input_size = X_train_tensor.shape[1]
model = NeuralNetwork(input_size)


# %%


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {trainable_params:,}")


# %%


# Define loss function and optimizer

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# We will do 2 complete runs through the data

EPOCHS = 2
loss = torch.tensor(0.0)

for epoch in range(EPOCHS):
    model.train()
    for batch_X, batch_y in tqdm(train_loader):
        optimizer.zero_grad()

        # The next 4 lines are the 4 stages of training: forward pass, loss calculation, backward pass, optimize

        outputs = model(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = loss_function(val_outputs, y_val)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {loss.item():.3f}, Val Loss: {val_loss.item():.3f}"
    )


# %%


def neural_network(item):
    model.eval()
    with torch.no_grad():
        vector = vectorizer.transform([item.summary])
        vector = torch.FloatTensor(vector.toarray())  # type: ignore[union-attr]
        result = model(vector)[0].item()
    return max(0, result)


# %%


evaluate(neural_network, test)


# # And now - to the frontier!
#
# Let's see how Frontier models do out of the box; no training, just inference based on their world knowledge.
#
# Tomorrow we will do some training.

# %%


def messages_for(item):
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [{"role": "user", "content": message}]


# %%


print(test[0].summary)


# %%


messages_for(test[0])


# %%


# The function for gpt-4.1-nano


def gpt_4__1_nano(item):
    response: Any = completion(model="openai/gpt-4.1-nano", messages=messages_for(item))
    return response.choices[0].message.content


# %%


gpt_4__1_nano(test[0])


# %%


test[0].price


# %%


evaluate(gpt_4__1_nano, test)


# %%


def claude_opus_4_5(item):
    response: Any = completion(
        model="anthropic/claude-opus-4-5", messages=messages_for(item)
    )
    return response.choices[0].message.content


# %%


evaluate(claude_opus_4_5, test)


# %%


def gemini_3_pro_preview(item):
    response: Any = completion(
        model="gemini/gemini-3-pro-preview",
        messages=messages_for(item),
        reasoning_effort="low",
    )
    return response.choices[0].message.content


# %%


evaluate(gemini_3_pro_preview, test, size=50, workers=2)


# %%


def gemini_2__5_flash_lite(item):
    response: Any = completion(
        model="gemini/gemini-2.5-flash-lite", messages=messages_for(item)
    )
    return response.choices[0].message.content


# %%


evaluate(gemini_2__5_flash_lite, test)


# %%


def grok_4__1_fast(item):
    response: Any = completion(
        model="xai/grok-4-1-fast-non-reasoning", messages=messages_for(item), seed=42
    )
    return response.choices[0].message.content


# %%


evaluate(grok_4__1_fast, test)


# %%


# The function for gpt-5.1


def gpt_5__1(item):
    response: Any = completion(
        model="gpt-5.1", messages=messages_for(item), reasoning_effort="high", seed=42
    )
    return response.choices[0].message.content


# %%


evaluate(gpt_5__1, test)


# %%
