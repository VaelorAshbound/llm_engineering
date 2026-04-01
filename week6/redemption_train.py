#!/usr/bin/env python
# coding: utf-8

# # Week 6 Optional Extra - Deep Neural Network
#
# Just to redeem ourselves from the disappointing result
#
# This is very optional to run yourself! Switch to the other notebook redemption_run to load the trained file and run it!

# In[1]:


import os

from dotenv import load_dotenv
from huggingface_hub import login
from pricer.deep_neural_network import DeepNeuralNetworkRunner
from pricer.evaluator import evaluate
from pricer.items import Item

# In[3]:


# environment

LITE_MODE = False

load_dotenv(override=True)
hf_token = os.environ["HF_TOKEN"]
login(hf_token, add_to_git_credential=True)


# In[4]:


username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODE else f"{username}/items_full"

train, val, test = Item.from_hub(dataset)

print(
    f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items"
)


# In[5]:


runner = DeepNeuralNetworkRunner(train, val[:1000])
runner.setup()


# In[6]:


runner.train(epochs=5)


# %%


def deep_neural_network(item):
    return runner.inference(item)


evaluate(deep_neural_network, test)


# In[8]:


runner.save("deep_neural_network.pth")


# %%
