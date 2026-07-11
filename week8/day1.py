#!/usr/bin/env python
# coding: utf-8

# # Welcome to a very busy Week 8 folder
#
# ## We have lots to do this week!
#
# We'll move at a faster pace than usual, particularly as you're becoming proficient LLM engineers.
#
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

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#b22;">Especially important this week: pull the latest</h2>
#             <span style="color:#b22;">I'm continually improving these labs, adding more examples and exercises.
#             At the start of each week, it's worth checking you have the latest code.<br/>
#             Please see Guide 3 in the Guides folder if you're not sure how to do a <code>git pull</code>
#             </span>
#         </td>
#     </tr>
# </table>

# %%


import locale
import os

import modal
from dotenv import load_dotenv

from agents.preprocessor import Preprocessor

load_dotenv(override=True)


# %%


# To check that your computer can output special characters, make sure this outputs UTF-8

print(locale.getpreferredencoding())  # Should print 'UTF-8'


# %%


os.environ["PYTHONIOENCODING"] = "utf-8"


# # Setting up the modal tokens
#
# ## IMPORTANT - please do read and follow these instructions!
#
# First please visit: https://modal.com
#
# And sign up for an account. Then click the Avatar menu on the top right and select "Settings"
#
# Then click "API Tokens" in the left sidebar, then click "New Token".
#
# You will be given something like this to run:
#
# `modal token set --token-id ak-somethinghere --token-secret as-somethinghere`
#
# But because we're using uv, the real thing to run is this:
#
# `uv run modal token set --token-id ak-somethinghere --token-secret as-somethinghere`
#
# ### Troubleshooting
#
# If you have problems, 3 things to try:
#
# 1. Try running `uv run modal token new` before the `uv run modal token set..`
#
# 2. Suggestion from student David S. on Windows:
#
# > In case anyone else using Windows hits this problem: Along with having to run `modal token new` from a command prompt, you have to move the generated token file. It will deploy the token file (.modal.toml) to your Windows profile folder. The virtual environment couldn't see that location (strangely, it couldn't even after I set environment variables for it and rebooted). I moved that token file to the folder I'm operating out of for the lab and it stopped throwing auth errors.
#
# 3. Doing in the manual way:
#
# It might be totally fine to simply add the 2 keys directly to your .env file:
#
# ```
# MODAL_TOKEN_ID=ak-...
# MODAL_TOKEN_SECRET=as-...
# ```
#
# Then rerun `load_dotenv(override=True)` to load these environment variables.

# %%


from hello import app, hello, hello_europe

# %%


with app.run():
    reply = hello.local()
reply


# %%


with app.run():
    reply = hello.remote()
reply


# ## Added thanks to student Tue H.
#
# If you look in hello.py, I've added a simple function hello_europe
#
# That uses the decorator:
# `@app.function(image=image, region="eu")`
#
# See the result below! More region specific settings are [here](https://modal.com/docs/guide/region-selection)
#
# Note that it does consume marginally more credits to specify a region.

# %%


with app.run():
    reply = hello_europe.remote()
reply


# # Before we move on -
#
# ## We need to set your HuggingFace Token as a secret in Modal
#
# ## Super important - please read - this confuses a lot of people!
#
# Secrets in Modal are given a **name** that describes the secret.
# Then the secret itself has a KEY and a VALUE.
# We will be setting up a secret with:
#
# Name: huggingface-secret
# Key: HF_TOKEN
# Value: hf_...
#
# ## The bulletproof recipe:
#
# 1. Go to modal.com, sign in and go to your dashboard
# 2. Click on Secrets in the nav bar
# 3. Create new secret, click on Hugging Face, this new secret needs to be called **huggingface-secret** because that's how we refer to it in the code
# 4. Fill in your key as HF_TOKEN and the value as your actual token hf_...
# 5. Click done
#
# ### And now back to business: time to work with Llama

# %%


# This import may give a deprecation warning about adding local Python modules to the Image
# That warning can be safely ignored. You may get the same warning in other places, too..

from llama import app, generate

# %%


with modal.enable_output():
    with app.run():
        result = generate.remote("Never gonna give you up, never gonna")
result

# Or it you object to being rickrolled, try this: "Hey Jude, don't make it"


# %%


from pricer_ephemeral import app, price

# %%


with modal.enable_output():
    with app.run():
        result = price.remote(
            "Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio"
        )
result


# %%


preprocessor = Preprocessor()
text = preprocessor.preprocess(
    "Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio"
)
print(text)


# %%


preprocessor = Preprocessor(model_name="groq/openai/gpt-oss-20b")
text = preprocessor.preprocess(
    "Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio"
)
print(text)


# ### Add this to your .env if you want the Preprocessor to use a different model by default:
#
# `PRICER_PREPROCESSOR_MODEL=groq/openai/gpt-oss-20b`

# %%


with modal.enable_output():
    with app.run():
        result = price.remote(text)
print(result)


# %%


print(text)


# ## Transitioning From Ephemeral Apps to Deployed Apps
#
# From a command line, `uv run modal deploy xxx` will deploy your code as a Deployed App
#
# This is how you could package your AI service behind an API to be used in a Production System.
#
# You can also build REST endpoints easily, although we won't cover that as we'll be calling direct from Python.
#
# ## Important note about secrets
#
# In both the files `pricer_service.py` and `pricer_service2.py` you will find code like this near the top:
# `secrets = [modal.Secret.from_name("hf-secret")]`
# You may need to change from `hf-secret` to `huggingface-secret` depending on how the Secret is configured in modal.
# To check, visit this page and look in the first column:
# https://modal.com/secrets/
#
# ## Important note for Windows people:
#
# On the next line, I call `uv run modal deploy` from within Jupyter lab; I've heard that on some versions of Windows this gives a strange unicode error because modal prints emojis to the output which can't be displayed. If that happens to you, open a Terminal and run `uv run modal deploy..`

# %%


# You can also run "uv run modal deploy -m pricer_service" in the Terminal

# !uv run modal deploy -m pricer_service


# %%


pricer = modal.Function.from_name("pricer-service", "price")


# Watch it happening:
#
# https://modal.com

# %%


# This can take a while! We'll use faster approaches shortly

pricer.remote(text)


# %%


# You can also run "modal deploy -m pricer_service2" at the command line in an activated environment

# !modal deploy -m pricer_service2


# %%


Pricer = modal.Cls.from_name("pricer-service", "Pricer")
pricer = Pricer()
reply = pricer.price.remote(text)
print(reply)


# %%


reply = pricer.price.remote(text)
print(reply)


# # Optional: Keeping Modal warm
#
# ## A way to improve the speed of the Modal pricer service
#
# The first time you run this modal class, it might take as much as 10 minutes to build.
# Subsequently it should be much faster.. 30 seconds if it needs to wake up, otherwise 2 seconds.
# If you want it to always be 2 seconds, you can keep the container from going to sleep by editing this constant in pricer_service2.py:
#
# `MIN_CONTAINERS = 0`
#
#
#
# Make it 1 to keep a container alive.
# But please note: this will eat up credits! Only do this if you are comfortable to have a process running continually.
#
# Alternatively, you can run this code and it will stay warm for 20 mins rather than 2 mins.
#
# ### Code to keep warm for 20 mins before cooling down:
#
# ```python
# import modal
# Pricer = modal.Cls.from_name("pricer-service", "Pricer")
# pricer = Pricer()
# pricer.update_autoscaler(scaledown_window=1200)
# ```
#
# ### Code to revert to keeping warm for only 2 mins
#
# ```python
# import modal
# Pricer = modal.Cls.from_name("pricer-service", "Pricer")
# pricer = Pricer()
# pricer.update_autoscaler(scaledown_window=120)
# ```

# ## And now introducing our Agent class
#
# By default this will preprocess using Llama3.2
#
# If you'd prefer to use Groq, then add this env variable like:
#
# ```
# PRICER_PREPROCESSOR_MODEL=groq/openai/gpt-oss-20b
# ```

# %%


import logging

root = logging.getLogger()
root.setLevel(logging.INFO)


# %%


from agents.specialist_agent import SpecialistAgent

# %%


agent = SpecialistAgent()


# %%


agent.price("iPhone 10")


# %%
