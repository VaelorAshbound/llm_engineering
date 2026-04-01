#!/usr/bin/env python
# coding: utf-8

import subprocess
import time

from IPython.display import Markdown, display
from tqdm import tqdm

# # Notebooks in Cursor
#
# This course makes heavy use of a brilliant thing called Notebooks (also known as Jupyter Notebooks or Labs.) Those from a traditional software engineering background may feel discomfort with the "hacky" nature of Notebooks, but I must assure you: part of working with AI is being comfortable being a Scientist. As a Scientist, there's a lot of exploration and experimentation. And Notebooks are ideal for this kind of activity.
#
# A notebook is a file with the extension ".ipynb" which stands for IPython Notebook, an early name for these.
#
# ## Briefing on Notebooks in Cursor
#
# First, here's a briefing on how this fits together, and how to create and run a notebook in Cursor:
#
# https://chatgpt.com/share/6806291a-25f0-8012-a08b-057acb5045ae
#
#
# ## A broader guide to Notebooks with examples
#
# The Notebook is a Data Science playground where you can easily write code and investigate the results. It's an ideal environment for:
# - Research & Development
# - Prototyping
# - Learning (that's us!)
#
# The notebook consists of a series of square boxes called "cells". Some of them contain text, like this cell, and some of them contain code, like the cell below.
#
# First, you may need to click the `Select Kernel` button on the top right, and then pick `venv (Python 3.12.x)` or similar.
#
# Click in a cell with code and press `Shift + Return` (or `Shift + Enter`) to run the code and print the output.
#
# Do that now for the cell below this:

# %%:


# Click anywhere in this cell and press Shift + Return

print(2 + 2)


# ## Congrats!
#
# Now run the next cell which sets a value, followed by the cells after it to print the value

# In[3]:


# Set a value for a variable

favorite_fruit = "bananas"


# %%:


# The result of the last statement is shown after you run it

print(favorite_fruit)


# %%:


# Use the variable

print(f"My favorite fruit is {favorite_fruit}")


# In[6]:


# Now change the variable

favorite_fruit = f"anything but {favorite_fruit}"


# ## Now go back and rerun the cell with the print statement, two cells back
#
# See how it prints something different, even though favorite_fruit was changed further down in the notebook?
#
# The order that code appears in the notebook doesn't matter. What matters is the order that the code is **executed**. There's a python process sitting behind this notebook in which the variables are being changed.
#
# This catches some people out when they first use notebooks.

# %%:


# Then run this cell twice, and see if you understand what's going on

print(f"My favorite fruit is {favorite_fruit}")

favorite_fruit = "apples"


# # Explaining the 'kernel'
#
# Sitting behind this notebook is a Python process which executes each cell when you run it. That Python process is known as the Kernel. Each notebook has its own separate Kernel.
#
# You can click the button above "Restart Kernel".
#
# If you then try to run the next cell, you'll get an error, because favorite_fruit is no longer defined. You'll need to run the cells from the top of the notebook again. Then the next cell should run fine.

# %%:


print(f"My favorite fruit is {favorite_fruit}")


# # Adding and removing cells
#
# Click in this cell, then click the \[+ Code\] button in the toolbar above to create a new cell immediately below this one. Copy and paste in the code in the prior cell, then run it! There are also icons in the top right of the selected cell to delete it (bin).
#

# %%:


# # Cell output
#
# When you execute a cell, the standard output and the result of the last statement is written to the area immediately under the code, known as the 'cell output'. When you save a Notebook from the file menu (or ctrl+S or command+S), the output is also saved, making it a useful record of what happened.
#
# You can clean this up by clicking "Clear All Outputs" in the toolbar. It's a good idea to clear outputs before you push code to a repo like GitHub, otherwise the files can be large and harder to read.

# %%:


spams = ["spam"] * 1000
print(spams)

# Might be worth clearing output after running this!


# # Using markdown
#
# So what's going on with these areas with writing in them, like this one? Well, there's actually a different kind of cell called a 'Markdown' cell for adding explanations like this. Click the [+ Markdown] button to add a new markdown cell.
#
# Add some comments using Markdown format, perhaps copying and pasting from here:
#
# ```
# # This is a heading
# ## This is a sub-head
# ### And a sub-sub-head
#
# I like Jupyter Lab because it's
# - Easy
# - Flexible
# - Satisfying
# ```
#
# And to turn this into formatted text simply with Shift+Return in the cell.
# Click in the cell and press the Bin icon if you want to remove it.

# %%:


# # The exclamation point
#
# There's a super useful feature of jupyter labs; you can type a command with a ! in front of it in a code cell, like:
#
# !ls
# !pwd
#
# And it will run it at the command line (as if in Windows Powershell or Mac Terminal) and print the result

# %%:


# list the current directory

subprocess.run(["ls"])


# %%:


# ping cnn.com - press the stop / interrupt button in the toolbar when you're bored

subprocess.run(["ping", "cnn.com"])


# # Minor things we encounter on the course
#
# This isn't necessarily a feature of notebooks, but it's a nice package to know about that is useful in notebooks.
#
# The package `tqdm` will print a nice progress bar if you wrap any iterable.

# In[12]:


# Here's some code with no progress bar
# It will take 10 seconds while you wonder what's happpening..

spams = ["spam"] * 1000

for spam in spams:
    time.sleep(0.01)


# %%:


# And now, with a nice little progress bar:

spams = ["spam"] * 1000

for spam in tqdm(spams):
    time.sleep(0.01)


# %%:


# On a different topic, here's a useful way to print output in markdown

display(
    Markdown(
        "# This is a big heading!\n\n- And this is a bullet-point\n- So is this\n- Me, too!"
    )
)


# # That's it! You're up to speed on Notebooks / Labs in Cursor.
#
# ## Want to be even more advanced?
#
# If you want to become a pro at Jupyter Lab (the technology behind this), you can read their tutorial [here](https://jupyterlab.readthedocs.io/en/latest/). But this isn't required for our course; just a good technique for hitting Shift + Return and enjoying the result!

#
