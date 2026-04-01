# Oh dear!

If you've got here, then you're still having problems setting up your environment. I'm so sorry! Hang in there and we should have you up and running in no time.

Setting up a Data Science environment can be challenging because there's a lot going on under the hood. But we will get there.

And please remember - I'm standing by to help out. Message me or email ed@edwarddonner.com and I'll get on the case. The very last cell in this notebook has some diagnostics that will help me figure out what's happening.

You might want to have a quick look at the [faq](https://edwarddonner.com/faq) on my homepage in case your problem is specifically addressed.


# If you are taking the new version of the course (using uv and Cursor, rather than Anaconda) then check this section first, otherwise jump ahead to the next section headed "Starting with the basics"

## 1. Check Cursor extensions

Just to confirm that the extensions are installed:
- Open extensions (View >> extensions)
- Search for python, and when the results show, click on the ms-python one, and Install it if not already installed
- Search for jupyter, and when the results show, click on the Microsoft one, and Install it if not already installed  
Then View >> Explorer to bring back the File Explorer.

## 2. Connect this Kernel:

If you see the words `Select Kernel` in a button near the top right of this Window, then press the button!

You should see a drop down titled "Select kernel for.." or you might need to pick "Python environment" first.

Pick the one that begins `.venv python 3.12` - it should be the top choice. You might need to click "Python Environments" first.

It should now say `.venv (Python 3.12.x)` where it used to say `Select Kernel`.

After you click "Select Kernel", if there is no option like `.venv (Python 3.12.x)` then please do the following:  
1. On Mac: From the Cursor menu, choose Settings >> VS Code Settings (NOTE: be sure to select `VSCode Settings` not `Cursor Settings`);  
Or on Windows PC: From the File menu, choose Preferences >> VS Code Settings (NOTE: be sure to select `VSCode Settings` not `Cursor Settings`)  
2. In the Settings search bar, type "venv"  
3. In the field "Path to folder with a list of Virtual Environments" put the path to the project root, like C:\Users\username\projects\llm_engineering (on a Windows PC) or /Users/username/projects/llm_engineering (on Mac or Linux).  
And then try again.

## 3. Any uv problems

Please check out this comprehensive guide on my [FAQ Q11](https://edwarddonner.com/faq/#11)

# Starting with the basics

## Checking your internet connection

First let's check that there's no VPN or Firewall or Certs problem.

Click in the cell below and press Shift+Return to run it.  
If this gives you problems, then please try working through these instructions to address:  
https://chatgpt.com/share/676e6e3b-db44-8012-abaa-b3cf62c83eb3

I've also heard that you might have problems if you are using a work computer that's running security software zscaler.

Some advice from students in this situation with zscaler:

> In the anaconda prompt, this helped sometimes, although still got failures occasionally running code in Jupyter:
`conda config --set ssl_verify false`  
Another thing that helped was to add `verify=False` anytime where there is `request.get(..)`, so `request.get(url, headers=headers)` becomes `request.get(url, headers=headers, verify=False)`


```python
import urllib.request

try:
    response = urllib.request.urlopen("https://www.google.com", timeout=10)
    if response.status != 200:
        print("Unable to reach google - there may be issues with your internet / VPN / firewall?")
    else:
        print("Connected to the internet and can reach Google")
except Exception as e:
    print(f"Failed to connect with this error: {e}")
```

## Another mention of occasional "gotchas" for PC people

There are 4 snafus on Windows to be aware of:  
1. Permissions. Please take a look at this [tutorial](https://chatgpt.com/share/67b0ae58-d1a8-8012-82ca-74762b0408b0) on permissions on Windows
2. Anti-virus, Firewall, VPN. These can interfere with installations and network access; try temporarily disabling them as needed
3. The evil Windows 260 character limit to filenames - here is a full [explanation and fix](https://chatgpt.com/share/67b0afb9-1b60-8012-a9f7-f968a5a910c7)!
4. If you've not worked with Data Science packages on your computer before, you might need to install Microsoft Build Tools. Here are [instructions](https://chatgpt.com/share/67b0b762-327c-8012-b809-b4ec3b9e7be0). A student also mentioned that [these instructions](https://github.com/bycloudai/InstallVSBuildToolsWindows) might be helpful for people on Windows 11.  

## And for Mac people

1. If you're new to developing on your Mac, you may need to install XCode developer tools. Here are [instructions](https://chatgpt.com/share/67b0b8d7-8eec-8012-9a37-6973b9db11f5).
2. As with PC people, Anti-virus, Firewall, VPN can be problematic. These can interfere with installations and network access; try temporarily disabling them as needed

# Step 1

Try running the next cell (click in the cell under this one and hit shift+return).

If this gives an error, then you're likely not running in an "activated" environment. Please check back in Part 5 of the SETUP guide for [PC](../SETUP-PC.md) or [Mac](../SETUP-mac.md) for setting up the Anaconda (or virtualenv) environment and activating it, before running `jupyter lab`.

If you look in the Anaconda prompt (PC) or the Terminal (Mac), you should see `(llms)` in your prompt where you launch `jupyter lab` - that's your clue that the llms environment is activated.

If you are in an activated environment, the next thing to try is to restart everything:
1. Close down all Jupyter windows, like this one
2. Exit all command prompts / Terminals / Anaconda
3. Repeat Part 5 from the SETUP instructions to begin a new activated environment and launch `jupyter lab` from the `llm_engineering` directory  
4. Come back to this notebook, and do Kernel menu >> Restart Kernel and Clear Outputs of All Cells
5. Try the cell below again.

If **that** doesn't work, then please contact me! I'll respond quickly, and we'll figure it out. Please run the diagnostics (last cell in this notebook) so I can debug. If you used Anaconda, it might be that for some reason your environment is corrupted, in which case the simplest fix is to use the virtualenv approach instead (Part 2B in the setup guides).


```python
# Some quick checks that your Conda environment or VirtualEnv is as expected
# The Environment Name should be: llms

import os
conda_name, venv_name = "", ""

conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    print("Anaconda environment is active:")
    print(f"Environment Path: {conda_prefix}")
    conda_name = os.path.basename(conda_prefix)
    print(f"Environment Name: {conda_name}")

virtual_env = os.environ.get('VIRTUAL_ENV')
if virtual_env:
    print("Virtualenv is active:")
    print(f"Environment Path: {virtual_env}")
    venv_name = os.path.basename(virtual_env)
    print(f"Environment Name: {venv_name}")

if conda_name != "llms" and venv_name != "llms" and venv_name != "venv" and venv_name != ".venv":
    print("Neither Anaconda nor Virtualenv seem to be activated with the expected name 'llms' or 'venv' or '.venv'")
    print("Did you run 'jupyter lab' from an activated environment with (llms) showing on the command line?")
    print("If in doubt, close down all jupyter lab, and follow Part 5 in the SETUP-PC or SETUP-mac guide.")
```

# Step 1.1

## It's time to check that the environment is good and dependencies are installed

And now, this next cell should run with no output - no import errors.  

For people on the new version of the course (October 2025 on) - an error would suggest that you don't have the right kernel.

For people on the original course:

> Import errors might indicate that you started jupyter lab without your environment activated? See SETUP Part 5.  
> Or you might need to restart your Kernel and Jupyter Lab.  
> Or it's possible that something is wrong with Anaconda. If so, here are some recovery instructions:  
> First, close everything down and restart your computer.  
> Then in an Anaconda Prompt (PC) or Terminal (Mac), from an activated environment, with **(llms)** showing in the prompt, from the llm_engineering directory, run this:  
> `python -m pip install --upgrade pip`  
> `pip install --retries 5 --timeout 15 --no-cache-dir --force-reinstall -r requirements.txt`  
> Watch carefully for any errors, and let me know.  
> If you see instructions to install Microsoft Build Tools, or Apple XCode tools, then follow the instructions.  
> Then try again!
> Finally, if that doesn't work, please try SETUP Part 2B, the alternative to Part 2 (with Python 3.11 or Python 3.12).  
> If you're unsure, please run the diagnostics (last cell in this notebook) and then email me at ed@edwarddonner.com


```python
# This import should work if your environment is active and dependencies are installed!

from openai import OpenAI
```

# Step 2

Let's check your .env file exists and has the OpenAI key set properly inside it.  
Please run this code and check that it prints a successful message, otherwise follow its instructions.

If it isn't successful, then it's not able to find a file called `.env` in the `llm_engineering` folder.  
The name of the file must be exactly `.env` - it won't work if it's called `my-keys.env` or `.env.doc`.  
Is it possible that `.env` is actually called `.env.txt`? In Windows, you may need to change a setting in the File Explorer to ensure that file extensions are showing ("Show file extensions" set to "On"). You should also see file extensions if you type `dir` in the `llm_engineering` directory.

Nasty gotchas to watch out for:  
- In the .env file, there should be no space between the equals sign and the key. Like: `OPENAI_API_KEY=sk-proj-...`
- If you copied and pasted your API key from another application, make sure that it didn't replace hyphens in your key with long dashes  

Note that the `.env` file won't show up in your Jupyter Lab file browser, because Jupyter hides files that start with a dot for your security; they're considered hidden files. If you need to change the name, you'll need to use a command terminal or File Explorer (PC) / Finder Window (Mac). Ask ChatGPT if that's giving you problems, or email me!

If you're having challenges creating the `.env` file, we can also do it with code! See the cell after the next one.

It's important to launch `jupyter lab` from the project root directory, `llm_engineering`. If you didn't do that, this cell might give you problems.


```python
from pathlib import Path

parent_dir = Path("..")
env_path = parent_dir / ".env"

if env_path.exists() and env_path.is_file():
    print(".env file found.")

    # Read the contents of the .env file
    with env_path.open("r") as env_file:
        contents = env_file.readlines()

    key_exists = any(line.startswith("OPENAI_API_KEY=") for line in contents)
    good_key = any(line.startswith("OPENAI_API_KEY=sk-proj-") for line in contents)
    classic_problem = any("OPEN_" in line for line in contents)
    
    if key_exists and good_key:
        print("SUCCESS! OPENAI_API_KEY found and it has the right prefix")
    elif key_exists:
        print("Found an OPENAI_API_KEY although it didn't have the expected prefix sk-proj- \nPlease double check your key in the file..")
    elif classic_problem:
        print("Didn't find an OPENAI_API_KEY, but I notice that 'OPEN_' appears - do you have a typo like OPEN_API_KEY instead of OPENAI_API_KEY?")
    else:
        print("Didn't find an OPENAI_API_KEY in the .env file")
else:
    print(".env file not found in the llm_engineering directory. It needs to have exactly the name: .env")
    
    possible_misnamed_files = list(parent_dir.glob("*.env*"))
    
    if possible_misnamed_files:
        print("\nWarning: No '.env' file found, but the following files were found in the llm_engineering directory that contain '.env' in the name. Perhaps this needs to be renamed?")
        for file in possible_misnamed_files:
            print(file.name)
```

## Fallback plan - python code to create the .env file for you

Only run the next cell if you're having problems making the .env file.  
Replace the text in the first line of code with your key from OpenAI.


```python
# Only run this code in this cell if you want to have a .env file created for you!

# Put your key inside the quote marks
make_me_a_file_with_this_key = "put your key here inside these quotes.. it should start sk-proj-"

# Change this to True if you already have a .env file and you want me to replace it
overwrite_if_already_exists = False 

from pathlib import Path

parent_dir = Path("..")
env_path = parent_dir / ".env"

if env_path.exists() and not overwrite_if_already_exists:
    print("There is already a .env file - if you want me to create a new one, change the variable overwrite_if_already_exists to True above")
else:
    try:
        with env_path.open(mode='w', encoding='utf-8') as env_file:
            env_file.write(f"OPENAI_API_KEY={make_me_a_file_with_this_key}")
        print(f"Successfully created the .env file at {env_path}")
        if not make_me_a_file_with_this_key.startswith("sk-proj-"):
            print(f"The key that you provided started with '{make_me_a_file_with_this_key[:8]}' which is different to sk-proj- is that what you intended?")
        print("Now rerun the previous cell to confirm that the file is created and the key is correct.")
    except Exception as e:
        print(f"An error occurred while creating the .env file: {e}")
```

# Step 3

Now let's check that your API key is correct set up in your `.env` file, and available using the dotenv package.
Try running the next cell.


```python
# This should print your API key to the output - please follow the instructions that get printed

import os
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("No API key was found - please try Kernel menu >> Restart Kernel And Clear Outputs of All Cells")
elif not api_key.startswith("sk-proj-"):
    print(f"An API key was found, but it starts with {api_key[:8]} rather than sk-proj- please double check this is as expected.")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them")
else:
    print("API key found and looks good so far!")

if api_key:
    problematic_unicode_chars = ['\u2013', '\u2014', '\u201c', '\u201d', '\u2026', '\u2018', '\u2019']
    forbidden_chars = ["'", " ", "\n", "\r", '"']
    
    if not all(32 <= ord(char) <= 126 for char in api_key):
        print("Potential problem: there might be unprintable characters accidentally included in the key?")
    elif any(char in api_key for char in problematic_unicode_chars):
        print("Potential problem: there might be special characters, like long hyphens or curly quotes in the key - did you copy it via a word processor?")
    elif any(char in api_key for char in forbidden_chars):
        print("Potential problem: there are quote marks, spaces or empty lines in your key?")
    else:
        print("The API key contains valid characters")
    
print(f"\nHere is the key --> {api_key} <--")
print()
print("If this key looks good, please go to the Edit menu >> Clear Cell Output so that your key is no longer displayed here!")
```

## It should print some checks including something like:

`Here is the key --> sk-proj-blahblahblah <--`

If it didn't print a key, then hopefully it's given you enough information to figure this out. Or contact me!

There is a final fallback approach if you wish: you can avoid using .env files altogether, and simply always provide your API key manually.  
Whenever you see this in the code:  
`openai = OpenAI()`  
You can replace it with:  
`openai = OpenAI(api_key="sk-proj-xxx")`


# Step 4

Now run the below code and you will hopefully see that GPT can handle basic arithmetic!!

If not, see the cell below.


```python
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

my_api_key = os.getenv("OPENAI_API_KEY")

print(f"Using API key --> {my_api_key} <--")

openai = OpenAI()
completion = openai.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{"role":"user", "content": "What's 2+2?"}],
)
print(completion.choices[0].message.content)
print("Now go to Edit menu >> Clear Cell Output to remove the display of your key.")
```

## If the key was set correctly, and this still didn't work

### If there's an error from OpenAI about your key, or a Rate Limit Error, then there's something up with your API key!

First check [this webpage](https://platform.openai.com/settings/organization/billing/overview) to make sure you have a positive credit balance.
OpenAI requires that you have a positive credit balance and it has minimums, typically around $5 in local currency. My salespitch for OpenAI is that this is well worth it for your education: for less than the price of a music album, you will build so much valuable commercial experience. But it's not required for this course at all; the README has instructions to call free open-source models via Ollama whenever we use OpenAI.

OpenAI billing page with credit balance is here:   
https://platform.openai.com/settings/organization/billing/overview  
OpenAI can take a few minutes to enable your key after you top up your balance.  
A student outside the US mentioned that he needed to allow international payments on his credit card for this to work.  

It's unlikely, but if there's something wrong with your key, you could also try creating a new key (button on the top right) here:  
https://platform.openai.com/api-keys

### Check that you can use gpt-4o-mini from the OpenAI playground

To confirm that billing is set up and your key is good, you could try using gtp-4o-mini directly:  
https://platform.openai.com/playground/chat?models=gpt-4o-mini

### If there's a cert related error

If you encountered a certificates error like:  
`ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)`  
Then please replace:
`openai = OpenAI()`  
with:  
`import httpx`  
`openai = OpenAI(http_client=httpx.Client(verify=False))`  
And also please replace:  
`requests.get(url, headers=headers)`  
with:  
`requests.get(url, headers=headers, verify=False)`  
And if that works, you're in good shape. You'll just have to change the labs in the same way any time you hit this cert error.  
This approach isn't OK for production code, but it's fine for our experiments. You may need to contact IT support to understand whether there are restrictions in your environment.

## If all else fails:

(1) Try pasting your error into ChatGPT or Claude! It's amazing how often they can figure things out

(2) Try creating another key and replacing it in the .env file and rerunning!

(3) Contact me! Please run the diagnostics in the cell below, then email me your problems to ed@edwarddonner.com

Thanks so much, and I'm sorry this is giving you bother!

# Gathering Essential Diagnostic information

## Please run this next cell to gather some important data

Please run the next cell; it should take a minute or so to run. Most of the time is checking your network bandwidth.
Then email me the output of the last cell to ed@edwarddonner.com.  
Alternatively: this will create a file called report.txt - just attach the file to your email.


```python
# Run my diagnostics report to collect key information for debugging
# Please email me the results. Either copy & paste the output, or attach the file report.txt

!pip install -q requests speedtest-cli psutil setuptools
from diagnostics import Diagnostics
Diagnostics().run()
```


```python

```
