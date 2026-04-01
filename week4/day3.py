#!/usr/bin/env python
# coding: utf-8

# # Code Generator
#
# The requirement: use a Frontier model to generate high performance C++ code from Python code
#

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/resources.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#f71;">Reminder: fetch latest code</h2>
#             <span style="color:#f71;">I'm continually improving these labs, adding more examples and exercises.
#             At the start of each week, it's worth checking you have the latest code.<br/>
#             First do a <a href="https://chatgpt.com/share/6734e705-3270-8012-a074-421661af6ba9">git pull and merge your changes as needed</a>. Any problems? Try asking ChatGPT to clarify how to merge - or contact me!<br/><br/>
#             After you've pulled the code, from the llm_engineering directory, in a Cursor Terminal, run:<br/>
#             <code>uv sync</code><br/>
#             </span>
#         </td>
#     </tr>
# </table>

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h1 style="color:#900;">Important Note</h1>
#             <span style="color:#900;">
#             In this lab, I use high end models GPT 5, Claude 4.5 Sonnet, Gemini 2.5 Pro, Grok 4, which are the slightly higher priced models. The costs are still low, but if you'd prefer to keep costs ultra low, please pick lower cost models like gpt-5-nano.
#             </span>
#         </td>
#     </tr>
# </table>

# %%


# imports

import os
import subprocess

from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI
from system_info import retrieve_system_info

# %%


load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
grok_api_key = os.getenv("GROK_API_KEY")

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set (and this is optional)")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:2]}")
else:
    print("Google API Key not set (and this is optional)")

if grok_api_key:
    print(f"Grok API Key exists and begins {grok_api_key[:4]}")
else:
    print("Grok API Key not set (and this is optional)")


# %%


# Connect to client libraries

openai = OpenAI()

anthropic_url = "https://api.anthropic.com/v1/"
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
grok_url = "https://api.x.ai/v1"

anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)
grok = OpenAI(api_key=grok_api_key, base_url=grok_url)


# %%


OPENAI_MODEL = "gpt-5"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
GROK_MODEL = "grok-4"
GEMINI_MODEL = "gemini-2.5-pro"

# Want to keep costs ultra-low? Uncomment these lines:

# OPENAI_MODEL = "gpt-5-nano"
# CLAUDE_MODEL = "claude-haiku-4-5"
# GROK_MODEL = "grok-4-fast-non-reasoning"
# GEMINI_MODEL = "gemini-2.5-flash-lite"


# ## PLEASE NOTE:
#
# We will be writing a solution to convert Python into efficient, optimized C++ code for your machine, which can be compiled to native machine code and executed.
#
# It is not necessary for you to execute the code yourself - that's not the point of the exercise!
#
# But if you would like to (because it's satisfying!) then I'm including the steps here. Very optional!
#
# As an alternative, I'll also show you a website where you can run the C++ code.

# %%


system_info = retrieve_system_info()
display(system_info)


# %%


message = f"""
Here is a report of the system information for my computer.
I want to run a C++ compiler to compile a single C++ file called main.cpp and then execute it in the simplest way possible.
Please reply with whether I need to install any C++ compiler to do this. If so, please provide the simplest step by step instructions to do so.

If I'm already set up to compile C++ code, then I'd like to run something like this in Python to compile and execute the code:
```python
compile_command = # something here - to achieve the fastest possible runtime performance
compile_result = subprocess.run(compile_command, check=True, text=True, capture_output=True)
run_command = # something here
run_result = subprocess.run(run_command, check=True, text=True, capture_output=True)
return run_result.stdout
```
Please tell me exactly what I should use for the compile_command and run_command.

System information:
{system_info}
"""

response = openai.chat.completions.create(
    model=OPENAI_MODEL, messages=[{"role": "user", "content": message}]
)
display(Markdown(response.choices[0].message.content))


# ## If you need to install something
#
# If you would like to, please follow GPTs instructions! Then rerun the analysis afterwards (you might need to Restart the notebook) to confirm you're set.
#
# You should now be equipped with the command to compile the code, and the command to run it!
#
# Enter that in the cell below:

# %%


compile_command = [
    "clang++",
    "-std=c++17",
    "-Ofast",
    "-mcpu=native",
    "-flto=thin",
    "-fvisibility=hidden",
    "-DNDEBUG",
    "main.cpp",
    "-o",
    "main",
]
run_command = ["./main"]


# ## And now, on with the main task

# %%


system_prompt = """
Your task is to convert Python code into high performance C++ code.
Respond only with C++ code. Do not provide any explanation other than occasional comments.
The C++ response needs to produce an identical output in the fastest possible time.
"""


def user_prompt_for(python):
    return f"""
Port this Python code to C++ with the fastest possible implementation that produces identical output in the least time.
The system information is:
{system_info}
Your response will be written to a file called main.cpp and then compiled and executed; the compilation command is:
{compile_command}
Respond only with C++ code.
Python code to port:

```python
{python}
```
"""


# %%


def messages_for(python):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(python)},
    ]


# %%


def write_output(cpp):
    with open("main.cpp", "w", encoding="utf-8") as f:
        f.write(cpp)


# %%


def port(client, model, python):
    reasoning_effort = "high" if "gpt" in model else None
    response = client.chat.completions.create(
        model=model, messages=messages_for(python), reasoning_effort=reasoning_effort
    )
    reply = response.choices[0].message.content
    reply = reply.replace("```cpp", "").replace("```", "")
    write_output(reply)


# %%


pi = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(200_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""


# %%


def run_python(code):
    globals = {"__builtins__": __builtins__}
    exec(code, globals)


# %%


run_python(pi)


# %%


port(openai, OPENAI_MODEL, pi)


# # Compiling C++ and executing
#
# This next cell contains the command to compile a C++ file based on the instructions from GPT.
#
# Again, it's not crucial to do this step if you don't wish to!
#
# OR alternatively: student Sandeep K.G. points out that you can run Python and C++ code online to test it out that way. Thank you Sandeep!
# > Not an exact comparison but you can still get the idea of performance difference.
# > For example here: https://www.programiz.com/cpp-programming/online-compiler/

# %%


# Use the commands from GPT 5


def compile_and_run():
    subprocess.run(compile_command, check=True, text=True, capture_output=True)
    print(
        subprocess.run(run_command, check=True, text=True, capture_output=True).stdout
    )
    print(
        subprocess.run(run_command, check=True, text=True, capture_output=True).stdout
    )
    print(
        subprocess.run(run_command, check=True, text=True, capture_output=True).stdout
    )


# %%


compile_and_run()


# %%


# ## OK let's try the other contenders!

# %%


port(anthropic, CLAUDE_MODEL, pi)
compile_and_run()


# %%


port(grok, GROK_MODEL, pi)
compile_and_run()


# %%


port(gemini, GEMINI_MODEL, pi)
compile_and_run()


#
#

# %%


print(f"""
In Ed's experiments, the performance speedups were:

4th place: Claude Sonnet 4.5: {19.178207 / 0.104241:.0f}X speedup
3rd place: GPT-5: {19.178207 / 0.082168:.0f}X speedup
2nd place: Grok 4: {19.178207 / 0.018092:.0f}X speedup
1st place: Gemini 2.5 Pro: {19.178207 / 0.013314:.0f}X speedup
""")


# %%


# %%
