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
#             <h2 style="color:#f71;">Reminder: OPTIONAL to execute C++ code</h2>
#             <span style="color:#f71;">As an alternative, you can run it on the website given yesterday</span>
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
#             In this lab, I use free open source models on Ollama. I also use paid open-source models via Groq and OpenRouter. Only pick the models you want to!
#             </span>
#         </td>
#     </tr>
# </table>

# %%


# imports

import io
import os
import subprocess
import sys

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from system_info import retrieve_system_info

# %%


load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
grok_api_key = os.getenv("GROK_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

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

if groq_api_key:
    print(f"Groq API Key exists and begins {groq_api_key[:4]}")
else:
    print("Groq API Key not set (and this is optional)")

if openrouter_api_key:
    print(f"OpenRouter API Key exists and begins {openrouter_api_key[:6]}")
else:
    print("OpenRouter API Key not set (and this is optional)")


# %%


# Connect to client libraries

openai = OpenAI()

anthropic_url = "https://api.anthropic.com/v1/"
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
grok_url = "https://api.x.ai/v1"
groq_url = "https://api.groq.com/openai/v1"
ollama_url = "http://localhost:11434/v1"
openrouter_url = "https://openrouter.ai/api/v1"

anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)
grok = OpenAI(api_key=grok_api_key, base_url=grok_url)
groq = OpenAI(api_key=groq_api_key, base_url=groq_url)
ollama = OpenAI(api_key="ollama", base_url=ollama_url)
openrouter = OpenAI(api_key=openrouter_api_key, base_url=openrouter_url)


# %%


models = [
    "gpt-5",
    "claude-sonnet-4-5-20250929",
    "grok-4",
    "gemini-2.5-pro",
    "qwen2.5-coder",
    "deepseek-coder-v2",
    "gpt-oss:20b",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "openai/gpt-oss-120b",
]

clients = {
    "gpt-5": openai,
    "claude-sonnet-4-5-20250929": anthropic,
    "grok-4": grok,
    "gemini-2.5-pro": gemini,
    "openai/gpt-oss-120b": groq,
    "qwen2.5-coder": ollama,
    "deepseek-coder-v2": ollama,
    "gpt-oss:20b": ollama,
    "qwen/qwen3-coder-30b-a3b-instruct": openrouter,
}

# Want to keep costs ultra-low? Replace this with models of your choice, using the examples from yesterday


# %%


system_info = retrieve_system_info()
print(system_info)


# ## Overwrite this with the commands from yesterday
#
# Or just use the website like yesterday:
#
#  https://www.programiz.com/cpp-programming/online-compiler/

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
    with open("main.cpp", "w") as f:
        f.write(cpp)


# %%


def port(model, python):
    client = clients[model]
    reasoning_effort = "high" if "gpt" in model else None
    response = client.chat.completions.create(
        model=model,
        messages=messages_for(python),  # type: ignore[arg-type]
        reasoning_effort=reasoning_effort,
    )
    reply = response.choices[0].message.content or ""
    reply = reply.replace("```cpp", "").replace("```", "")
    write_output(reply)
    return reply


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
    globals_dict = {"__builtins__": __builtins__}

    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    try:
        exec(code, globals_dict)
        output = buffer.getvalue()
    except Exception as e:
        output = f"Error: {e}"
    finally:
        sys.stdout = old_stdout

    return output


# %%


def compile_and_run():
    try:
        subprocess.run(compile_command, check=True, text=True, capture_output=True)
        print(
            subprocess.run(
                run_command, check=True, text=True, capture_output=True
            ).stdout
        )
        print(
            subprocess.run(
                run_command, check=True, text=True, capture_output=True
            ).stdout
        )
        print(
            subprocess.run(
                run_command, check=True, text=True, capture_output=True
            ).stdout
        )
    except subprocess.CalledProcessError as e:
        print(f"An error occurred:\n{e.stderr}")


# %%


with gr.Blocks() as ui:
    with gr.Row():
        python = gr.Textbox(label="Python code:", lines=28, value=pi)
        cpp = gr.Textbox(label="C++ code:", lines=28)
    with gr.Row():
        model = gr.Dropdown(models, label="Select model", value=models[0])
        convert = gr.Button("Convert code")

    convert.click(port, inputs=[model, python], outputs=[cpp])

ui.launch(inbrowser=True)


# %%


# %%


compile_and_run()


# Qwen 2.5 Coder: Fail
# DeepSeek Coder v2: 0.114050084
# OpenAI gpt-oss 20B: 0.080438
# Qwen 30B: 0.113734
# OpenAI gpt-oss 120B: 1.407383
#
#
#

# In Ed's experiments, the performance speedups were:
#
# 9th place: Qwen 2.5 Coder: Fail
# 8th place: OpenAI GPT-OSS 120B: 14X speedup
# 7th place: DeepSeek Coder v2: 168X speedup
# 6th place: Qwen3 Coder 30B: 168X speedup
# 5th place: Claude Sonnet 4.5: 184X speedup
# 4th place: GPT-5: 233X speedup
# **3rd place: oss-20B: 238X speedup**
# 2nd place: Grok 4: 1060X speedup
# 1st place: Gemini 2.5 Pro: 1440X speedup

#
