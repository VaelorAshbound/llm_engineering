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
#             <h2 style="color:#f71;">Reminder: OPTIONAL to execute C++ code or Rust code</h2>
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
#             In this lab, I use high end models GPT 5, Claude 4.5 Sonnet, Gemini 2.5 Pro, Grok 4, which are the slightly higher priced models. The costs are still low, but if you'd prefer to keep costs ultra low, please pick lower cost models like gpt-5-nano.
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
from IPython.display import Markdown, display
from openai import OpenAI
from styles import CSS
from system_info import retrieve_system_info, rust_toolchain_info

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
rust_info = rust_toolchain_info()
display(rust_info)


# %%


message = f"""
Here is a report of the system information for my computer.
I want to run a Rust compiler to compile a single rust file called main.rs and then execute it in the simplest way possible.
Please reply with whether I need to install a Rust toolchain to do this. If so, please provide the simplest step by step instructions to do so.

If I'm already set up to compile Rust code, then I'd like to run something like this in Python to compile and execute the code:
```python
compile_command = # something here - to achieve the fastest possible runtime performance
compile_result = subprocess.run(compile_command, check=True, text=True, capture_output=True)
run_command = # something here
run_result = subprocess.run(run_command, check=True, text=True, capture_output=True)
return run_result.stdout
```
Please tell me exactly what I should use for the compile_command and run_command.
Have the maximum possible runtime performance in mind; compile time can be slow. Fastest possible runtime performance for this platform is key.
Reply with the commands in markdown.

System information:
{system_info}

Rust toolchain information:
{rust_info}
"""

response = openai.chat.completions.create(
    model=models[0], messages=[{"role": "user", "content": message}]
)
display(Markdown(response.choices[0].message.content))


# ## For C++, overwrite this with the commands from yesterday, or for Rust, use the new commands
#
# Or just use the website like yesterday:
#
#  https://www.programiz.com/cpp-programming/online-compiler/

# %%


compile_command = [
    "/Users/ed/.cargo/bin/rustc",
    "main.rs",
    "-C",
    "opt-level=3",
    "-C",
    "target-cpu=native",
    "-C",
    "codegen-units=1",
    "-C",
    "lto=fat",
    "-C",
    "panic=abort",
    "-C",
    "strip=symbols",
    "-o",
    "main",
]

run_command = ["./main"]


# ## And now, on with the main task

# %%


language = "Rust"  # or "C++"
extension = "rs" if language == "Rust" else "cpp"

system_prompt = f"""
Your task is to convert Python code into high performance {language} code.
Respond only with {language} code. Do not provide any explanation other than occasional comments.
The {language} response needs to produce an identical output in the fastest possible time.
"""


def user_prompt_for(python):
    return f"""
Port this Python code to {language} with the fastest possible implementation that produces identical output in the least time.
The system information is:
{system_info}
Your response will be written to a file called main.{language} and then compiled and executed; the compilation command is:
{compile_command}
Respond only with {language} code.
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


def write_output(code):
    with open(f"main.{extension}", "w") as f:
        f.write(code)


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
    reply = reply.replace("```cpp", "").replace("```rust", "").replace("```", "")
    return reply


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


# Use the commands from GPT 5


def compile_and_run(code):
    write_output(code)
    try:
        subprocess.run(compile_command, check=True, text=True, capture_output=True)
        run_result = subprocess.run(
            run_command, check=True, text=True, capture_output=True
        )
        return run_result.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred:\n{e.stderr}"


# %%


python_hard = """# Be careful to support large numbers

def lcg(seed, a=1664525, c=1013904223, m=2**32):
    value = seed
    while True:
        value = (a * value + c) % m
        yield value

def max_subarray_sum(n, seed, min_val, max_val):
    lcg_gen = lcg(seed)
    random_numbers = [next(lcg_gen) % (max_val - min_val + 1) + min_val for _ in range(n)]
    max_sum = float('-inf')
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += random_numbers[j]
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum

def total_max_subarray_sum(n, initial_seed, min_val, max_val):
    total_sum = 0
    lcg_gen = lcg(initial_seed)
    for _ in range(20):
        seed = next(lcg_gen)
        total_sum += max_subarray_sum(n, seed, min_val, max_val)
    return total_sum

# Parameters
n = 10000         # Number of random numbers
initial_seed = 42 # Initial seed for the LCG
min_val = -10     # Minimum value of random numbers
max_val = 10      # Maximum value of random numbers

# Timing the function
import time
start_time = time.time()
result = total_max_subarray_sum(n, initial_seed, min_val, max_val)
end_time = time.time()

print("Total Maximum Subarray Sum (20 runs):", result)
print("Execution Time: {:.6f} seconds".format(end_time - start_time))
"""


# %%


with gr.Blocks(
    css=CSS,
    theme=gr.themes.Monochrome(),  # type: ignore[attr-defined]
    title=f"Port from Python to {language}",
) as ui:
    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            python = gr.Code(
                label="Python (original)",
                value=python_hard,
                language="python",
                lines=26,
            )
        with gr.Column(scale=6):
            cpp = gr.Code(
                label=f"{language} (generated)", value="", language="cpp", lines=26
            )

    with gr.Row(elem_classes=["controls"]):
        python_run = gr.Button("Run Python", elem_classes=["run-btn", "py"])
        model = gr.Dropdown(models, value=models[0], show_label=False)
        convert = gr.Button(f"Port to {language}", elem_classes=["convert-btn"])
        cpp_run = gr.Button(f"Run {language}", elem_classes=["run-btn", "cpp"])

    with gr.Row(equal_height=True):
        with gr.Column(scale=6):
            python_out = gr.TextArea(
                label="Python result", lines=8, elem_classes=["py-out"]
            )
        with gr.Column(scale=6):
            cpp_out = gr.TextArea(
                label=f"{language} result", lines=8, elem_classes=["cpp-out"]
            )

    convert.click(fn=port, inputs=[model, python], outputs=[cpp])
    python_run.click(fn=run_python, inputs=[python], outputs=[python_out])
    cpp_run.click(fn=compile_and_run, inputs=[cpp], outputs=[cpp_out])

ui.launch(inbrowser=True)


# ## RESULTS!
#
# Qwen 2.5 Coder: FAIL
# Gemini 2.5 Pro: FAIL
# DeepSeek Coder v2: FAIL
# Qwen3 Coder 30B: FAIL
# Claude Sonnet 4.5: FAIL
# GPT-5: FAIL
#
# 3rd place: GPT-oss-20B: 0.000341
# 2nd place: Grok 4: 0.000317
# **1st place: OpenAI GPT-OSS 120B: 0.000304**

# %%


print(
    f"In Ed's experimenet, the GPT-OSS 120B model outcome is {33.755209 / 0.000304:,.0f} times faster than the Python code."
)


# %%
