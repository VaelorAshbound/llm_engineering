#!/usr/bin/env python
# coding: utf-8

# # Gradio Day!
#
# Today we will build User Interfaces using the outrageously simple Gradio framework.
#
# Prepare for joy!
#
# Please note: your Gradio screens may appear in 'dark mode' or 'light mode' depending on your computer settings.

# %%


import os

# %%
import gradio as gr  # oh yeah!
from dotenv import load_dotenv
from openai import OpenAI
from scraper import fetch_website_contents

# %%


# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging
# You can choose whichever providers you like - or all Ollama

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")


# %%


# Connect to OpenAI, Anthropic and Google; comment out the Claude or Google lines if you're not using them

openai = OpenAI()

anthropic_url = "https://api.anthropic.com/v1/"
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)


# %%


# Let's wrap a call to GPT-4.1-mini in a simple function

system_message = "You are a helpful assistant"


def message_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    response = openai.chat.completions.create(model="gpt-4.1-mini", messages=messages)  # type: ignore[arg-type]
    return response.choices[0].message.content


# %%


# This can reveal the "training cut off", or the most recent date in the training data

message_gpt("What is today's date?")


# ## User Interface time!

# %%


# here's a simple function


def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()


# %%


shout("hello")


# %%


gr.Interface(
    fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never"
).launch()


# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#900;">NOTE: Using Gradio's Share tool</h2>
#             <span style="color:#900;">I'm about to show you a really cool way to share your Gradio UI with others. This deploys your gradio app as a demo on gradio's website, and then allows gradio to call the 'shout' function. This uses an advanced technology known as 'HTTP tunneling' (like ngrok for people who know it) which isn't allowed by many Antivirus programs and corporate environments. If you get an error, just skip the next cell.<br/>
#             </span>
#         </td>
#     </tr>
# </table>

# %%


# Adding share=True means that it can be accessed publically
# A more permanent hosting is available using a platform called Spaces from HuggingFace, which we will touch on next week
# NOTE: Some Anti-virus software and Corporate Firewalls might not like you using share=True.
# If you're at work on on a work network, I suggest skip this test.

gr.Interface(
    fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never"
).launch(share=True)


# %%


# Adding inbrowser=True opens up a new browser window automatically

gr.Interface(
    fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never"
).launch(inbrowser=True)


# ## Adding authentication
#
# Gradio makes it very easy to have userids and passwords
#
# Obviously if you use this, have it look properly in a secure place for passwords! At a minimum, use your .env

# %%


gr.Interface(
    fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never"
).launch(inbrowser=True, auth=("ed", "bananas"))


# ## Forcing dark mode
#
# Gradio appears in light mode or dark mode depending on the settings of the browser and computer. There is a way to force gradio to appear in dark mode, but Gradio recommends against this as it should be a user preference (particularly for accessibility reasons). But if you wish to force dark mode for your screens, below is how to do it.

# %%


# Define this variable and then pass js=force_dark_mode when creating the Interface

force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
gr.Interface(
    fn=shout,
    inputs="textbox",
    outputs="textbox",
    flagging_mode="never",
    js=force_dark_mode,
).launch()


# %%


# Adding a little more:

message_input = gr.Textbox(
    label="Your message:", info="Enter a message to be shouted", lines=7
)
message_output = gr.Textbox(label="Response:", lines=8)

view = gr.Interface(
    fn=shout,
    title="Shout",
    inputs=[message_input],
    outputs=[message_output],
    examples=["hello", "howdy"],
    flagging_mode="never",
)
view.launch()


# %%


# And now - changing the function from "shout" to "message_gpt"

message_input = gr.Textbox(
    label="Your message:", info="Enter a message for GPT-4.1-mini", lines=7
)
message_output = gr.Textbox(label="Response:", lines=8)

view = gr.Interface(
    fn=message_gpt,
    title="GPT",
    inputs=[message_input],
    outputs=[message_output],
    examples=["hello", "howdy"],
    flagging_mode="never",
)
view.launch()


# %%


# Let's use Markdown
# Are you wondering why it makes any difference to set system_message when it's not referred to in the code below it?
# I'm taking advantage of system_message being a global variable, used back in the message_gpt function (go take a look)
# Not a great software engineering practice, but quite common during Jupyter Lab R&D!

system_message = (
    "You are a helpful assistant that responds in markdown without code blocks"
)

message_input = gr.Textbox(
    label="Your message:", info="Enter a message for GPT-4.1-mini", lines=7
)
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=message_gpt,
    title="GPT",
    inputs=[message_input],
    outputs=[message_output],
    examples=[
        "Explain the Transformer architecture to a layperson",
        "Explain the Transformer architecture to an aspiring AI engineer",
    ],
    flagging_mode="never",
)
view.launch()


# %%


# Let's create a call that streams back results
# If you'd like a refresher on Generators (the "yield" keyword),
# Please take a look at the Intermediate Python guide in the guides folder


def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    stream = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        stream=True,  # type: ignore[call-overload]
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


# %%


message_input = gr.Textbox(
    label="Your message:", info="Enter a message for GPT-4.1-mini", lines=7
)
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_gpt,
    title="GPT",
    inputs=[message_input],
    outputs=[message_output],
    examples=[
        "Explain the Transformer architecture to a layperson",
        "Explain the Transformer architecture to an aspiring AI engineer",
    ],
    flagging_mode="never",
)
view.launch()


# %%


def stream_claude(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    stream = anthropic.chat.completions.create(
        model="claude-sonnet-4-5-20250929",
        messages=messages,
        stream=True,  # type: ignore[call-overload]
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


# %%


message_input = gr.Textbox(
    label="Your message:", info="Enter a message for Claude 4.5 Sonnet", lines=7
)
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_claude,
    title="Claude",
    inputs=[message_input],
    outputs=[message_output],
    examples=[
        "Explain the Transformer architecture to a layperson",
        "Explain the Transformer architecture to an aspiring AI engineer",
    ],
    flagging_mode="never",
)
view.launch()


# ## And now getting fancy
#
# Remember to check the Intermediate Python Guide if you're unsure about generators and "yield"

# %%


def stream_model(prompt, model):
    if model == "GPT":
        result = stream_gpt(prompt)
    elif model == "Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result


# %%


message_input = gr.Textbox(
    label="Your message:", info="Enter a message for the LLM", lines=7
)
model_selector = gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT")
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_model,
    title="LLMs",
    inputs=[message_input, model_selector],
    outputs=[message_output],
    examples=[
        ["Explain the Transformer architecture to a layperson", "GPT"],
        ["Explain the Transformer architecture to an aspiring AI engineer", "Claude"],
    ],
    flagging_mode="never",
)
view.launch()


# # Building a company brochure generator
#
# Now you know how - it's simple!

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#900;">Before you read the next few cells</h2>
#             <span style="color:#900;">
#                 Try to do this yourself - go back to the company brochure in week1, day5 and add a Gradio UI to the end. Then come and look at the solution.
#             </span>
#         </td>
#     </tr>
# </table>

# %%


# %%


# Again this is typical Experimental mindset - I'm changing the global variable we used above:

system_message = """
You are an assistant that analyzes the contents of a company website landing page
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
"""


# %%


def stream_brochure(company_name, url, model):
    yield ""
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += fetch_website_contents(url)
    if model == "GPT":
        result = stream_gpt(prompt)
    elif model == "Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result


# %%


name_input = gr.Textbox(label="Company name:")
url_input = gr.Textbox(label="Landing page URL including http:// or https://")
model_selector = gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT")
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_brochure,
    title="Brochure Generator",
    inputs=[name_input, url_input, model_selector],
    outputs=[message_output],
    examples=[
        ["Hugging Face", "https://huggingface.co", "GPT"],
        ["Edward Donner", "https://edwarddonner.com", "Claude"],
    ],
    flagging_mode="never",
)
view.launch()


# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/resources.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#f71;">Gradio Resources</h2>
#             <span style="color:#f71;">If you'd like to go deeper on Gradio, check out the amazing documentation - a wonderful rabbit hole.<br/>
#             <a href="https://www.gradio.app/guides/quickstart">https://www.gradio.app/guides/quickstart</a><br/>Gradio is primarily designed for Demos, Prototypes and MVPs, but I've also used it frequently to make internal apps for power users.
#             </span>
#         </td>
#     </tr>
# </table>
