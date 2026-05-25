# %%

import json
import os
from urllib.parse import urljoin

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from scraper import fetch_website_contents, fetch_website_links

# %%
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
OPENAI = OpenAI(api_key=api_key)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


# Create a system prompt by including persona, process, rules, format, audience, and style.
RELEVANT_LINKS_SYSTEM_PROMPT = """
You are an Information Retrieval Specialist, skilled in identifying and extracting relevant links from a list of a company's website links.
Your task is to analyze the provided list of links and determine which ones are most relevant to the content of the company's website.

To determine relevance, consider the following criteria:
1. The link should be related to the main topic of the company's website.
2. The link should provide additional information or resources that complement the content of the company's website.
3. The link should not include privacy policies, terms of service, or other administrative pages.
4. The link can be a blog post, an about page, a product page, or any other page that adds value to the company's website's main content.

The output should in the following JSON format:
{
    "relevant_links": [
        {
            "type": "Type of the linked page (e.g., Blog, About Page, Product Page, etc.)",
            "url": "<URL of the linked page>"
        },
        ...
    ]
}

For example:
{
    "relevant_links": [
        {
            "type": "About Page",
            "url": "https://website.com/about"
        },
        {
            "type": "Blog",
            "url": "https://website.com/blog"
        }
    ]
}
"""

# Create a user prompt by including Actions and Information (tasks, history, and environment state)
RELEVANT_LINKS_USER_PROMPT = """
I am trying to make a sales brochure for the following company: {website_url}. I want to include content from relevant pages on the  company's website in the brochure,
I am providing you with a list of links of webpages from a company's website.
Your task is to analyze these links and identify which ones are relevant to the main content of the company's website.
Please return the relevant links.

Here are the links:

"""

BROCHURE_SYSTEM_PROMPT = """
You are a Sales Brochure Creator, skilled in crafting compelling and informative brochures that highlight the key features and benefits of a company.
Your task is to create a sales brochure based on the content of a company's website.
To create an effective sales brochure, consider the following criteria:
1. The brochure should highlight the main features and benefits of the company.
2. The brochure should be concise and easy to read, with clear headings and bullet points.
3. The brochure should be tailored to the target audience, addressing their needs and interests.
"""

BROCHURE_USER_PROMPT = """
I am trying to create a sales brochure for the following company: {website_url}.
I want to include content from the website in the brochure. Here is the content of the website:

"""


# %%
def generate_messages(
    system_prompt: str, user_prompt: str
) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


# %%
def relevant_links(links, url, client, model):
    user_prompt = RELEVANT_LINKS_USER_PROMPT.format(website_url=url) + "\n".join(links)
    messages = generate_messages(
        RELEVANT_LINKS_SYSTEM_PROMPT,
        user_prompt,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    result = response.choices[0].message.content
    if not result:
        print("No response from the model for relevant links.")
        return None
    parsed = json.loads(result)
    return parsed


# %%
def create_brochure(website_url: str, website_content: str, client, model):
    user_prompt = (
        BROCHURE_USER_PROMPT.format(website_url=website_url)
        + website_content
        + "\n\nPlease create a sales brochure based on the content of the website."
    )
    messages = generate_messages(
        BROCHURE_SYSTEM_PROMPT,
        user_prompt,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    brochure = response.choices[0].message.content or ""
    return brochure


# %%
def display_brochure(url: str, model: str):
    client = OPENAI if model == "gpt-4.1-mini" else OLLAMA
    links = fetch_website_links(url)
    relevant = relevant_links(links, url, client, model)
    contents = fetch_website_contents(url)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    if not relevant:
        print("No relevant links found.")
        return

    for link in relevant["relevant_links"]:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])
    for link in relevant.get("relevant_links", []):
        link_url = link.get("url")
        link_type = link.get("type", "Unknown")
        link_url = urljoin(url, link_url)
        result += f"\n\n### Link: {link_type}\n"
        result += fetch_website_contents(link_url)
    brochure = create_brochure(url, result, client, model)
    return brochure


url_input = gr.Textbox(label="Enter Website URL")
markdown_output = gr.Markdown(label="Brochure")
model_selection = gr.Dropdown(
    label="Select Model",
    choices=["gpt-4.1-mini", "llama3.2:3b"],
    value="gpt-4.1-mini",
)

view = gr.Interface(
    fn=display_brochure,
    inputs=[url_input, model_selection],
    outputs=[markdown_output],
    flagging_mode="never",
)
view.launch()
