#!/usr/bin/env python
# coding: utf-8

# # A full business solution
#
# ## Now we will take our project from Day 1 to the next level
#
# ### BUSINESS CHALLENGE:
#
# Create a product that builds a Brochure for a company to be used for prospective clients, investors and potential recruits.
#
# We will be provided a company name and their primary website.
#
# See the end of this notebook for examples of real-world business applications.
#
# And remember: I'm always available if you have problems or ideas! Please do reach out.

# %%


# imports
# If these fail, please check you're running from an 'activated' environment with (llms) in the command prompt

import json
import os

from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from scraper import fetch_website_contents, fetch_website_links

# %%


# Initialize and constants

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

if api_key and api_key.startswith("sk-proj-") and len(api_key) > 10:
    print("API key looks good so far")
else:
    print(
        "There might be a problem with your API key? Please visit the troubleshooting notebook!"
    )

MODEL = "gpt-5-nano"
openai = OpenAI()


# %%


links = fetch_website_links("https://edwarddonner.com")
print(links)


# ## First step: Have GPT-5-nano figure out which links are relevant
#
# ### Use a call to gpt-5-nano to read the links on a webpage, and respond in structured JSON.
# It should decide which links are relevant, and replace relative links such as "/about" with "https://company.com/about".
# We will use "one shot prompting" in which we provide an example of how it should respond in the prompt.
#
# This is an excellent use case for an LLM, because it requires nuanced understanding. Imagine trying to code this without LLMs by parsing and analyzing the webpage - it would be very hard!
#
# Sidenote: there is a more advanced technique called "Structured Outputs" in which we require the model to respond according to a spec. We cover this technique in Week 8 during our autonomous Agentic AI project.

# %%


link_system_prompt = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""


# %%


def get_links_user_prompt(url):
    user_prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company,
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):

"""
    links = fetch_website_links(url)
    user_prompt += "\n".join(str(link) for link in links)
    return user_prompt


# %%


print(get_links_user_prompt("https://edwarddonner.com"))


# %%


def select_relevant_links(url):
    print(f"Selecting relevant links for {url} by calling {MODEL}")
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)},
        ],
        response_format={"type": "json_object"},
    )
    result = response.choices[0].message.content
    links = json.loads(result or "{}")
    print(f"Found {len(links['links'])} relevant links")
    return links


# %%


select_relevant_links("https://edwarddonner.com")


# %%


select_relevant_links("https://huggingface.co")


# ## Second step: make the brochure!
#
# Assemble all the details into another prompt to GPT-5-nano

# %%


def fetch_page_and_all_relevant_links(url):
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links["links"]:
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_website_contents(link["url"])
    return result


# %%


print(fetch_page_and_all_relevant_links("https://huggingface.co"))


# %%


brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""

# Or uncomment the lines below for a more humorous brochure - this demonstrates how easy it is to incorporate 'tone':

# brochure_system_prompt = """
# You are an assistant that analyzes the contents of several relevant pages from a company website
# and creates a short, humorous, entertaining, witty brochure about the company for prospective customers, investors and recruits.
# Respond in markdown without code blocks.
# Include details of company culture, customers and careers/jobs if you have the information.
# """


# %%


def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    user_prompt += fetch_page_and_all_relevant_links(url)
    user_prompt = user_prompt[:5_000]  # Truncate if more than 5,000 characters
    return user_prompt


# %%


get_brochure_user_prompt("HuggingFace", "https://huggingface.co")


# %%


def create_brochure(company_name, url):
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)},
        ],
    )
    result = response.choices[0].message.content
    display(Markdown(result))


# %%


create_brochure("HuggingFace", "https://huggingface.co")


# ## Finally - a minor improvement
#
# With a small adjustment, we can change this so that the results stream back from OpenAI,
# with the familiar typewriter animation

# %%


def stream_brochure(company_name, url):
    stream = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)},
        ],
        stream=True,
    )
    response = ""
    display_handle = display(Markdown(""), display_id=True)
    assert display_handle is not None
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        update_display(Markdown(response), display_id=display_handle.display_id)


# %%


stream_brochure("HuggingFace", "https://huggingface.co")


# %%


# Try changing the system prompt to the humorous version when you make the Brochure for Hugging Face:

stream_brochure("HuggingFace", "https://huggingface.co")


# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/business.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#181;">Business applications</h2>
#             <span style="color:#181;">In this exercise we extended the Day 1 code to make multiple LLM calls, and generate a document.
#
# This is perhaps the first example of Agentic AI design patterns, as we combined multiple calls to LLMs. This will feature more in Week 2, and then we will return to Agentic AI in a big way in Week 8 when we build a fully autonomous Agent solution.
#
# Generating content in this way is one of the very most common Use Cases. As with summarization, this can be applied to any business vertical. Write marketing content, generate a product tutorial from a spec, create personalized email content, and so much more. Explore how you can apply content generation to your business, and try making yourself a proof-of-concept prototype. See what other students have done in the community-contributions folder -- so many valuable projects -- it's wild!</span>
#         </td>
#     </tr>
# </table>

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#900;">Before you move to Week 2 (which is tons of fun)</h2>
#             <span style="color:#900;">Please see the week1 EXERCISE notebook for your challenge for the end of week 1. This will give you some essential practice working with Frontier APIs, and prepare you well for Week 2.</span>
#         </td>
#     </tr>
# </table>

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/resources.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#f71;">A reminder on 3 useful resources</h2>
#             <span style="color:#f71;">1. The resources for the course are available <a href="https://edwarddonner.com/2024/11/13/llm-engineering-resources/">here.</a><br/>
#             2. I'm on LinkedIn <a href="https://www.linkedin.com/in/eddonner/">here</a> and I love connecting with people taking the course!<br/>
#             3. I'm trying out X/Twitter and I'm at <a href="https://x.com/edwarddonner">@edwarddonner<a> and hoping people will teach me how it's done..
#             </span>
#         </td>
#     </tr>
# </table>

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/thankyou.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#090;">Finally! I have a special request for you</h2>
#             <span style="color:#090;">
#                 My editor tells me that it makes a MASSIVE difference when students rate this course on Udemy - it's one of the main ways that Udemy decides whether to show it to others. If you're able to take a minute to rate this, I'd be so very grateful! And regardless - always please reach out to me at ed@edwarddonner.com if I can help at any point.
#             </span>
#         </td>
#     </tr>
# </table>
