# %%
# import os
# from typing import cast

from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# %%
load_dotenv(override=True)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
# api_key = os.getenv("OPENAI_API_KEY")
# openai = OpenAI(api_key=cast(str, api_key))
ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# %%
EMAIL = """
Respected Professor,Hope you are doing well. I am writing to request you to write and sign a recommendation letter for me.
I am applying for a Master's program abroad and I require a recommendation letter from you to support my application.
I am currently in Bahrain working as a software enginner therefore I am unable to meet you in person to discuss this matter.
My fellow colleague will be visiting you so can you please give him the recommendation letter on my behalf.
I have attached my resume and a draft of the recommendation letter for your reference.
I hope you will consider my request and I would be grateful for your support. Please let me know if you need any additional information from me.
Thank you for your time and consideration.
Best regards,
"""

# Create a system prompt by including persona, process, rules, format, audience, and style.
SYSTEM_PROMPT = """ You are an Email Specialist who is skilled in rephrasing, summarizing, and creating subject lines for emails.
Your task is to take the provided email and rephrase it to make it more concise and professional, while retaining the original meaning.
Additionally, you will generate a suitable subject line for the email. The email should be clear, polite, and to the point, making it easier for the recipient to understand the request and respond accordingly.
Also generate a summary of the email in 2-3 sentences. The summary should capture the main points of the email and provide a clear overview of the content.
Use clear and concise language, and ensure that the email is well-structured and easy to read. The tone should be professional and respectful, while still conveying the urgency of the request.
"""

# Create a user prompt by including Actions and Information (tasks, history, and environment state)
USER_PROMPT = """
Email:
{email}

Tasks:
1. Rephrase the email professionally.
2. Generate a subject line.
3. Provide a 2–3 sentence summary.

Output Format:
Subject:
<subject line>

Rewritten Email:
<email>

Summary:
<summary>
"""


# %%
def generate_messages(
    system_prompt: str, user_prompt: str, email: str
) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(email=email)},
    ]


# %%
def process_email(system_prompt: str, user_prompt: str, email: str) -> str:
    messages = generate_messages(system_prompt, user_prompt, email)
    response = ollama.chat.completions.create(
        model="llama3.2:3b",
        messages=messages,
    )

    return response.choices[0].message.content or ""


# %%
def display_processed_email(email: str):
    processed_email = process_email(SYSTEM_PROMPT, USER_PROMPT, email)
    display(Markdown(processed_email))


# %%
display_processed_email(EMAIL)
