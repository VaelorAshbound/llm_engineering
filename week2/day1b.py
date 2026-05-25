# %%

import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=api_key)

RUBY_SYSTEM_PROMPT = """
You are Ruby. You are optimistic, kind-hearted, and driven by a genuine desire to be a hero.
"""

WEISS_SYSTEM_PROMPT = """
You are Weiss. You are disciplined, intelligent, and initially proud due to your privileged upbringing.
"""

BLAKE_SYSTEM_PROMPT = """
You are Blake. You are quiet, introspective, and burdened by you past and ideals of justice.
"""

YANG_SYSTEM_PROMPT = """
You are Yang. You are confident, outgoing, and fiercely protective of those you love.
"""

USER_PROMPT = """
You are {you} in a conversation with {person1}, {person2} and {person3}. Here is the conversation so far:
{conversation}
Now with this, respond with what you would like to say next, as {you}
Respond in the following format:
"{you}": "<what you would like to say next>"
for example, if you are Weiss, and you want to say "I am Weiss", you would respond with:
"Weiss": "I am Weiss"
"""

characters = ["Weiss", "Blake", "Yang", "Ruby"]

history = []


def gpt_response(message: list[ChatCompletionMessageParam]):
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=message,
    )
    result = response.choices[0].message.content or ""
    history.append(result)


def generate_response(you, person1, person2, person3):
    if you == "Ruby":
        system_prompt = RUBY_SYSTEM_PROMPT
    elif you == "Weiss":
        system_prompt = WEISS_SYSTEM_PROMPT
    elif you == "Blake":
        system_prompt = BLAKE_SYSTEM_PROMPT
    elif you == "Yang":
        system_prompt = YANG_SYSTEM_PROMPT
    else:
        raise ValueError("Invalid character name")

    message: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                you=you,
                person1=person1,
                person2=person2,
                person3=person3,
                conversation="\n".join(history),
            ),
        },
    ]
    gpt_response(message)


def initial_response():
    message: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": RUBY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": """
                Greet your friends Weiss, Blake and Yang. Respond in the following text format:
                    "Ruby": "<what you would like to say next>"
                """,
        },
    ]
    gpt_response(message)


initial_response()
for i in range(3):
    for j in range(len(characters)):
        generate_response(
            you=characters[j % len(characters)],
            person1=characters[(j + 1) % len(characters)],
            person2=characters[(j + 2) % len(characters)],
            person3=characters[(j + 3) % len(characters)],
        )

print("\n".join(history))
