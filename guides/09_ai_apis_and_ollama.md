# LLM APIs and Ollama - beyond OpenAI

_IMPORTANT: If you're not as familiar with APIs in general, and with Environment Variables on your PC or Mac, please review the APIs section in Guide 4 Technical Foundations before proceeding with this guide (topics 3 and 5 in Guide 4)._

## Crucial context for using models other than OpenAI - please read this first!

Throughout the course, we use APIs for connecting with the strongest LLMs on the planet.

The companies behind these LLMs, such as OpenAI, Anthropic, Google and DeepSeek, have built web endpoints. You call their models by making an HTTP request to a Web Address and passing in all the information about your prompts.

But it would be painful if we needed to build HTTP requests every time we wanted to call an API.

To make this simple, the team at OpenAI wrote a python utility known as a "Python Client Library" which wraps the HTTP call. So you write python code and it calls the web.

And THAT is what the library `openai` is.

### What is the `openai` python client library

It is:
- A lightweight python utility
- Turns your python requests into an HTTP call
- Converts the results coming back from the HTTP call into python objects

### What it is NOT

- It's not got any code to actually run a Large Language Model! No GPT code! It just makes a web request
- There's no scientific computing code, and nothing particularly specialized for OpenAI

### How to use it:

```python
# Create an OpenAI python client for making web calls to OpenAI
openai = OpenAI()

# Make the call
response = openai.chat.completions.create(model="gpt-4.1-mini", messages=[{"role":"user", "content": "what is 2+2?"}])

# Print the result
print(response.choices[0].message.content)
```

### What does this do

When you make the python call: `openai.chat.completions.create()`  
It simply makes a web request to this url: `https://api.openai.com/v1/chat/completions`  
And it converts the response to python objects.

That's it.

Here's the API documentation if you make [direct web HTTP calls](https://platform.openai.com/docs/guides/text?api-mode=chat&lang=curl)  
And here's the same API documentation if you use the [Python Client Library](https://platform.openai.com/docs/guides/text?api-mode=chat&lang=python)

## With that context - how do I use other LLMs?

It turns out - it's super easy!

All the other major LLMs have API endpoints that are compatible with OpenAI.

And so OpenAI did everyone a favor: they said, hey look - you can all use our utility for converting python to web requests. We'll allow you to change the utility from calling `https://api.openai/com/v1` to calling any web address that you specify.

And so you can use the OpenAI utility even for calling models that are NOT OpenAI, like this:

`not_actually_openai = OpenAI(base_url="https://somewhere.completely.different/", api_key="another_providers_key")`

It's important to appreciate that this OpenAI code is just a utility for making HTTP calls to endpoints. So even though we're using code from the OpenAI team, we can use it to call models other than OpenAI.

Here are all the OpenAI-compatible endpoints from the major providers. It even includes using Ollama, locally. Ollama provides an endpoint on your local machine, and they made it OpenAI compatible too - very convenient.

```python
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROK_BASE_URL = "https://api.x.ai/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
```

## Here are examples for Gemini, DeepSeek, Ollama and OpenRouter

### Example 1: Using Gemini instead of OpenAI

1. Visit Google Studio to set up an account: https://aistudio.google.com/  
2. Add your key as GOOGLE_API_KEY to your `.env`  
3. Also add it a second time as GEMINI_API_KEY to your `.env` - this will be helpful later.

Then:

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
response = gemini.chat.completions.create(model="gemini-2.5-flash-preview-05-20", messages=[{"role":"user", "content": "what is 2+2?"}])
print(response.choices[0].message.content)
```

### Example 2: Using DeepSeek API instead of OpenAI (cheap, and only $2 upfront)

1. Visit DeepSeek API to set up an account: https://platform.deepseek.com/  
2. You will need to add an initial $2 minimum balance.  
3. Add your key as DEEPSEEK_API_KEY to your `.env`  

Then:

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
response = deepseek.chat.completions.create(model="deepseek-chat", messages=[{"role":"user", "content": "what is 2+2?"}])
print(response.choices[0].message.content)
```

### Example 3: Using Ollama to be free and local instead of OpenAI

Ollama allows you to run models locally; it provides an OpenAI compatible API on your machine.  
There's no API key for Ollama; there's no third party with your credit card, so no need for any kind of key.

1. If you're new to Ollama, install it by following the instructions here: https://ollama.com   
2. Then in a Cursor Terminal, do `ollama run llama3.2` to chat with Llama 3.2  
BEWARE: do not use llama3.3 or llama4 - these are massive models not designed for home computing! They will fill up your disk.  

Then:

```python
!ollama pull llama3.2

from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="anything")
response = ollama.chat.completions.create(model="llama3.2", messages=[{"role":"user", "content": "what is 2+2?"}])
print(response.choices[0].message.content)
```

### Example 4: Using the popular service [OpenRouter](https://openrouter.ai) which has an easier billing process instead of OpenAI

OpenRouter is very convenient: it gives you free access to many models, and easy access with small upfront to paid models.

1. Sign up at https://openrouter.ai
2. Add the minimum upfront balance as needed
3. Add your key as OPENROUTER_API_KEY to your `.env` file

Then:

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)
response = openrouter.chat.completions.create(model="openai/gpt-4.1-nano", messages=[{"role":"user", "content": "what is 2+2?"}])
print(response.choices[0].message.content)
```


### Using different API providers with Agent Frameworks

The Agent Frameworks make it easy to switch between these providers. You can switch LLMs and pick different ones at any point in the course. There are more notes below on each of them. For OpenAI Agents SDK, see a section later in this notebook. For CrewAI, we cover it on the course, but it's easy: just use the full path to the model that LiteLLM expects.

## Costs of APIs

The cost of each API call is very low indeed - most calls to models we use on this course are fractions of cents.

But it's extremely important to note:

1. A complex Agentic project could involve many LLM calls - perhaps 20-30 - and so it can add up. It's important to set limits and monitor usage.

2. With Agentic AI, there is a risk of Agents getting into a loop or carrying out more processing than intended. You should monitor your API usage, and never put more budget than you are comfortable with. Some APIs have an "auto-refill" setting that can charge automatically to your card - I strongly recommend you keep this off.

3. You should only spend what you are comfortable with. There is a free alternative in Ollama that you can use as a replacement if you wish. DeepSeek, Gemini 2.5 Flash and gpt-4.1-nano are significantly cheaper.

Keep in mind that these LLM calls typically involve trillions of floating point calculations - someone has to pay the electricity bills!

### Ollama: Free alternative to Paid APIs (but please see Warning about llama version)

Ollama is a product that runs locally on your machine. It can run open-source models, and it provides an API endpoint on your computer that is compatible with OpenAI.

First, download Ollama by visiting:
https://ollama.com

Then from your Terminal in Cursor (View menu >> Terminal), run this command to download a model:

```shell
ollama pull llama3.2
```

WARNING: Be careful not to use llama3.3 or llama4 - these are much larger models that are not suitable for home computers.

And now, any time that we have code like:  
`openai = OpenAI()`  
You can use this as a direct replacement:  
`openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')`  
And also replace model names like **gpt-4o-mini** with **llama3.2**.  

You don't need to put anything in your .env file for this; with Ollama, everything is running on your computer. You're not calling out to a third party on the cloud, nobody has your credit card details, so there's no need for a secret key! The code `api_key='ollama'` above is only required because the OpenAI client library expects an api_key to be passed in, but the value is ignored by Ollama.

Below is a full example:

```python
# You need to do this one time on your computer
!ollama pull llama3.2

from openai import OpenAI
MODEL = "llama3.2"
openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = openai.chat.completions.create(
 model=MODEL,
 messages=[{"role": "user", "content": "What is 2 + 2?"}]
)

print(response.choices[0].message.content)
```

You will need to make similar changes to use Ollama within any of the Agent Frameworks - you should be able to google for an exact example, or ask me.

### OpenRouter: Convenient gateway platform for OpenAI and others

OpenRouter is a third party service that allows you to connect to a wide range of LLMs, including OpenAI.

It's known for having a simpler billing process that may be easier for some countries outside the US.

First, check out their website:  
https://openrouter.ai/

Then, take a peak at their quickstart:  
https://openrouter.ai/docs/quickstart

And add your key to your .env file:  
```shell
OPENROUTER_API_KEY=sk-or....
```

And now, any time you have code like this:  
```python
MODEL = "gpt-4o-mini"
openai = OpenAI()
```

You can replace it with code like this:

```python
MODEL = "openai/gpt-4o-mini"
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openai = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)

response = openai.chat.completions.create(
 model=MODEL,
 messages=[{"role": "user", "content": "What is 2 + 2?"}]
)

print(response.choices[0].message.content)
```

You will need to make similar changes to use OpenRouter within any of the Agent Frameworks - you should be able to google for an exact example, or ask me.

## OpenAI Agents SDK - specific instructions

With OpenAI Agents SDK (weeks 2 and 6), it's particularly easy to use any model provided by OpenAI themselves. Simply pass in the model name:

`agent = Agent(name="Jokester", instructions="You are a joke teller", model="gpt-4o-mini")`

You can also substitute in any other provider with an OpenAI compatible API. You do it in 3 steps like this:

```python
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
deepseek_model = OpenAIChatCompletionsModel(model="deepseek-chat", openai_client=deepseek_client)
```

And then you simply provide this model when you create an Agent.

`agent = Agent(name="Jokester", instructions="You are a joke teller", model=deepseek_model)`

And you can use a similar approach for any other OpenAI compatible API, with the same 3 steps:

```python
# extra imports
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

# Step 1: specify the base URL endpoints where the provider offers an OpenAI compatible API
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROK_BASE_URL = "https://api.x.ai/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Step 2: Create an AsyncOpenAI object for that endpoint
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
grok_client = AsyncOpenAI(base_url=GROK_BASE_URL, api_key=grok_api_key)
groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key)
openrouter_client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)
ollama_client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# Step 3: Create a model object to provide when creating an Agent
gemini_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=gemini_client)
grok_3_model = OpenAIChatCompletionsModel(model="grok-3-mini-beta", openai_client=openrouter_client)
llama3_3_model = OpenAIChatCompletionsModel(model="llama-3.3-70b-versatile", openai_client=groq_client)
grok_3_via_openrouter_model = OpenAIChatCompletionsModel(model="x-ai/grok-3-mini-beta", openai_client=openrouter_client)
llama_3_2_local_model = OpenAIChatCompletionsModel(model="llama3.2", openai_client=ollama_client)
```

### To use Azure with OpenAI Agents SDK

See instructions here:  
https://techcommunity.microsoft.com/blog/azure-ai-services-blog/use-azure-openai-and-apim-with-the-openai-agents-sdk/4392537

Such as this:
```python
from openai import AsyncAzureOpenAI
from agents import set_default_openai_client
from dotenv import load_dotenv
import os
 
# Load environment variables
load_dotenv()
 
# Create OpenAI client using Azure OpenAI
openai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)
 
# Set the default OpenAI client for the Agents SDK
set_default_openai_client(openai_client)
```

## CrewAI setup

Here's Crew's docs for LLM connections with the model names to use for all models. As student Sadan S. pointed out (thank you!), it's worth knowing that for Google you need to use the environment variable `GEMINI_API_KEY` instead of `GOOGLE_API_KEY`:

https://docs.crewai.com/concepts/llms

And here's their tutorial with some more info:

https://docs.crewai.com/how-to/llm-connections

## LangGraph setup

To use LangGraph with Ollama (and follow similar for other models):  
https://python.langchain.com/docs/integrations/chat/ollama/#installation

First add the package:  
`uv add langchain-ollama`

Then in the lab, make this replacement:   
```python
from langchain_ollama import ChatOllama
# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOllama(model="gemma3:4b")
```

And obviously run `!ollama pull gemma3:4b` (or whichever model) beforehand.

Many thanks to Miroslav P. for adding this, and to Arvin F. for the question!

## LangGraph with other models

Just follow the same recipe as above, but use any of the models from here:  
https://python.langchain.com/docs/integrations/chat/



## AutoGen with other models

Here's another contribution from Miroslav P. (thank you!) for using Ollama + local models with AutoGen, and Miroslav has a great example showing gemma3 performing well.

```python
# model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
 
from autogen_ext.models.ollama import OllamaChatCompletionClient
 
model_client = OllamaChatCompletionClient(
    model="gemma3:4b",
    model_info={
        "vision": True,
        "function_calling": False,
        "json_output": True,
        "family": "unknown",
    },
)
```

## Worth keeping in mind

1. If you wish to use Ollama to run models locally, you may find that smaller models struggle with the more advanced projects. You'll need to experiment with different model sizes and capabilities, and plenty of patience may be needed to find something that works well. I expect several of our projects are too challenging for llama3.2. As an alternative, consider the free models on openrouter.ai, or the very cheap models that are almost free - like DeepSeek.

2. Chat models often do better than Reasoning models because Reasoning models can "over-think" some assignments. It's important to experiment. Bigger isn't always better...

3. It's confusing, but there are 2 different providers that sound similar!  
- Grok is the LLM from Elon Musk's X
- Groq is a platform for fast inference of open source models

A student pointed out to me that "Groq" came first!

