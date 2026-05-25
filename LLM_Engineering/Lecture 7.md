# 📘 Context Windows, Tokens, and API Costs

---

## 1. Overview of the Lecture

This lecture introduces some of the most important practical concepts in working with LLMs:

- **context windows**
    
- **tokens**
    
- **API costs**
    
- how conversation history affects both memory and pricing
    
- why large context windows matter
    
- how different models compare on context length and cost
    

It also reinforces that these ideas are foundational for everything that follows in the course.

---

# 2. What Is a Context Window?

## Definition

The **context window** is the maximum number of **tokens** a model can consider at one time.

It is the maximum amount of text the model can look back on when generating the next token.

## In simple terms

The context window is the model’s working memory for a single interaction.

If the total amount of text exceeds the context window:

- the model cannot process it
    
- the request may fail
    
- or some earlier content may need to be dropped/truncated
    

---

# 3. What Must Fit Inside the Context Window?

A very important point is that the context window does **not** only include the latest message.

It includes the **entire input sequence** being used at that moment.

This usually means:

- earlier parts of the conversation
    
- previous user messages
    
- previous assistant replies
    
- the newest user message
    
- and the generated output as it is being produced
    

## Example

Suppose the conversation is:

- “Hi, my name is Ed.”
    
- “Nice to meet you, Ed.”
    
- “What’s my name?”
    

All of that conversation history must be part of the context if the model is going to answer correctly.

---

# 4. Why the Output Also Matters

The lecture explains something subtle but important:

LLMs generate output **one token at a time**.

For example, if the model is going to generate:

> “Your name is Ed.”

it does not generate the whole sentence at once.

Instead it might generate:

1. “Your”
    
2. then “name”
    
3. then “is”
    
4. then “Ed”
    

After each new token is generated, the whole sequence is effectively reconsidered to predict the next token.

## Practical implication

The context window must be able to hold:

- the original prompt
    
- the full prior conversation
    
- any inserted memory or retrieved information
    
- the model’s partial generated answer so far
    

So the limit affects not only input size, but also how much output can be produced.

---

# 5. Why Context Window Size Matters

The context window determines how much background information the model can keep in play.

This affects how well it can:

- remember previous parts of a conversation
    
- reference earlier details
    
- work with long documents
    
- use few-shot or multi-shot prompting
    
- use RAG systems
    
- carry out more advanced inference-time techniques
    

## Key idea

A larger context window means:

- more information can be included
    
- more examples can be provided
    
- longer histories can be preserved
    
- larger documents can be processed
    

---

# 6. Multi-Shot Prompting

The lecture mentions **multi-shot prompting**.

## Definition

Multi-shot prompting means giving the model several examples in the prompt before asking it to solve the actual task.

For example:

- Example question 1 + answer
    
- Example question 2 + answer
    
- Example question 3 + answer
    
- Then the new question
    

## Why it matters

This can improve performance because the model can imitate the pattern shown in the examples.

## Why context window matters here

All of those examples consume tokens.

So a larger context window allows you to:

- provide more examples
    
- provide richer demonstrations
    
- support more sophisticated prompting
    

---

# 7. RAG and Context Usage

The lecture also refers to **RAG** (Retrieval-Augmented Generation), which will be covered later.

## Main idea

In RAG, external information is retrieved and inserted into the prompt so the model can use it.

## Why context window matters

Retrieved passages take up space in the context window.

So if you want the model to answer using:

- documents
    
- product information
    
- support knowledge
    
- ticket prices
    
- company data
    

that information must fit inside the context window.

---

# 8. Long Documents and Huge Context Windows

The lecturer gives the example of the **complete works of Shakespeare**.

A question about a very large body of text would require an extremely large context window if the full text had to be included.

## Important insight

Some models now have context windows large enough to fit massive documents in a single prompt.

This is one of the most impressive features of modern LLMs.

---

# 9. API Costs vs Chat Product Subscriptions

The lecture distinguishes between two ways of paying for model use.

---

## A. Chat products

Examples:

- ChatGPT
    
- Claude
    
- Gemini
    

These usually offer:

- a free tier
    
- paid subscription tiers
    

Typical subscription examples:

- around **$20/month**
    
- up to **$200/month**
    

### Important feature

With a subscription, you are not usually charged **per individual message**, though you may be subject to:

- rate limits
    
- message limits
    
- usage caps
    

---

## B. APIs

If you use the API:

- you pay **per use**
    
- subscription status usually does **not** matter
    

The API is intended for:

- building your own products
    
- integrating models into software
    
- creating custom workflows or agents
    

---

# 10. Why API Calls Cost Money

API pricing exists because the provider must pay for:

- inference compute
    
- servers and infrastructure
    
- enormous numbers of calculations
    
- and, indirectly, the cost of model development and training
    

The lecture notes that training frontier models can cost:

- **$100 million or more**
    

So API pricing reflects both:

- ongoing runtime cost
    
- recovery of investment
    

---

# 11. How API Pricing Usually Works

Typically, API pricing depends on:

- **input tokens**
    
- **output tokens**
    

## Input tokens

Everything you send to the model

## Output tokens

Everything the model generates in response

---

# 12. Catch #1: Input Tokens Include the Full Sequence

A common misunderstanding is that input cost only reflects the latest message.

That is not true.

The input token count usually includes:

- the whole conversation so far
    
- prior messages
    
- inserted memory
    
- retrieved documents
    
- examples in the prompt
    
- any other extra context
    

## Why this makes sense

The model has to process all of that information to generate the next token.

So even if it feels expensive, the cost reflects real computation.

## Alternative

You _could_ send only the latest message and omit the history.

But then:

- the model will lose context
    
- responses will be weaker
    
- results will often be worse
    

---

# 13. Catch #2: Reasoning Tokens Also Cost Money

For reasoning models, output cost may include not just the final answer, but also the reasoning process.

## Important point

Some models generate internal reasoning tokens.

Even if you do not fully see those tokens, you may still be charged for them because compute was used to produce them.

## Why this matters

This can make costs:

- less predictable
    
- harder to estimate in advance
    

especially for reasoning-heavy tasks

---

# 14. Why Hidden Reasoning Still Costs Money

The lecturer emphasizes that, even if the reasoning is hidden from the user, the computation still happened.

So charging for those tokens is justified because:

- the hardware did the work
    
- the inference cost is real
    

This is especially relevant for:

- reasoning models
    
- hybrid models
    
- agentic workflows that generate lots of internal processing
    

---

# 15. Leaderboards for Comparing Models

The lecture mentions **leaderboards** as a way to compare LLMs.

These become especially important later in the course.

One example mentioned is the **Vellum leaderboard**, which shows useful information such as:

- context window size
    
- API pricing
    
- model comparisons
    

---

# 16. Example Model Comparison Data Mentioned in the Lecture

The lecture gives several approximate examples of model properties.

## GPT-5

- Context window: **400,000 tokens**
    
- Input cost: about **$1.25 per million tokens**
    
- Output cost: about **$10 per million tokens**
    

### Important clarification

At first glance, “$10 output cost” sounds expensive.

But it is:

- **$10 per million output tokens**
    

That is a very large amount of text.

---

## GPT-5 Nano

A much cheaper, smaller version.

- Input cost: about **$0.05 per million tokens**
    
- Output cost: about **$0.40 per million tokens**
    

This shows how dramatically cheaper smaller models can be.

---

## Claude

- Context window around **200,000 tokens**
    

---

## Open-source GPT OSS

- Context window around **130,000 tokens**
    

---

## Gemini 2.5 Flash

- Context window: **1 million tokens**
    

This is presented as especially striking, because it means Gemini can handle extremely large inputs in one prompt.

---

# 17. How to Think About Token Costs Realistically

The lecturer stresses that many people overestimate the cost of small API experiments.

## For individual experimentation

If you are just sending simple prompts like:

- “Hi, my name is Ed.”
    

the token count is tiny.

That means:

- the actual cost per call is very small
    
- often almost negligible for casual experimentation
    

## For large-scale systems

Costs become much more important when:

- you have many users
    
- many concurrent conversations
    
- long histories
    
- agent loops
    
- large outputs
    
- heavy reasoning usage
    

In those cases, you need to understand:

- unit economics
    
- per-user cost
    
- total operational cost
    

---

# 18. Why Agent Loops Can Become Expensive

The lecture briefly warns that **agent loops** can consume lots of tokens.

This is because they may repeatedly:

- call the model
    
- include long context
    
- generate reasoning
    
- use tools and update history
    
- repeat the cycle many times
    

So even if one API call is cheap, a full agentic workflow may become much more expensive.

---

# 19. Caching

The lecture also mentions **caching**.

## What caching means

If you send the same input repeatedly within a short period, some providers may charge less because the work can be partially reused.

## Example

For GPT models, this may happen automatically in some situations.

For Claude, the rules are more nuanced.

## Why caching matters

Caching can reduce input cost when:

- the same long prompt is reused
    
- the same context is sent frequently
    
- repeated requests share a common prefix
    

This can significantly reduce cost in production systems.

---

# 20. Why Big Context Windows Are So Powerful

Large context windows make it possible to:

- keep long chat histories
    
- include large retrieved document sets
    
- run few-shot or multi-shot prompting more effectively
    
- analyze large documents
    
- support more advanced workflows without truncating context
    

## Example from the lecture

A context window of **1 million tokens** means a model like Gemini can nearly fit the complete works of Shakespeare into one prompt.

That is a powerful illustration of how far these systems have come.

---

# 21. Concepts Introduced or Reinforced

## Token

A unit of text processed by the model

## Context window

The maximum number of tokens the model can consider at once

## Input tokens

All tokens sent to the model, including full conversation history and inserted context

## Output tokens

All tokens generated by the model

## Reasoning tokens

Intermediate tokens used during reasoning, which may still incur cost

## Multi-shot prompting

Giving multiple examples in the prompt before asking the real question

## RAG

Retrieval-Augmented Generation; inserting retrieved external information into the prompt

## Caching

Reusing computation or reducing charges for repeated inputs

---

# 22. Comparison Summary

| Concept              | Why It Matters                                           |
| -------------------- | -------------------------------------------------------- |
| Context window       | Limits how much the model can remember/use at once       |
| Large context window | Supports long chats, documents, and advanced prompting   |
| Input tokens         | Drive cost and include more than just the latest message |
| Output tokens        | Also contribute to cost                                  |
| Reasoning tokens     | Hidden reasoning can increase cost                       |
| Caching              | Can reduce repeated input cost                           |

---

# 23. Practical Takeaways

- The context window is one of the most important constraints when working with LLMs.
    
- It includes not just the latest prompt, but the **whole conversation and generated continuation**.
    
- Large context windows enable long-memory workflows, document analysis, RAG, and multi-shot prompting.
    
- API costs are based mainly on **input and output tokens**.
    
- Input cost includes the full history and any inserted context.
    
- Reasoning models may cost more because internal reasoning tokens also count.
    
- Small personal API experiments are usually very cheap.
    
- Costs become more important when building large systems or agents.
    
- Model comparison tools and leaderboards are useful for understanding tradeoffs between price and capability.
    

---

# 24. Course Progress Recap from the Lecture

By this point, students can already:

- write code to call OpenAI and Ollama
    
- summarize text
    
- compare frontier models
    
- understand transformers, tokens, context windows, and API costs
    

By the end of the week, students are expected to be comfortable with:

- the OpenAI API
    
- chat completions
    
- one-shot prompting
    
- streaming
    
- markdown and JSON outputs
    
- building useful business solutions quickly
    

---

# 25. One-Paragraph Revision Summary

The context window is the maximum number of tokens an LLM can consider at once, and it includes not just the latest prompt but the full conversation history, inserted context, and even the generated output as it unfolds. This makes context windows crucial for long conversations, multi-shot prompting, RAG, and working with long documents. API costs are usually based on both input and output tokens, and hidden reasoning tokens may also increase cost for reasoning models. Large context windows, such as Gemini’s million-token window, make remarkable tasks possible, while smaller models and variants can make experimentation extremely cheap. Understanding context size and token pricing is essential for both effective prompting and building scalable AI systems.

---
