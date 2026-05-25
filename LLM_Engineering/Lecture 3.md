
# 📘 Types of LLMs — Base, Chat, Reasoning, and Hybrid Models

---

## 1. Overview of the Lesson

### What has been covered so far

By this point in the course, students have:

- Seen a range of **frontier models**
    
- Worked with models such as:
    
    - OpenAI models
        
    - Gemini
        
    - LLaMA
        
    - distilled/open-source variants like DeepSeek Distilled and LLaMA 3.2
        
- Built a tool that can **summarize web pages** using:
    
    - the OpenAI API
        
    - possibly Ollama/local models as well
        

### What this lesson aims to teach

By the end of this lesson, students should be able to:

- Compare different frontier models
    
- Understand the difference between:
    
    - **chat models**
        
    - **reasoning models**
        
    - **base models**
        
    - **hybrid models**
        
- Recognize what each type does well
    
- Understand where each type struggles
    

---

# 2. The Three Main Breeds of LLMs

The lecture explains that there are **three major types of LLMs**, based on what they were trained to do:

1. **Base models**
    
2. **Chat / instruct models**
    
3. **Reasoning models**
    

A newer category, **hybrid models**, is presented as an extension of reasoning models.

---

# 3. Base Models

## Definition

A **base model** is the most fundamental form of a language model.

Its job is simple:

- Take a sequence of text as input
    
- Predict what comes next
    

That is all it does.

## Key idea

A base model is **not specifically trained to chat** or follow instructions. It is just a next-token predictor.

## Everyday analogy: predictive text

A familiar example of a base-model-like behavior is **predictive text on a phone**.

Example:

- You type: “Hello there”
    
- The phone predicts the next likely word
    
- Each selected word becomes part of the sequence
    
- The system predicts again
    

This is essentially sequence completion, which is what a base model does.

## Historical example

Before ChatGPT, earlier GPT systems were closer to **base models**.

For example:

- GPT-3 was commonly used as a completion model
    
- Users had to structure prompts carefully to get good results
    

A common pattern looked like this:

- Q: question
    
- A: answer
    
- Q: another question
    
- A: another answer
    
- Q: your new question
    
- A:
    

This prompt style nudged the model into continuing in a **question-answer format**.

---

# 4. Chat Models / Instruct Models

## Why chat models were created

Researchers realized that models could be trained further on structured examples of:

- one message
    
- one response
    
- repeated over many examples
    

This additional training produced a model that was better at conversation and instruction following.

## Other names

A chat model is also often called an:

- **instruct model**
    
- **chat variant**
    

## Core structure of chat models

Chat models are trained around a structured conversation format:

### System prompt

- High-level instruction that defines overall behavior
    

### User prompt

- Message from the user
    

### Assistant reply

- Model’s response
    

This creates a repeated pattern:

- system prompt
    
- user message
    
- assistant reply
    
- user message
    
- assistant reply
    

## Why ChatGPT was such a breakthrough

The jump from GPT to ChatGPT came from converting a base-style model into a **chat-oriented model**.

A key technique involved was:

- **RLHF** = Reinforcement Learning from Human Feedback
    

This helped train the model to give more useful, aligned, and conversational answers.

---

# 5. Reasoning Models

## Origin

After chat models became popular, users discovered prompt tricks that improved performance.

One famous trick was:

> “Please think step by step.”

This often made the model perform better, especially on complex tasks.

## Why this worked

When prompted to think step by step, the model would generate a more structured reasoning process, which often improved the final answer.

This inspired researchers to train models directly on examples where they:

- reason through a problem
    
- then produce the answer
    

## Definition

A **reasoning model** is a model trained to:

1. produce intermediate reasoning steps
    
2. then give the final answer
    

These are also sometimes called:

- **thinking models**
    

## Core behavior

Instead of going straight to the answer, the model first works through the problem internally or explicitly.

This makes them stronger on:

- puzzles
    
- multi-step reasoning
    
- logic tasks
    
- hard problem solving
    

---

# 6. Hybrid Models

## Definition

A **hybrid model** is a model that can decide **how much reasoning to do**.

It is basically an advanced form of a reasoning model.

## Key idea

A hybrid model does not always reason heavily.

For example:

- For a simple greeting like “Hi”
    
    - it may respond quickly with little or no reasoning
        
- For a difficult puzzle
    
    - it may spend more time reasoning
        

## Examples mentioned

The lecture identifies modern models like:

- **Gemini 2.5 Pro**
    
- **Claude 4**
    
- **GPT-5**
    

as examples of **hybrid models**

## Why hybrid models matter

They combine:

- the speed of chat models for simple tasks
    
- the deeper problem-solving ability of reasoning models for hard tasks
    

---

# 7. Reasoning Budget / Reasoning Effort

## Definition

The amount of reasoning a model does is called its:

- **reasoning budget**
    
- or **reasoning effort**
    

A larger reasoning budget means the model spends more tokens/thought on solving the problem.

## General trend

Higher reasoning budget usually leads to:

- better benchmark performance
    
- stronger problem solving
    
- better performance across many tasks
    

But it also increases:

- latency
    
- cost
    
- token usage
    

---

# 8. Budget Forcing

## Definition

**Budget forcing** means deliberately making a reasoning model think longer.

The goal is to increase the depth of reasoning before the model finalizes its answer.

## Surprising technique from the S1 paper

A notable result discussed in the lecture comes from the **S1 paper** (January 2025).

The surprising finding:

- You can sometimes improve reasoning simply by inserting the word:
    

> “Wait”

into the reasoning trace.

## Why this helps

After “Wait,” the model tends to continue with thoughts like:

- “Wait, I should rethink this”
    
- “Am I sure?”
    
- “Let me reconsider”
    

This can cause the model to:

- reflect more deeply
    
- challenge its earlier reasoning
    
- explore alternatives
    
- catch mistakes
    

## Why this is surprising

It sounds like it should require a complicated mathematical method, but in this case the mechanism is surprisingly simple and somewhat hacky.

That is one reason the technique is memorable.

---

# 9. Why Not Always Use Reasoning Models?

At first glance, reasoning models seem better. So why not use them for everything?

The lecture explains that reasoning models are **not always the best choice**.

---

# 10. Strengths of Reasoning Models

Reasoning models are usually better for:

## Problem solving

They are especially strong at:

- puzzles
    
- math-like reasoning
    
- logic-heavy questions
    
- multi-step analysis
    

## Intelligence-style benchmarks

They tend to score better on many benchmark tasks related to reasoning and problem solving.

## Hard tasks

If the task is difficult and accuracy matters more than speed, reasoning models are often preferred.

---

# 11. Strengths of Chat Models

Chat models still have important advantages.

## Faster

They do not spend time producing long reasoning traces.

## Cheaper

They use fewer intermediate reasoning tokens, so they are often more cost-effective.

## Better for interactive use

For live conversational settings, speed matters. Chat models feel more responsive.

## Often better for fluid content generation

The lecture suggests that chat models may sometimes be better for:

- emails
    
- natural writing
    
- conversational drafting
    
- more fluid content generation
    

Reasoning models can sometimes:

- overthink
    
- sound cold
    
- sound overly analytical
    

Important note:

- This point is presented as **anecdotal rather than rigorously proven**
    
- There are not strong formal metrics for this yet
    

---

# 12. Strengths of Base Models

Base models are less commonly used directly by end users, but they are useful in a special case:

## Best starting point for custom training

If you want to train a model to acquire a **new skill** or operate in a **different format**, it is often better to start with a base model.

Why?

- A chat model is already shaped around a chat structure
    
- A reasoning model is already shaped around reasoning traces
    
- A base model is more neutral and flexible
    

So base models are useful when you want to:

- fine-tune a model
    
- train a custom behavior
    
- build a different interaction structure
    

---

# 13. Comparison of Model Types

| Model Type                | Main Purpose                            | Strengths                                   | Weaknesses                                          |
| ------------------------- | --------------------------------------- | ------------------------------------------- | --------------------------------------------------- |
| **Base Model**            | Predict next token in a sequence        | Flexible starting point for custom training | Not naturally conversational                        |
| **Chat / Instruct Model** | Follow prompts and chat naturally       | Fast, cheap, good for interactive use       | Weaker than reasoning models on hard problems       |
| **Reasoning Model**       | Think through problems before answering | Strong on complex tasks and problem solving | Slower, more expensive                              |
| **Hybrid Model**          | Adjust reasoning depth based on task    | Good balance of speed and intelligence      | More complex, not always necessary for simple tasks |

---

# 14. Important Terms to Know

## Base model

A model trained only to predict the next token.

## Chat model

A model trained for structured conversation and instruction following.

## Instruct model

Another name for a chat-oriented model.

## System prompt

High-level instruction that guides the entire conversation.

## User prompt

The message sent by the user.

## Assistant reply

The model’s response.

## RLHF

Reinforcement Learning from Human Feedback; used to improve instruction following and helpfulness.

## Chain-of-thought prompting

Prompting technique where the model is asked to reason step by step.

## Reasoning model

A model trained to produce reasoning before its answer.

## Hybrid model

A model that dynamically decides how much reasoning to do.

## Reasoning budget / effort

The amount of reasoning computation or token budget used before answering.

## Budget forcing

Techniques that encourage a reasoning model to think longer.

---

# 15. Main Takeaways

- LLMs can be grouped into **base**, **chat**, and **reasoning** models.
    
- **Hybrid models** are newer models that adapt how much reasoning they use.
    
- A **base model** just predicts the next token.
    
- A **chat model** is trained to interact in structured dialogue.
    
- A **reasoning model** is trained to think through problems before answering.
    
- **Reasoning models** are usually better at complex tasks.
    
- **Chat models** are usually faster, cheaper, and better for fluid interaction.
    
- **Base models** are especially useful as a starting point for custom training.
    
- The amount of reasoning can be controlled through **reasoning budget** and **budget forcing**.
    

---

# 16. Exam-Style Summary

If asked to explain the lesson briefly:

> Base models predict the next token in a sequence. Chat models are further trained to follow prompts and hold structured conversations using system, user, and assistant messages. Reasoning models go a step further by generating intermediate reasoning before answering, which improves problem solving. Hybrid models combine both approaches and decide how much reasoning is needed based on the task. Reasoning models are usually stronger on hard problems, while chat models are faster and cheaper for interactive tasks.

---