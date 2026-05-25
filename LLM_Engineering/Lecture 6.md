
# 📘 From LSTMs to Transformers, Emergent Intelligence, Context Engineering, and Agentic AI

---

## 1. Overview of the Lecture

This lecture explains:

- What came **before transformers**
    
- Why transformers overtook earlier sequence models
    
- Why LLMs surprised the world
    
- The idea of **emergent intelligence**
    
- The rise and decline of **prompt engineering** as a distinct role
    
- The shift toward **context engineering**
    
- What **agentic AI** means
    
- How autonomy, loops, and tools fit into modern AI systems
    

---

# 2. Before Transformers: LSTMs

Before transformers became dominant, one of the main model types used for sequence tasks was the:

- **LSTM** = Long Short-Term Memory
    

LSTMs are a type of:

- **RNN** = Recurrent Neural Network
    

## What LSTMs were good at

LSTMs were designed to handle sequences and keep track of relationships across time or across tokens in a sequence.

Many people believed, and some still believe, that LSTMs had certain strengths over transformers in terms of:

- modeling sequence relationships
    
- preserving longer-term dependencies
    
- building a deeper sense of ordered progression through input
    

---

# 3. Why LSTMs Lost to Transformers

The key weakness of LSTMs was not necessarily capability, but **efficiency**.

## Main limitation: poor parallelization

LSTMs process sequences step by step.

That means:

- one part of the computation depends on the previous part
    
- you cannot easily compute everything at once
    
- training becomes slower and harder to scale
    

## Why this mattered

Because LSTMs were sequential, they were:

- time-consuming to train
    
- computationally harder to scale
    
- less suited to very large datasets and huge models
    

---

# 4. Why the Transformer Won

The transformer can be thought of, in some ways, as a **simplification**.

It removed much of the sequential recurrence used by LSTMs and instead relied on a simpler mechanism:

- **attention**
    

## Why that mattered

Even if the transformer was in some ways less elaborate, it had a huge engineering advantage:

- much better **parallelization**
    
- faster training
    
- better scalability
    
- more practical use of large compute systems
    

## Key takeaway

The performance gains from being able to train much larger models more efficiently outweighed the theoretical advantages LSTMs may have had.

---

# 5. Meaning of the Title: “Attention Is All You Need”

The title of the 2017 paper was making a bold claim:

> We do not need all the complicated machinery used before.  
> Attention alone is enough.

## What the authors were saying

The core message was:

- the simpler attention-based setup is sufficient
    
- even if it seems less sophisticated
    
- the ability to scale it efficiently makes it far more powerful in practice
    

This turned out to be one of the most important ideas in modern AI.

---

# 6. The World’s Reaction to Transformers and LLMs

When transformer-based LLMs became mainstream, especially around **2023**, the public reaction was dramatic.

## Initial reaction

People were:

- astonished
    
- amazed
    
- shocked
    

The word **transformer** suddenly became widely known, even outside technical circles.

For practitioners, this was unusual: terms that once lived inside research and engineering communities suddenly entered everyday conversation.

---

# 7. The Backlash: “Stochastic Parrots”

After the excitement came criticism and concern.

One famous critique came in the form of the paper:

- **“On the Dangers of Stochastic Parrots”**
    

## Core concern

Critics argued that LLMs were:

- statistical systems
    
- predicting plausible next words
    
- not actually understanding truth in a robust way
    

The fear was that people might mistake:

- fluent text  
    for
    
- genuine understanding or reliable knowledge
    

## Why the critique mattered

It highlighted concerns about:

- misinformation
    
- overtrust in AI
    
- bias
    
- the danger of confusing plausibility with truth
    

The lecturer suggests this paper is worth reading because it captures how many people were thinking at the time.

---

# 8. Why the Critique Did Not Fully Capture What Happened

The lecture argues that, although those concerns were serious, the critique did not age perfectly.

## Why?

Because LLMs turned out to do something more surprising than just produce plausible text.

The real surprise was not:

- that they could predict realistic-sounding words
    

The real surprise was:

- that those predicted words were so often **correct**
    

---

# 9. The Central Surprise of LLMs

No one is shocked that a sufficiently large neural network can produce:

- fluent text
    
- plausible continuations
    
- realistic language
    

What _is_ shocking is that these predictions often turn out to be:

- accurate
    
- useful
    
- logically sound
    
- even mathematically correct in many cases
    

## Example

If you give a model a math problem, you might expect it to generate:

- words that sound like the kind of answer someone would give
    

What is surprising is that it often gives:

- the actual correct answer
    

That is what caught researchers and practitioners off guard.

---

# 10. Emergent Intelligence

This surprising behavior is often described as:

- **emergent intelligence**
    

## Definition

Emergent intelligence is the idea that when a neural network becomes large enough and is trained at sufficient scale, new capabilities appear that were not obvious from the simple next-token objective alone.

## Key idea

At sufficient scale, the model does not just produce plausible tokens. It produces outputs that:

- resemble reasoning
    
- resemble intelligence
    
- imitate intelligent behavior with remarkable accuracy
    

## Important nuance

We understand:

- the training process
    
- the statistics
    
- the optimization
    

But we do not fully understand why these systems become as capable as they do.

So the mystery is not “how the code runs,” but:

- why scaling produces such strong intelligence-like behavior
    

---

# 11. Prompt Engineering: Rise and Decline

There was a period when:

- **prompt engineer** was an actual job title
    

It even commanded:

- high six-figure salaries in some cases
    

## Why

People discovered that carefully phrasing prompts could dramatically improve model performance.

This included tricks like:

- providing context
    
- specifying style
    
- giving examples
    
- clarifying intent
    

## What changed

Over time, prompt engineering became a normal skill that many users learned.

So the lecturer’s point is:

- prompt engineering did not disappear
    
- but it stopped being a rare, specialized title
    

Now, in a sense:

- everyone working with LLMs is a prompt engineer
    

---

# 12. Copilots

Another major development was the rise of:

- **copilots**
    

Examples include:

- Microsoft Copilot
    
- GitHub Copilot
    

## What copilots represent

They showed that humans and LLMs could work **collaboratively**.

Instead of replacing the user, the AI acts as:

- an assistant
    
- a collaborator
    
- a helper embedded in the workflow
    

## Why this mattered

Copilots help:

- automate repetitive tasks
    
- accelerate drafting and coding
    
- enrich the human’s work in real time
    

This was a major shift in how AI tools were positioned and used.

---

# 13. From Prompt Engineering to Context Engineering

A newer and broader concept is:

- **context engineering**
    

The lecture describes this as the new evolution of prompt engineering.

## Definition

Context engineering means thinking broadly about **all the information and structure** you provide to the model so it can succeed.

It is not just about wording a clever prompt.

It includes:

- background information
    
- task instructions
    
- business-specific data
    
- retrieved knowledge
    
- examples
    
- external tools
    
- structured context
    

## Core principle

The more relevant information you include in the model’s input sequence, the more likely it is to produce a useful output sequence.

---

# 14. Simple Intuition for Context Engineering

Suppose you want an LLM to answer customer questions about travel ticket prices.

If the ticket prices are **not** in the input context, the model may:

- guess
    
- hallucinate
    
- use outdated knowledge
    

If the ticket prices **are** included in the context, the model is much more likely to:

- answer correctly
    
- stay aligned with the task
    

## Key takeaway

Context engineering is fundamentally about:

- giving the model the right information at the right time
    

So that when it predicts the next tokens, those predictions are grounded in what you want it to know.

---

# 15. Tools as Part of Context Engineering

The lecturer also notes that context engineering includes giving the model access to:

- **tools**
    

These tools might let the model:

- search for information
    
- call APIs
    
- retrieve documents
    
- perform calculations
    
- take actions
    

This sounds advanced, but the underlying idea is still simple:

- improve the model’s input and environment
    
- so it can produce better outputs
    

---

# 16. Agentic AI

One of the hottest concepts in AI right now is:

- **agentic AI**
    

The lecturer notes that there are several definitions, but gives two common ones.

---

# 17. Definition 1: The LLM Controls the Workflow

Under one definition, an agentic AI system is:

> a system in which the LLM decides what happens next

This may include:

- deciding which step to take
    
- deciding whether to call another model
    
- deciding whether to use a tool
    
- deciding what information to gather next
    

In this sense, the LLM acts like a controller or orchestrator.

---

# 18. Definition 2: An LLM in a Loop with Tools

A second common definition is:

> an LLM operating in a loop, with access to tools

## What that means

The system repeatedly calls the model.

Each round, the model may:

- inspect the situation
    
- plan the next step
    
- use a tool
    
- update its state
    
- continue
    

This creates a loop:

1. model thinks
    
2. model acts
    
3. system updates
    
4. model thinks again
    

This repeated cycle is central to many modern AI agents.

---

# 19. Agent Loops

This repeated process is often called an:

- **agent loop**
    

## Why it matters

An agent loop allows the system to perform multi-step work rather than just answering one prompt once.

This is useful for tasks like:

- coding
    
- research
    
- planning
    
- booking
    
- data gathering
    
- tool use across multiple steps
    

---

# 20. Real Example: Claude Code

The lecture uses **Claude Code** as an example of agentic AI.

When using it, you can often see:

- a to-do list
    
- a plan
    
- steps being executed one by one
    
- actions being taken iteratively
    

Under the hood, this is essentially:

- a sequence of LLM calls
    
- operating in a loop
    
- with tools/actions available
    

So the visible “agent” behavior is produced by repeated model invocations plus tool usage.

---

# 21. Autonomy

A word often associated with agentic AI is:

- **autonomy**
    

## What it means here

Autonomy does **not** mean consciousness or free will.

It means the system is allowed to:

- choose the next action
    
- decide which tool to use
    
- determine what to do next within its defined framework
    

## Important clarification

Even here, what is happening is still:

- input sequence in
    
- output sequence out
    

But the output sequence may include:

- instructions
    
- plans
    
- tool calls
    
- next-step decisions
    

That makes the system appear autonomous.

---

# 22. Why Agentic AI Can Feel Mysterious

Agentic AI can sound almost magical or vague, but the lecturer emphasizes that it is still built on the same fundamental mechanism:

- the model receives context
    
- it predicts tokens
    
- those tokens may encode actions or instructions
    
- the system executes them
    
- the loop continues
    

So although the behavior looks sophisticated, the underlying process is still based on sequence prediction.

---

# 23. Key Examples Mentioned

The lecturer references examples students may already have seen:

- **Claude Code**
    
- a **GPT-based agent** that found a reservation for Banoffee Pie in the evening
    

These illustrate how agentic AI feels in practice:

- not just answering
    
- but actively progressing through a task
    

---

# 24. Important Concepts to Know

## LSTM

Long Short-Term Memory; a type of recurrent neural network used before transformers for sequence modeling.

## RNN

Recurrent Neural Network; a class of models that process sequences step by step.

## Parallelization

Running many computations at once. Transformers are much better at this than LSTMs.

## Attention

A mechanism that lets the model focus on the relevant parts of an input sequence.

## Emergent intelligence

The surprising appearance of intelligence-like behavior when models are scaled up.

## Prompt engineering

The practice of carefully crafting prompts to get better outputs.

## Context engineering

The broader practice of designing all the information, structure, and tools surrounding the model input.

## Copilot

An AI assistant that collaborates with a human user inside a workflow.

## Agentic AI

An AI system where the LLM controls actions or operates in a loop, often with tools.

## Agent loop

A repeated cycle of model calls, actions, and updated context.

## Autonomy

The ability of an agentic system to choose its next action within a task framework.

---

# 25. Comparison: LSTMs vs Transformers

| Feature                   | LSTM / RNN           | Transformer             |
| ------------------------- | -------------------- | ----------------------- |
| Sequence handling         | Strong, step-by-step | Strong, attention-based |
| Parallelization           | Poor                 | Excellent               |
| Scalability               | Limited              | High                    |
| Training efficiency       | Slower               | Faster                  |
| Practical dominance today | Rare for LLMs        | Standard for LLMs       |

---

# 26. Comparison: Prompt Engineering vs Context Engineering

| Concept             | Focus                                                       |
| ------------------- | ----------------------------------------------------------- |
| Prompt engineering  | Crafting the wording of the prompt                          |
| Context engineering | Designing the full information environment around the model |

---

# 27. Main Takeaways

- Before transformers, **LSTMs** were a major sequence model, but they were hard to parallelize.
    
- Transformers succeeded largely because they scaled much more efficiently.
    
- The title _Attention Is All You Need_ reflects the idea that a simpler attention-based method turned out to be enough.
    
- The surprising part of LLMs is not just that they generate plausible text, but that they often generate **correct and intelligent-seeming** outputs.
    
- This phenomenon is often called **emergent intelligence**.
    
- **Prompt engineering** was once a distinct role, but is now a common skill.
    
- **Context engineering** is a broader and more important idea: giving the model the right information and tools.
    
- **Copilots** showed how humans and AI can work together.
    
- **Agentic AI** refers to systems where LLMs control steps, use tools, and operate in loops.
    
- **Autonomy** in AI usually means choosing the next action, not anything mystical.
    

---

# 28. One-Paragraph Revision Summary

Before transformers, LSTMs were widely used for sequence modeling, but they were difficult to scale because they processed inputs step by step and could not be parallelized efficiently. Transformers simplified the architecture by relying on attention, which turned out to be sufficient and far easier to scale, leading to their dominance. What surprised researchers most was not that these systems could generate plausible text, but that, at scale, they often produced genuinely accurate and intelligence-like responses — a phenomenon often described as emergent intelligence. As LLM usage evolved, prompt engineering expanded into context engineering, which focuses on giving models the right information, structure, and tools. This has led directly to copilots and agentic AI systems, where models operate in loops, use tools, and make decisions about what to do next.

---
