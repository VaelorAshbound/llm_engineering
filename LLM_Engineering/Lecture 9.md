# 📘 Study Notes: Common Uses of Tools & Their Role in Agentic AI

---

# 1. Overview of the Lecture

This lecture continues the discussion of **tools** in LLM systems and focuses on:

- common real-world uses of tools
    
- why tools are so powerful
    
- how tools enable **agentic AI**
    
- orchestration of multiple LLMs
    
- planning loops and autonomous workflows
    

The key theme is:

> Tools transform an LLM from “just a chatbot” into an active system that can retrieve information, perform actions, execute code, and coordinate workflows.

---

# 2. Typical Uses of Tools

The lecture explains several common categories of tool usage.

---

# 3. Database Lookup Tools

One of the most common uses of tools is:

- retrieving information from a database
    

## Examples

An LLM assistant might:

- check ticket prices
    
- retrieve customer information
    
- look up product inventory
    
- search company records
    
- fetch support data
    

## Why this matters

Without tools, the model only knows:

- its training data
    
- the current prompt
    

With tools, it can access:

- live
    
- dynamic
    
- up-to-date  
    information.
    

---

# 4. Action-Taking Tools

Tools can also allow an LLM-powered assistant to perform actions.

## Examples

The assistant could:

- book a meeting
    
- reserve airline tickets
    
- send messages
    
- update systems
    
- schedule appointments
    

## Key idea

The assistant moves beyond:

- answering questions
    

and begins:

- interacting with external systems
    

This is one of the foundations of practical AI assistants.

---

# 5. Mathematical Calculation Tools

LLMs are traditionally weak at precise arithmetic and formal calculations.

## Problem

A language model predicts plausible tokens, which means:

- arithmetic can be unreliable
    
- calculations may be hallucinated
    

## Solution

Give the system a tool that performs real calculations.

## Important insight

The lecture suggests that systems like ChatGPT likely use tools internally for:

- mathematics
    
- symbolic computation
    
- accurate calculation
    

rather than relying entirely on raw next-token prediction.

---

# 6. Code Execution Tools

Another important category is:

- executing code
    

## Example

An LLM might generate Python code and then execute it to:

- analyze data
    
- compute results
    
- test logic
    
- generate charts
    
- manipulate files
    

---

# 7. Secure Code Execution

The lecture emphasizes that code execution should usually happen inside a secure environment.

Examples:

- Docker containers
    
- sandboxes
    
- isolated runtimes
    

## Why?

Because allowing arbitrary code execution directly on a host machine would be dangerous.

Sandboxing helps:

- contain risk
    
- isolate execution
    
- prevent damage to the broader system
    

---

# 8. What Is a Coder Agent?

The lecture introduces the term:

- **coder agent**
    

## Important clarification

A coder agent does **not** necessarily mean:

- an agent that writes code
    

Instead, it often means:

- an LLM system that can **execute code** as part of solving tasks
    

## Key capability

The agent can:

- generate code
    
- run the code
    
- inspect results
    
- iterate based on feedback
    

This is a major step toward autonomous problem solving.

---

# 9. UI-Manipulation Tools

Tools can also directly affect the user interface.

## Examples

An LLM might:

- generate a chart
    
- update a visualization
    
- change a dashboard
    
- manipulate displayed content
    
- trigger UI interactions
    

## Why this matters

The assistant becomes interactive and dynamic rather than purely text-based.

The system can:

- immediately affect what the user sees
    

---

# 10. There Are Many More Tool Types

The lecturer stresses that the list is not exhaustive.

There are many additional possibilities, including:

- search tools
    
- browser tools
    
- API integrations
    
- file operations
    
- workflow automation
    
- robotics
    
- multimodal systems
    
- retrieval systems
    

The number of possible tools is effectively unlimited.

---

# 11. The Most Exciting Category: Agentic AI

The lecture then shifts focus to the area generating the most excitement:

- **agentic AI**
    

The lecturer identifies two especially important patterns involving tools.

---

# 12. Agentic Pattern #1: LLMs Calling Other LLMs

One powerful idea is:

> a tool can itself be another LLM call

## Example

Suppose an LLM has access to tools:

- Tool A
    
- Tool B
    
- Tool C
    

Each of those tools might actually:

- call a different model
    
- specialize in a different task
    

For example:

- Tool A → summarization model
    
- Tool B → coding model
    
- Tool C → research model
    

---

# 13. Orchestration

This creates a system where one LLM:

- coordinates
    
- routes
    
- orchestrates
    

the activities of multiple other LLMs.

## Key idea

The controlling LLM becomes a workflow manager.

It decides:

- which model to use
    
- when to call it
    
- how to combine outputs
    

This is one of the core ideas behind agentic workflows.

---

# 14. Agentic Workflows

An **agentic workflow** is a workflow where:

- an LLM controls the sequence of actions
    

instead of following a rigid, pre-programmed pipeline.

The LLM becomes responsible for:

- planning
    
- delegation
    
- sequencing
    
- tool selection
    

---

# 15. Agentic Pattern #2: Planning and To-Do Lists

The second major idea involves giving the LLM a planning tool.

## Example

The model can:

- create a to-do list
    
- track progress
    
- mark tasks complete
    
- revise the plan
    
- continue iterating
    

This allows the system to:

- persist across multiple steps
    
- gradually move toward a goal
    

---

# 16. Evaluation and Refinement

A sophisticated agentic system may also:

- evaluate whether tasks are finished
    
- assess quality
    
- refine outputs
    
- continue looping until criteria are met
    

This introduces iterative improvement.

The system does not just:

- answer once
    

It:

- plans
    
- acts
    
- evaluates
    
- revises
    
- continues
    

---

# 17. Agentic Loops

This repeated process is called:

- an **agentic loop**
    

## Structure of an agentic loop

1. Think
    
2. Plan
    
3. Use tools
    
4. Observe results
    
5. Update plan
    
6. Repeat
    

This continues until:

- the objective is achieved
    
- or a stopping condition is reached
    

---

# 18. Why Tools Enable Agentic AI

Tools are essential because they give the agent:

- capabilities
    
- memory structures
    
- planning systems
    
- execution mechanisms
    
- external actions
    

Without tools, the LLM can only:

- generate text
    

With tools, it can:

- interact with the world
    
- retrieve data
    
- manipulate systems
    
- execute workflows
    

---

# 19. Claude Code as a Concrete Example

The lecture points to **Claude Code** as a very visible example of these ideas.

Users can observe:

- task planning
    
- to-do lists
    
- progress tracking
    
- iterative execution
    

This makes the agent loop tangible and easy to understand.

Under the hood, it is still:

- repeated LLM calls
    
- combined with tools and memory structures
    

---

# 20. Key Concept: The LLM Controls the Workflow

One of the defining characteristics of agentic systems is:

> the LLM decides what happens next

It may decide:

- which tool to call
    
- what task to prioritize
    
- whether more work is needed
    
- whether to revise a previous step
    

This creates flexible, dynamic workflows.

---

# 21. Important Clarification

Even though agentic systems seem sophisticated, the underlying mechanism is still fundamentally:

- input sequence → output sequence
    

The difference is:

- the output now includes instructions and actions
    

Those actions then:

- modify the environment
    
- generate new context
    
- feed back into the loop
    

---

# 22. Key Terms

## Tool

An external function or capability available to an LLM system.

## Database lookup tool

A tool that retrieves information from stored data.

## Action-taking tool

A tool that performs external actions like booking or scheduling.

## Coder agent

An LLM system capable of executing code during task completion.

## Sandbox

An isolated environment for safe code execution.

## Orchestration

Coordinating multiple tools or models within a workflow.

## Agentic workflow

A workflow controlled dynamically by an LLM.

## To-do list tool

A planning mechanism that allows an agent to track tasks and progress.

## Agentic loop

A repeated cycle of planning, acting, evaluating, and refining.

---

# 23. Common Tool Use Cases Summary

|Tool Type|Purpose|
|---|---|
|Database lookup|Retrieve live information|
|Booking/action tools|Perform real-world actions|
|Calculation tools|Ensure mathematical accuracy|
|Code execution tools|Run programs and computations|
|UI tools|Modify visual interfaces|
|LLM orchestration tools|Coordinate multiple models|
|Planning tools|Enable long-running workflows|

---

# 24. Relationship Between Tools and Agentic AI

|Without Tools|With Tools|
|---|---|
|Static chatbot|Active assistant|
|Single response|Multi-step workflows|
|No external actions|Can interact with systems|
|Limited memory/planning|Can plan and revise|
|Pure text generation|Real-world task execution|

---

# 25. Main Takeaways

- Tools extend LLM systems beyond pure text generation.
    
- Common uses include:
    
    - database access
        
    - booking actions
        
    - calculations
        
    - code execution
        
    - UI interaction
        
- Secure execution environments are important for code-running tools.
    
- A coder agent is often an LLM that can execute code, not just write it.
    
- Tools are fundamental to modern agentic AI.
    
- One major pattern is using tools that themselves call other LLMs.
    
- Another major pattern is enabling planning and task-tracking loops.
    
- Agentic workflows allow the LLM to orchestrate actions dynamically.
    
- Agentic loops involve repeated cycles of planning, acting, evaluating, and refining.
    

---

# 26. One-Paragraph Revision Summary

Tools allow LLM systems to go beyond generating text by giving them access to external functions such as databases, booking systems, calculators, code execution environments, and UI controls. This makes assistants more accurate, interactive, and capable of performing real-world actions. Tools are also essential for agentic AI, where an LLM dynamically controls workflows, coordinates other LLM calls, plans tasks, tracks progress, and iterates in loops until objectives are achieved. Modern systems like Claude Code demonstrate these ideas by combining repeated LLM calls with tools, planning structures, and execution environments to create autonomous multi-step workflows.