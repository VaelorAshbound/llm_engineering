# 📘 Study Notes: Tool Calling in LLMs

---

## 1. Overview of the Lecture

This lecture introduces **tools**, one of the most important ideas in modern LLM applications.

Tools are essential for:

- commercial chatbot assistants
    
- agentic AI
    
- database lookups
    
- calculations
    
- code execution
    
- real-world actions
    

By the end of this topic, students should understand:

- what tools are
    
- why they matter
    
- how tool calling actually works
    
- why LLMs do **not** literally run code themselves
    

---

# 2. Course Progress So Far

By this point, students can already:

- understand basic transformer terminology
    
- use the Chat Completions API
    
- call GPT and Claude through APIs
    
- build an AI chatbot assistant
    
- work with interactive API-based applications
    

The next step is learning how to make assistants more powerful using **tools**.

---

# 3. What Are Tools?

## Definition

A **tool** is an external function that an LLM-powered application can use.

A tool is usually:

- a function
    
- a piece of code
    
- an API call
    
- a database lookup
    
- a calculator
    
- a code execution environment
    
- an action such as booking or updating something
    

## Key idea

Giving a model tools means allowing the model-powered system to connect to external functionality.

---

# 4. Why Tools Matter

Tools let LLM applications go beyond just text generation.

They allow systems to:

## Extend model knowledge

Example:

- retrieve information from a database
    
- look up ticket prices
    
- search documents
    
- query product catalogs
    

## Carry out actions

Example:

- book a ticket
    
- update a customer record
    
- send a message
    
- create a calendar event
    

## Improve accuracy

Example:

- use a calculator rather than guessing arithmetic
    
- run code rather than hallucinating code output
    
- check facts from a reliable data source
    

## Build commercial assistants

Tools are a foundation for practical business applications such as:

- customer support bots
    
- travel assistants
    
- booking systems
    
- internal company assistants
    

## Enable agentic AI

Tools are also a basic building block of **agentic AI**, where an LLM can decide what action to take next.

---

# 5. Common Misunderstanding About Tools

A common first impression is:

> “The LLM is running my code.”

But that is **not** what actually happens.

This can feel confusing because an LLM is just:

- a neural network
    
- a statistical model
    
- a system that predicts tokens
    

So how could it suddenly:

- connect to your computer?
    
- run Python?
    
- call your database?
    
- execute a booking function?
    

The answer is: it does not.

---

# 6. What LLMs Actually Do

An LLM still only does one thing:

- it generates tokens
    

Even when using tools, the model is still just producing text or structured output.

It does not directly run code.

It does not magically reach into your system.

It does not execute a Python function by itself.

---

# 7. The “Theory” Version of Tool Calling

People often describe tool calling as if this happens:

1. Your code calls the LLM.
    
2. The LLM decides to use a tool.
    
3. The LLM directly runs your function.
    
4. The LLM gets the result.
    
5. The LLM answers the user.
    

This is a useful simplified explanation, but it is not what literally happens.

---

# 8. The Real Version of Tool Calling

In practice, tool calling works like this:

1. Your code sends a prompt to the LLM.
    
2. The prompt includes descriptions of available tools.
    
3. The LLM replies with a structured request saying which tool it wants to use.
    
4. Your code reads that response.
    
5. Your code runs the actual function.
    
6. Your code gets the tool result.
    
7. Your code sends a second message to the LLM containing:
    
    - the original conversation
        
    - the model’s tool request
        
    - the result returned by the tool
        
8. The LLM uses that updated conversation history to generate the final answer.
    

## Key takeaway

The LLM requests the tool.  
Your application executes the tool.

---

# 9. Tool Calling Is Just Conversation History

Tool calling works by adding more messages to the conversation.

The conversation may include:

- user message
    
- assistant message saying “call this tool”
    
- tool result message
    
- assistant final answer
    

From the model’s perspective, it is still only responding based on the conversation history.

There is no magic.

---

# 10. Tool Calling Is Stateless

Each LLM call is usually **stateless**.

That means the model does not automatically remember what happened before.

To continue the interaction, your code must send the relevant history again.

So after a tool is called, your code sends back:

- the original user request
    
- the assistant’s request to call the tool
    
- the tool’s returned data
    

Then the model can respond as though it “knows” what the tool returned.

---

# 11. How Does the Model Know It Can Use Tools?

The model knows because you tell it.

In the initial request, your code includes instructions such as:

- what tools are available
    
- what each tool does
    
- what arguments each tool accepts
    
- what format to use when requesting a tool
    

Modern models have been trained to understand this format, often using structured JSON.

---

# 12. Tool Definitions

A tool definition usually describes:

- the tool name
    
- what the tool does
    
- the input arguments
    
- the expected argument types
    
- when the tool should be used
    

For example, a flight-price tool might include:

- tool name: `fetch_ticket_price`
    
- input: `city`
    
- output: ticket price for that city
    

---

# 13. Simple Example: Airline Support Agent

The lecturer gives a very simple example using ChatGPT.

## Prompt idea

The model is told:

- You are an airline support agent.
    
- You can query ticket prices.
    
- To get a ticket price, respond with:
    
    - “use tool to fetch ticket price for [city]”
        

Then the user asks:

> “I’d like to go to Paris. How much is a flight?”

## Model response

Instead of answering directly, the model replies:

> “use tool to fetch ticket price for Paris”

## What this demonstrates

The model has not called a real tool.

It has simply followed the instruction and generated text asking for the tool to be used.

Your code would then detect that response and actually run the function.

---

# 14. The Application Code’s Role

Your code is responsible for the real work.

It must:

1. Detect that the model requested a tool.
    
2. Identify which tool was requested.
    
3. Extract the tool arguments.
    
4. Run the appropriate function.
    
5. Capture the result.
    
6. Send that result back to the model.
    
7. Ask the model to produce the final response.
    

In simple terms, your code needs an `if` statement or equivalent logic:

- if the model requests a tool, call the tool
    
- otherwise, return the response directly
    

---

# 15. Why JSON Is Often Used

Modern APIs often use JSON to define and request tools.

## Why JSON?

Because it is:

- structured
    
- easy for code to parse
    
- easy for APIs to validate
    
- familiar to models because they have been trained on many examples
    

A model might respond with something like:

```json
{
  "tool": "fetch_ticket_price",
  "arguments": {
    "city": "Paris"
  }
}
```

Your application can then parse this and call:

```python
fetch_ticket_price("Paris")
```

---

# 16. The Tool Calling Loop

A typical tool calling loop looks like this:

1. User asks a question.
    
2. App sends question + tool definitions to model.
    
3. Model requests a tool.
    
4. App executes the tool.
    
5. App sends tool result back to model.
    
6. Model gives final answer.
    

This is the foundation of many modern assistant systems.

---

# 17. Why Tools Are So Powerful

Tools allow LLMs to overcome some of their natural limitations.

Without tools, an LLM can only use:

- its training data
    
- the prompt
    
- the conversation history
    

With tools, an LLM-powered system can access:

- live data
    
- private databases
    
- external APIs
    
- calculators
    
- code execution
    
- business systems
    

This makes the assistant much more useful in real applications.

---

# 18. Important Clarification

Tool calling does **not** mean the model has become a normal program that can execute arbitrary functions.

Instead:

- the model produces a structured request
    
- the surrounding application executes it
    
- the result is passed back into the conversation
    

So tools are really a collaboration between:

- the LLM
    
- your application code
    
- external services/functions
    

---

# 19. Relationship to Agentic AI

Tools are a building block of **agentic AI**.

An agentic system often involves:

- an LLM deciding what step to take next
    
- selecting tools
    
- using tool outputs
    
- looping until a task is complete
    

So understanding tools is necessary before building more advanced agents.

---

# 20. Mental Model

Think of the LLM as saying:

> “To answer this properly, I need the result of this function.”

Then your application says:

> “Okay, I will run that function and give you the result.”

Then the LLM says:

> “Now that I have the result, here is the answer.”

That is the essence of tool calling.

---

# 21. Key Terms

## Tool

An external function or capability available to an LLM-powered application.

## Tool calling

The process where a model requests that a tool be used, and the application executes it.

## Tool definition

A structured description of a tool’s name, purpose, and inputs.

## Tool result

The output from the actual function/API/database call.

## Stateless API call

A model call that does not remember previous messages unless they are sent again.

## JSON

A structured data format often used to define tools and pass arguments.

## Agentic AI

A system where an LLM can choose actions, use tools, and operate in loops.

---

# 22. Main Takeaways

- Tools let LLM systems access external functions, data, and actions.
    
- The LLM does **not** directly run code.
    
- The LLM only generates tokens, including structured tool requests.
    
- Your application code detects the tool request and executes the function.
    
- The tool result is sent back to the model as part of conversation history.
    
- Tool calling is just structured messaging plus application logic.
    
- Tools are essential for commercial assistants and agentic AI.
    

---

# 23. One-Paragraph Revision Summary

Tool calling lets an LLM-powered application connect to external functions such as database lookups, calculators, booking systems, or code execution. However, the LLM itself does not directly run the tool. It only generates a structured request saying which tool it wants and with what arguments. The surrounding application code detects that request, runs the actual function, then sends the tool result back to the model as part of the conversation history. The model then uses that information to produce the final answer. This simple pattern is the foundation of practical AI assistants and agentic AI systems.

---