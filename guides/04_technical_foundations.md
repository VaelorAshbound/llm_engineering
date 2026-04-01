# Technical Foundations

It's crucial that you feel comfortable with the basic technical concepts that we work with. This will make your experience of the entire course so much better - it can be very frustrating if you're not sure what's gong on.

These guides should build confidence in the underlying technologies we work with.

## Topic 1: ChatGPT versus OpenAI API

### What’s the difference between ChatGPT and the GPT API, both offered by OpenAI?

#### ChatGPT is an end-user tool. It’s a Chat product designed for consumers who are AI users.
- It has a free plan, and it also has paid subscription plans with more features.
- The subscription plans give the user near-unlimited access to use the Chat product.

#### The API is a service provided for AI engineers - software engineers and data scientists - working on other commercial products.
- It allows technical people, like you and me, to access the underlying models (like “GPT4.1” and “o3”) so that we can build our own products.
- If we wanted to, we could build our own version of ChatGPT using the API, and charge our end-users for it.
- Like most APIs, OpenAI charges a small amount based on API usage. For most examples on the course using gpt-4o-mini, it’s of the order of $0.001 per API call.

### I’m paying $20/month for ChatGPT - why do I need to pay more for the API?

- Hopefully this is now clear. The API is not for consumers; it’s for engineers to build their own platforms that they can charge for.
- If you were to have access to the API based on your subscription, then you could offer ChatGPT tools to others at a cheaper price, and put OpenAI out of business!
- Keep in mind: each API call may require 10,000,000,000,000 floating point calculations - that compute uses electricity!

Instead of calling the API, you can run open source models locally, but typically they have 1,000 times fewer calculations — and even though it’s tiny, that processing still hits your electricity bill..

## Topic 2: Taking a Screenshot

You may already be familiar with "taking a screenshot" on your computer, but if not (or if you think this means taking a photo with your camera..), please review this tutorial:

https://chatgpt.com/share/681f691b-6644-8012-b07d-207c68f259d5

## Topic 3: Environment Variables and the `.env` file

This tutorial walks you through everything you need to know about .env files!

Obiously you don't need to add the .env file to .gitignore, as I've already done that for you. But it hopefully explains the point well.

https://chatgpt.com/share/68061e89-dd84-8012-829d-9f4506c7baaa

## Topic 4: Networking basics

This tutorial covers networking and typical issues with certificates, VPNs, DNS and the like.

The sections give a summary; you should ask ChatGPT to expand on any section if it's relevant to your situation.

https://chatgpt.com/share/680620ec-3b30-8012-8c26-ca86693d0e3d

This is a more in-depth guide to tackling SSL / certificate issues, which is common in corporate environments:

https://chatgpt.com/share/69934d95-85dc-8012-af71-497d17fcb0a6

## Topic 5: APIs and Client Libraries - foundational briefing

We use APIs a lot in this course!

It's essential to understand the fundamentals of what's going on when we make a call to an API, and to be comfortable with words like "endpoint" and "client library".

Please review this guide:

https://chatgpt.com/share/68062432-43c8-8012-ad91-6311d4ad5858

## Topic 6: uv, package management, environment management

This lays out the Dependency Management situation and why we love uv! And a crash course in how to use it.

https://chatgpt.com/share/68c34d46-18a0-8012-8d65-0a0cce615912

Note that this guide suggests `uv run python xxx` which works fine, but simply `uv run xxx` works too and is more common.
