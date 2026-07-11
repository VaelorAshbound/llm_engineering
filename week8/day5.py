#!/usr/bin/env python
# coding: utf-8

# # The Price is Right
#
# The final step is to build a User Interface
#
# We will use more advanced aspects of Gradio - building piece by piece.

# %%


import gradio as gr

from agents.deals import Deal, Opportunity
from deal_agent_framework import DealAgentFramework

# %%


with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
    with gr.Row():
        gr.Markdown(
            '<div style="text-align: center;font-size:24px">The Price is Right - Deal Hunting Agentic AI</div>'
        )
    with gr.Row():
        gr.Markdown(
            '<div style="text-align: center;font-size:14px">Autonomous agent framework that finds online deals, collaborating with a proprietary fine-tuned LLM deployed on Modal, and a RAG pipeline with a frontier model and Chroma.</div>'
        )


ui.launch(inbrowser=True)


# %%


# Updated to change from height to max_height due to change in Gradio v5
# With much thanks to student Ed B. for raising this

with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
    initial_deal = Deal(
        product_description="Example description", price=100.0, url="https://cnn.com"
    )
    initial_opportunity = Opportunity(deal=initial_deal, estimate=200.0, discount=100.0)
    opportunities = gr.State([initial_opportunity])

    def get_table(opps):
        return [
            [
                opp.deal.product_description,
                opp.deal.price,
                opp.estimate,
                opp.discount,
                opp.deal.url,
            ]
            for opp in opps
        ]

    with gr.Row():
        gr.Markdown(
            '<div style="text-align: center;font-size:24px">"The Price is Right" - Deal Hunting Agentic AI</div>'
        )
    with gr.Row():
        gr.Markdown(
            '<div style="text-align: center;font-size:14px">Deals surfaced so far:</div>'
        )
    with gr.Row():
        opportunities_dataframe = gr.Dataframe(
            headers=["Description", "Price", "Estimate", "Discount", "URL"],
            wrap=True,
            column_widths=[4, 1, 1, 1, 2],
            row_count=10,
            col_count=5,
            max_height=400,
        )

    ui.load(get_table, inputs=[opportunities], outputs=[opportunities_dataframe])

ui.launch(inbrowser=True)


# %%


agent_framework = DealAgentFramework()
agent_framework.init_agents_as_needed()

with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
    initial_deal = Deal(
        product_description="Example description", price=100.0, url="https://cnn.com"
    )
    initial_opportunity = Opportunity(deal=initial_deal, estimate=200.0, discount=100.0)
    opportunities = gr.State([initial_opportunity])

    def get_table(opps):
        return [
            [
                opp.deal.product_description,
                opp.deal.price,
                opp.estimate,
                opp.discount,
                opp.deal.url,
            ]
            for opp in opps
        ]

    def do_select(opportunities, selected_index: gr.SelectData):
        row = selected_index.index[0]
        opportunity = opportunities[row]
        agent_framework.planner.messenger.alert(opportunity)

    with gr.Row():
        gr.Markdown(
            '<div style="text-align: center;font-size:24px">"The Price is Right" - Deal Hunting Agentic AI</div>'
        )
    with gr.Row():
        gr.Markdown(
            '<div style="text-align: center;font-size:14px">Deals surfaced so far:</div>'
        )
    with gr.Row():
        opportunities_dataframe = gr.Dataframe(
            headers=["Description", "Price", "Estimate", "Discount", "URL"],
            wrap=True,
            column_widths=[4, 1, 1, 1, 2],
            row_count=10,
            col_count=5,
            max_height=400,
        )

    ui.load(get_table, inputs=[opportunities], outputs=[opportunities_dataframe])
    opportunities_dataframe.select(do_select, inputs=[opportunities], outputs=[])

ui.launch(inbrowser=True)


# # Time for the code
#
# And now we'll move to the price_is_right.py code

# %%


# Reset memory back to 2 deals discovered in the past

from deal_agent_framework import DealAgentFramework

DealAgentFramework.reset_memory()


# %%


import logging

root = logging.getLogger()
root.setLevel(logging.INFO)


# # Running the final product
#
# ## Just hit shift + enter in the next cell, and let the deals flow in!!

# %%


# !uv run price_is_right.py


# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../assets/important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#a22;">But wait!! There's more..</h2>
#             <span style="color:#a22;">If you're not fed up of product prices yet 😂 I've built this out some more!<br/>
#             If you look in my repo <a href="https://github.com/ed-donner/tech2ai">tech2ai</a>, in segment4 is a version that uses OpenAI Agents SDK for the AutonomousPlanningAgent, and makes use of MCP servers.  If you're intrigued by Agents and MCP, and would like to learn more, then I also have a companion course on this. This, along with my 2 other companion courses, are laid out in my <a href="https://edwarddonner.com/2025/05/28/connecting-my-courses-become-an-llm-expert-and-leader/">complete AI curriculum</a>.
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
#             <h2 style="color:#090;">CONGRATULATIONS AND THANK YOU!!!</h2>
#             <span style="color:#090;">
#                 It's so fabulous that you've made it to the very end! My heartiest congratulations. Please stay in touch! I'm <a href="https://www.linkedin.com/in/eddonner/">here on LinkedIn</a> if we're not already connected and I'm on X at <a href="https://x.com/edwarddonner">@edwarddonner</a>. And my editor would be cross with me if I didn't mention one more time: it makes a HUGE difference when students rate this course on Udemy - it's one of the main ways that Udemy decides whether to show it to others. <br/><br/>Massive thanks again for putting up with me for 8 weeks and getting all the way to the final cell! I'm excited to hear all about your career as an LLM Engineer. If you post on LinkedIn about completing the course and tag me, then I'll weigh in to amplify your achievement. <br/><b>You could not have picked a better time to be in this field.</b>
#             </span>
#         </td>
#     </tr>
# </table>

# %%
