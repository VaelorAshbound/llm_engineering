#!/usr/bin/env python
# coding: utf-8

# %%


import plotly.graph_objects as go

results = [
    ("Constant", "gray", 106.18),
    ("Linear Regression", "gray", 101.56),
    ("NLP + LR", "gray", 76.81),
    ("Random Forest", "gray", 72.28),
    ("XGBoost", "gray", 68.23),
    ("Human (Ed)", "black", 87.62),
    ("Neural Network", "orange", 63.97),
    ("GPT 4.1 Nano", "slateblue", 62.51),
    ("Grok 4.1 Fast", "slateblue", 57.62),
    ("Gemini 3 Pro", "slateblue", 50.54),
    ("Claude 4.5 Sonnet", "slateblue", 47.10),
    ("GPT 5.1", "slateblue", 44.74),
    ("GPT 4.1 Nano (Fine-tuned)", "skyblue", 75.91),
    ("Deep Neural Network", "orange", 46.49),
    ("Base Llama 3.2 4 bit", "darkred", 110.72),
    ("Fine-tuned Lite", "red", 65.40),
    ("Fine-tuned Full", "red", 39.85),
    ("GPT 5.1 RAG", "green", 30.19),
    ("Ensemble", "purple", 29.9),
]

labels, colors, values = zip(*results)

fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))

fig.update_layout(
    title="Prediction error from each model",
    yaxis=dict(range=[0, max(values)], title="Error"),
    xaxis=dict(tickangle=-45),
    width=1000,
    height=800,
)

fig.show()


# %%


# %%
