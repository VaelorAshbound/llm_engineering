# Meeting minutes creator

In this colab, we make a meeting minutes program.

It includes useful code to connect your Google Drive to your colab.

Upload your own audio to make this work!!

https://colab.research.google.com/drive/1KSMxOCprsl1QRpt_Rq0UqCAyMtPqDQYx?usp=sharing

This should run nicely on a low-cost or free T4 box.

### BUT FIRST - Something cool - really showing you how "model inference" works via OpenAI


```python
from visualizer import TokenPredictor, create_token_graph, visualize_predictions

message = "In one sentence, describe the color orange to someone who has never been able to see"
model_name = "gpt-4.1-mini"

predictor = TokenPredictor(model_name)
predictions = predictor.predict_tokens(message)
G = create_token_graph(model_name, predictions)
plt = visualize_predictions(G)
plt.show()
```


```python

```
