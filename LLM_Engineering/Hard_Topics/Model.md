# Complete Study Notes: How Neural Networks, Transformers, and LLMs Work Step by Step

## Example used throughout

We will use the same sentence:

> **“She eats apples”**

The goal is to follow this sentence through the whole model:

```text
Text
→ tokenizer
→ token IDs
→ embeddings
→ transformer decoder layers
→ self-attention
→ MLP/feed-forward network
→ repeated layers
→ LM head
→ next-token probabilities
→ next token
```

---

# 1. Big Picture

A modern language model such as Llama, GPT, or Claude is a large neural network trained to predict the next token.

Given:

```text
She eats apples
```

The model tries to predict what token might come next, for example:

```text
.
```

or:

```text
because
```

or:

```text
every
```

The model does this by turning text into numbers, processing those numbers through many layers, and finally producing scores for all possible next tokens.

---

# 2. Transformer vs Model

A useful distinction:

## Transformer

A **transformer** is the neural network architecture.

It describes the structure of the network, especially:

- embeddings
    
- self-attention
    
- MLP/feed-forward layers
    
- layer normalization
    
- residual connections
    
- repeated transformer blocks
    

## Model

A **model** is a trained instance of that architecture.

For example:

```text
Llama 3.2
GPT
BERT
Claude
```

A model has learned parameters: the actual numbers stored in its weights.

So:

```text
Transformer = architecture
Model = trained transformer with learned weights
```

---

# 3. Neural Network Foundation

A neural network is a stack of layers that transform numbers.

Each layer takes input numbers and produces output numbers.

A basic layer does something like:

```text
output = input × weights + bias
```

The weights are learned during training.

At first, the weights are mostly random. During training, the model makes predictions, measures errors, and adjusts the weights so future predictions become better.

For language models, the main training task is:

> Given previous tokens, predict the next token.

---

# 4. Step 1 — Text Enters the Model

Input sentence:

```text
She eats apples
```

Humans see words. The model cannot directly process words. It needs numbers.

So the first real step is tokenization.

---

# 5. Step 2 — Tokenization

Tokenization splits text into tokens.

For our simplified example:

```text
["She", "eats", "apples"]
```

In a real tokenizer, tokens are not always full words. A word can be split into smaller pieces.

For learning, we will pretend each word is one token.

---

# 6. Step 3 — Token IDs

The tokenizer converts each token into an integer ID.

Example:

| Token  | Token ID |
| ------ | -------: |
| She    |      101 |
| eats   |      245 |
| apples |      879 |

These IDs are just labels.

Important:

```text
Token ID 101 does not mean “smaller” or “less important” than token ID 879.
```

The raw number is arbitrary. The model needs a meaningful vector representation.

That is what the embedding layer does.

---

# 7. Step 4 — Embedding Layer

The embedding layer converts token IDs into vectors.

A vector is a list of numbers.

Toy example:

| Token  | Embedding |
| ------ | --------- |
| She    | [1, 0, 1] |
| eats   | [0, 1, 1] |
| apples | [1, 1, 0] |

In a real Llama-style model, each token might become a much larger vector, such as a 2048-dimensional vector.

So instead of:

```text
She = [1, 0, 1]
```

it is more like:

```text
She = [0.13, -0.44, 1.02, ..., 0.08]
```

with thousands of numbers.

The embedding layer is basically a learned lookup table:

```text
Token ID → learned vector
```

---

# 8. Step 5 — Positional Information

A transformer processes tokens largely in parallel.

That creates a problem:

```text
She eats apples
```

and:

```text
Apples eats she
```

contain similar tokens, but the order is very different.

So the model needs information about token position.

Different transformer models use different positional methods.

Llama-style models use **rotary positional embeddings**, often called **RoPE**.

The purpose is:

> Help the model understand where tokens are in the sequence and how positions relate to each other.

So after token embeddings and positional information, the model has a numeric representation of:

```text
Token meaning + token position
```

---

# 9. Step 6 — Enter the Transformer Decoder Layers

Now the vectors enter the main body of the model.

For a Llama-style model, this is a stack of decoder layers.

Example structure:

```text
Embedding vectors
→ Decoder layer 1
→ Decoder layer 2
→ Decoder layer 3
→ ...
→ Decoder layer 16
→ Final hidden vectors
```

Each decoder layer refines the token representations.

A decoder layer mainly contains:

```text
self-attention
MLP/feed-forward network
layer normalization
residual connections
```

The same general process repeats many times.

---

# 10. Step 7 — What a Decoder Layer Does

A simplified decoder layer looks like this:

```text
input vectors
→ layer norm
→ self-attention
→ residual connection
→ layer norm
→ MLP/feed-forward network
→ residual connection
→ output vectors
```

The key idea:

> Each layer takes the current representation of every token and makes it more context-aware.

For our sentence:

```text
She eats apples
```

At the beginning, “She” has a simple embedding.

After several layers, the vector for “She” contains richer information, such as:

- it is likely a subject
    
- it relates to the action “eats”
    
- it is connected to the object “apples”
    
- it appears before the verb
    

The model does not store these ideas as English sentences. It stores them as patterns in vectors.

---

# 11. Step 8 — Self-Attention: The Main Idea

Self-attention lets each token ask:

> Which other tokens should I pay attention to?

For:

```text
She eats apples
```

The token “She” may pay attention to:

- itself
    
- “eats”
    
- “apples”
    

The token “eats” may pay attention to:

- “She” as the subject
    
- “apples” as the object
    

The token “apples” may pay attention to:

- “eats” because apples are being eaten
    
- “She” because she is doing the eating
    

Self-attention allows tokens to exchange information.

---

# 12. Step 9 — Query, Key, and Value

Inside self-attention, each token creates three vectors:

```text
Query
Key
Value
```

They mean:

| Vector | Question it answers               |
| ------ | --------------------------------- |
| Query  | What am I looking for?            |
| Key    | What do I contain?                |
| Value  | What information do I pass along? |

There is also usually an output projection after attention.

So real attention uses:

```text
Q, K, V, O
```

where O means output projection.

---

# 13. Step 10 — Toy Attention Setup

Use the same simplified embeddings:

| Token  | Embedding |
| ------ | --------- |
| She    | [1, 0, 1] |
| eats   | [0, 1, 1] |
| apples | [1, 1, 0] |

Now choose fake learned weight matrices:

## Query matrix WQ

```text
[ 1   0 ]
[ 0   1 ]
[ 1   1 ]
```

## Key matrix WK

```text
[ 1   1 ]
[ 0   1 ]
[ 1   0 ]
```

## Value matrix WV

```text
[ 1   0 ]
[ 0   2 ]
[ 1   1 ]
```

These are fake small numbers for learning.

In a real model, these matrices are huge and learned during training.

---

# 14. Step 11 — Create Q, K, and V for “She”

The embedding for “She” is:

```text
She = [1, 0, 1]
```

## Query

```text
Q_she = [1, 0, 1] × WQ
      = [2, 1]
```

## Key

```text
K_she = [1, 0, 1] × WK
      = [2, 1]
```

## Value

```text
V_she = [1, 0, 1] × WV
      = [2, 1]
```

The model does the same for every token.

---

# 15. Step 12 — Q, K, and V for All Tokens

Using the same toy weights:

| Token  | Q      | K      | V      |
| ------ | ------ | ------ | ------ |
| She    | [2, 1] | [2, 1] | [2, 1] |
| eats   | [1, 2] | [1, 1] | [1, 3] |
| apples | [1, 1] | [1, 2] | [1, 2] |

Now attention can compare tokens.

---

# 16. Step 13 — Attention Scores

To understand how “She” attends to the sentence, compare:

```text
Q_she
```

with each token’s key:

```text
K_she
K_eats
K_apples
```

Using dot products:

```text
score(She → She)    = [2, 1] · [2, 1] = 5
score(She → eats)   = [2, 1] · [1, 1] = 3
score(She → apples) = [2, 1] · [1, 2] = 4
```

So the raw attention scores are:

```text
[5, 3, 4]
```

These scores say how strongly “She” matches each token.

---

# 17. Step 14 — Softmax

Raw scores are converted into probabilities using softmax.

For:

```text
[5, 3, 4]
```

softmax gives approximately:

| Token  | Score | Attention weight |
| ------ | ----: | ---------------: |
| She    |     5 |            0.665 |
| eats   |     3 |            0.090 |
| apples |     4 |            0.245 |

These weights add up to 1.

So “She” pays attention roughly:

```text
66.5% to She
9.0% to eats
24.5% to apples
```

---

# 18. Step 15 — Weighted Sum of Values

Now the model combines value vectors using the attention weights.

Values:

```text
V_she    = [2, 1]
V_eats   = [1, 3]
V_apples = [1, 2]
```

Weighted sum:

```text
output_she =
0.665 × [2, 1]
+ 0.090 × [1, 3]
+ 0.245 × [1, 2]
```

Calculate:

```text
0.665 × [2, 1] = [1.330, 0.665]
0.090 × [1, 3] = [0.090, 0.270]
0.245 × [1, 2] = [0.245, 0.490]
```

Add them:

```text
[1.330, 0.665]
+ [0.090, 0.270]
+ [0.245, 0.490]
= [1.665, 1.425]
```

So after attention:

```text
new_she = [1.665, 1.425]
```

This is a context-aware version of “She.”

---

# 19. Step 16 — Repeat Attention for Every Token

The same process happens for:

```text
She
Eats
Apples
```

Each token gets a new vector that includes information from the other tokens.

Toy final self-attention output:

```text
[
  [1.665, 1.425],  # She
  [1.425, 1.820],  # eats
  [1.421, 1.731]   # apples
]
```

The exact toy numbers are not important.

The important idea is:

> Self-attention updates each token by mixing information from the whole available context.

---

# 20. Step 17 — Causal Masking in Decoder-Only Models

Llama/GPT-style models are decoder-only and autoregressive.

That means they predict the next token using only previous tokens.

So during training or generation, the model must not cheat by looking at future tokens.

This is handled with a **causal mask**.

For a sequence:

```text
She eats apples
```

when predicting from position 1, the model can see only:

```text
She
```

when predicting from position 2, it can see:

```text
She eats
```

when predicting from position 3, it can see:

```text
She eats apples
```

So decoder-only self-attention usually means:

> Each token can attend to itself and earlier tokens, but not future tokens.

For teaching, we sometimes let all tokens attend to all tokens. But for GPT/Llama-style generation, causal masking matters.

---

# 21. Step 18 — Output Projection

After attention creates mixed vectors, a real transformer usually applies an output projection.

Conceptually:

```text
attention result × WO
```

where:

```text
WO = learned output projection matrix
```

This maps the attention result back into the model’s working dimension.

In a real model, if the hidden size is 2048, the attention output also returns to 2048 dimensions so the next layer can process it.

---

# 22. Step 19 — Residual Connection After Attention

A residual connection adds the original input back to the transformed output.

Simplified:

```text
attention_output = self_attention(input)
new_vector = input + attention_output
```

Why?

Because deep networks can lose information as it flows through many layers.

Residual connections help preserve useful information and make training more stable.

So attention does not completely replace the old vector. It updates it.

---

# 23. Step 20 — Layer Normalization

Layer normalization keeps numbers stable.

In deep networks, values can become too large, too small, or unstable.

Layer norm helps keep each vector in a healthier numerical range.

Simplified intuition:

```text
layer norm = keep the numbers well-behaved
```

This helps the model train and run reliably across many layers.

---

# 24. Step 21 — MLP / Feed-Forward Network

After attention, each token vector goes through an MLP.

MLP means:

```text
Multi-Layer Perceptron
```

This is the feed-forward part of the transformer layer.

Attention mixes information between tokens.

The MLP transforms each token’s vector individually.

So:

| Component      | What it does                           |
| -------------- | -------------------------------------- |
| Self-attention | Mixes information across tokens        |
| MLP            | Processes each token vector internally |

---

# 25. Step 22 — Toy MLP Example

Use an attention output for “She”:

```text
x_she = [1.665, 1.425]
```

A tiny toy MLP:

```text
Input size: 2
Hidden size: 3
Output size: 2
```

First weight matrix:

```text
W1 = [
  [1, 0, 1],
  [0, 1, 1]
]
```

Bias:

```text
b1 = [0, 0, 0]
```

Activation:

```text
ReLU(x) = max(0, x)
```

Second weight matrix:

```text
W2 = [
  [1, 0],
  [0, 1],
  [1, 1]
]
```

---

# 26. Step 23 — MLP First Linear Layer

Input:

```text
x_she = [1.665, 1.425]
```

Multiply by W1:

```text
[1.665, 1.425] × W1
```

Result:

```text
[
  1.665,
  1.425,
  1.665 + 1.425
]
```

So:

```text
[1.665, 1.425, 3.090]
```

The MLP has expanded the vector from 2 dimensions to 3 dimensions.

In real transformer MLPs, this expansion can be much larger, for example:

```text
2048 → around 8192 → 2048
```

---

# 27. Step 24 — Activation Function

Apply ReLU:

```text
ReLU([1.665, 1.425, 3.090])
```

Since all values are positive:

```text
[1.665, 1.425, 3.090]
```

The activation function adds non-linearity.

This is crucial.

Without non-linearity, many layers of matrix multiplication would collapse into one equivalent linear transformation.

Non-linearity makes deep neural networks powerful.

Real Llama-style models often use smoother activations such as SiLU/Swish-like functions rather than basic ReLU.

---

# 28. Step 25 — MLP Second Linear Layer

Now compress back down:

```text
[1.665, 1.425, 3.090] × W2
```

Calculate:

```text
first output  = 1.665 + 3.090 = 4.755
second output = 1.425 + 3.090 = 4.515
```

So:

```text
MLP_output_she = [4.755, 4.515]
```

This is the transformed vector for “She” after the MLP.

---

# 29. Step 26 — Residual Connection After MLP

Again, the model adds the previous vector back in:

```text
new_she = previous_she + MLP_output_she
```

This helps preserve information while still allowing the layer to modify the representation.

So one decoder layer has now done two big things:

```text
1. attention mixed information across tokens
2. MLP transformed each token representation
```

---

# 30. Step 27 — Repeat Across Many Layers

One layer is not enough.

The model repeats this process many times:

```text
Layer 1: basic relationships
Layer 2: richer relationships
Layer 3: more abstract patterns
...
Layer 16: final refined representation
```

For our sentence:

```text
She eats apples
```

early layers may capture simple relationships like:

```text
She → eats
```

middle layers may capture grammar-like patterns:

```text
subject → verb → object
```

later layers may capture prediction-relevant information:

```text
This is a complete simple sentence. A period may come next.
```

The model does not explicitly write these rules. They are encoded in the vectors and weights.

---

# 31. Step 28 — Final Hidden Vectors

After all decoder layers, each token has a final hidden vector.

Simplified:

```text
final_She    = [...]
final_eats   = [...]
final_apples = [...]
```

For next-token prediction, the most important vector is usually the final vector at the last position:

```text
final_apples
```

Why?

Because the model predicts the next token after the whole current sequence:

```text
She eats apples ___
```

The last hidden vector summarizes the context available so far.

---

# 32. Step 29 — LM Head

The LM head means:

```text
Language Model Head
```

Its job:

> Convert the final hidden vector into scores for every possible next token.

If the vocabulary has 128,256 tokens, the LM head outputs 128,256 scores.

Example:

| Candidate next token | Score |
| -------------------- | ----: |
| .                    |   8.2 |
| because              |   5.1 |
| and                  |   4.7 |
| every                |   2.3 |
| banana               |  -1.5 |

These are raw scores, often called logits.

---

# 33. Step 30 — Softmax Over Vocabulary

The raw scores are converted into probabilities.

Example:

| Candidate next token | Probability |
| -------------------- | ----------: |
| .                    |        0.52 |
| because              |        0.14 |
| and                  |        0.10 |
| every                |        0.04 |
| banana               |       0.001 |

Now the model has a probability distribution over possible next tokens.

---

# 34. Step 31 — Choosing the Next Token

The model can choose the next token in different ways.

## Greedy decoding

Pick the highest probability token:

```text
.
```

Output:

```text
She eats apples.
```

## Sampling

Randomly sample from likely tokens.

This can produce more varied text.

## Temperature

Temperature controls randomness.

Lower temperature:

```text
more predictable
```

Higher temperature:

```text
more creative/random
```

---

# 35. Step 32 — Autoregressive Generation

After choosing the next token, the model appends it to the text.

```text
She eats apples.
```

Then it can run again to predict the next token after that.

This loop continues:

```text
predict token
append token
predict token
append token
predict token
append token
```

That is autoregressive text generation.

---

# 36. Training: How the Model Learns

During training, the model sees huge amounts of text.

Example training sequence:

```text
She eats apples.
```

The model is trained on many prediction tasks:

```text
Given: She
Predict: eats

Given: She eats
Predict: apples

Given: She eats apples
Predict: .
```

At first, the model predicts badly.

Then training compares the prediction with the correct next token.

The difference is the loss/error.

Backpropagation adjusts the weights to reduce that error.

Over many examples, the model learns statistical patterns of language.

---

# 37. What Gets Learned?

The model learns many kinds of weights:

```text
embedding weights
attention projection weights
MLP weights
layer norm parameters
LM head weights
```

These weights determine:

- how tokens are represented
    
- which tokens attend to which other tokens
    
- how features are transformed
    
- which next tokens become likely
    

The model does not learn by storing a dictionary of answers.

It learns by adjusting huge matrices of numbers.

---

# 38. Why Attention Matters

Attention matters because words depend on context.

Example:

```text
She eats apples
```

The meaning of “eats” depends on:

```text
She
apples
```

The meaning of “apples” depends on:

```text
eats
```

Self-attention allows every token to build a context-aware meaning.

A token is no longer just itself.

It becomes:

```text
itself + relevant context
```

---

# 39. Why the MLP Matters

Attention mixes information between tokens.

But after the information is mixed, the model still needs to transform it.

The MLP helps the model create richer features.

For example, after attention, the model might have information like:

```text
subject = She
verb = eats
object = apples
```

The MLP can transform that into more useful internal patterns for prediction.

In simple terms:

```text
attention = communication between tokens
MLP = thinking inside each token vector
```

---

# 40. Why Layer Norm and Residuals Matter

Very deep networks are hard to train.

Two tools make them easier:

## Layer normalization

Keeps numbers stable.

## Residual connections

Let information flow through the network without being destroyed.

Together, they help the model train reliably across many layers.

---

# 41. Why Non-Linearity Matters

If every layer were only linear, then this:

```text
linear layer
→ linear layer
→ linear layer
```

could be simplified into:

```text
one linear layer
```

That would make depth much less useful.

Activation functions such as ReLU or SiLU add non-linearity.

This allows the network to learn complex patterns.

In simple words:

```text
Without activation functions, deep networks are just big linear mixers.
With activation functions, they become powerful pattern learners.
```

---

# 42. Encoder-Decoder vs Decoder-Only

The original transformer architecture had both:

```text
encoder + decoder
```

This is useful for tasks like translation.

Example:

```text
English sentence → encoder
French sentence generation → decoder
```

Modern LLMs such as GPT and Llama are usually:

```text
decoder-only
```

They are designed to generate text from left to right.

So they use repeated decoder layers and causal masking.

---

# 43. The Complete Flow Using “She eats apples”

Here is the full story:

```text
1. Text input
   "She eats apples"

2. Tokenization
   ["She", "eats", "apples"]

3. Token IDs
   [101, 245, 879]

4. Embeddings
   She    → [1, 0, 1]
   eats   → [0, 1, 1]
   apples → [1, 1, 0]

5. Positional information
   The model adds/uses information about token order.

6. Decoder layer 1
   a. Layer norm
   b. Self-attention
      - create Q, K, V
      - compare queries with keys
      - softmax scores
      - weighted sum of values
   c. Output projection
   d. Residual connection
   e. Layer norm
   f. MLP/feed-forward
      - expand vector
      - activation/non-linearity
      - compress vector
   g. Residual connection

7. Decoder layers repeat
   The vectors become more refined each time.

8. Final hidden vector
   The last token’s vector summarizes the context:
   "She eats apples"

9. LM head
   Converts final vector into scores for every possible next token.

10. Softmax
    Converts scores into probabilities.

11. Next-token selection
    The model chooses something like "."

12. Output
    "She eats apples."
```

---

# 44. Simple Mental Model

Think of the model like this:

```text
Tokenizer = turns text into token IDs
Embedding layer = turns token IDs into meaning vectors
Attention = lets tokens talk to each other
MLP = transforms each token’s internal representation
Layer norm = keeps numbers stable
Residual connections = preserve information
Decoder layers = repeat attention + MLP many times
LM head = predicts the next token
Softmax = turns scores into probabilities
Sampling = chooses the next token
```

---

# 45. One-Sentence Summary

A decoder-only transformer language model turns text into token vectors, repeatedly refines those vectors using self-attention, MLPs, normalization, and residual connections, then uses the final vector to predict the probability of every possible next token.

---

# 46. Final Intuition

For:

```text
She eats apples
```

The model does not “understand” the sentence the way a human does.

Instead, it builds numerical representations where:

- “She” becomes related to the action “eats”
    
- “eats” becomes related to the subject and object
    
- “apples” becomes related to the eating action
    
- the final representation suggests likely next tokens
    

After training on huge amounts of text, these numerical patterns become powerful enough to produce fluent and useful language.

That is the core of how neural networks, transformers, and language models fit together.