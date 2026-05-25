#LLMs #NeuralNetworks
### The basic idea

A neural network is a computer system that learns by example, rather than being told exact rules.

### An everyday analogy

Imagine teaching a child to recognize cats:

- You show them many pictures of cats and non-cats.
    
- At first, they guess almost randomly.
    
- After each guess, they check the real answer.
    
- If they were wrong, they slightly adjust what they pay attention to next time.
    
- Over many examples, they get better at noticing useful patterns like fur, ears, whiskers, and shape.
    

A neural network learns in a very similar way.

The main difference is that the child thinks consciously, while the neural network adjusts numbers mathematically.

### How it’s structured

A neural network is made of many tiny units called **neurons**, arranged in layers:

1. **Input layer** – takes in data (like pixels of an image or words in a sentence)
    
2. **Hidden layers** – process the information and look for patterns
    
3. **Output layer** – gives the final answer (e.g., “cat” or “not cat”)
    

Each neuron:

- Receives numbers
    
- Weighs how important they are
    
- Combines them
    
- Passes the result forward
    

### How learning happens

1. The network makes a guess.
    
2. It checks how wrong the guess was.
    
3. It slightly adjusts its internal “weights” to do better next time.
    
4. This repeats thousands or millions of times.
    

Over time, the network becomes accurate.

---

# 🧠 Neural Network Flow (5 Neurons Total)

We’ll use:

### **Layers**

- **Input layer:** 2 neurons
    
- **Hidden layer:** 2 neurons
    
- **Output layer:** 1 neuron
    

👉 **Total = 5 neurons**

---

# 🧩 Step 1: The problem

We want to answer:

> **“Will this person buy a product?”**

Input data:

- `x₁ = Age`
    
- `x₂ = Website visits`
    

Output:

- `y = 1` → buys
    
- `y = 0` → doesn’t buy
    

---

# 🔢 Step 2: Input values arrive

Example person:

- Age = 30
    
- Visits = 1
    

So input neurons hold:

```
x₁ = 30
x₂ = 1
```

---

# ⚙️ Step 3: Forward pass (data flows forward)

Each hidden neuron:

- multiplies inputs by weights
    
- adds a bias
    
- passes result through an activation function
    

When a neural network is first created:

- Every **neuron** starts with a small random **weight**
- Every **neuron has a bias**, also usually random (or sometimes zero)

Let’s assume these weights:

### Hidden neuron h₁

- weight from x₁ → 0.05
    
- weight from x₂ → 0.6
    
- bias = -1
    

Calculation:

```
(30 × 0.05) + (1 × 0.6) - 1
= 1.5 + 0.6 - 1
= 1.1
```

Activation squashes it:

```
h₁ ≈ 0.75
```

---

### Hidden neuron h₂

- weight from x₁ → 0.02
    
- weight from x₂ → 0.4
    
- bias = -0.5
    

Calculation:

```
(30 × 0.02) + (1 × 0.4) - 0.5
= 0.6 + 0.4 - 0.5
= 0.5
```

Activation:

```
h₂ ≈ 0.62
```

## What are Activation Functions?

Activation functions are mathematical functions applied to the output of a neuron after the weighted sum and bias. They transform the neuron’s raw output before passing it to the next layer.

A neuron typically computes:

z = Wx + b

Then activation is applied:

a = f(z)

where (f) is the activation function.

Common examples include Rectified Linear Unit, Sigmoid, and Tanh.

---

## Why are Activation Functions Needed?

Without activation functions, every layer would only perform linear/affine transformations. Stacking many such layers still results in a single linear transformation:

W_3(W_2(W_1x)) = (W_3W_2W_1)x

So **linear of a linear is still linear**.

That means a deep network without activations would be no more powerful than one layer and could only learn simple straight-line relationships.

Activation functions introduce **nonlinearity**, allowing the network to learn:

- complex patterns
    
- curved boundaries
    
- feature interactions
    
- hierarchical representations
    
- image, language, and speech patterns
    

Example using ReLU:

ReLU(x)=max⁡(0,x)

This creates piecewise behavior instead of one straight mapping.

---

## Key Idea to Remember

> Activation functions make depth useful.  
> Without them, many layers collapse into one linear layer.

---

# 🎯 Step 4: Hidden → Output layer

The output neuron takes `h₁` and `h₂`.

Weights:

- h₁ → 0.8
    
- h₂ → 0.6
    
- bias = -0.7
    

Calculation:

```
(0.75 × 0.8) + (0.62 × 0.6) - 0.7
= 0.6 + 0.372 - 0.7
= 0.272
```

Activation:

```
output ≈ 0.57
```

### 🟢 Final prediction:

**57% chance of buying → predicts BUY**

---

# 🔍 Step 5: Compare with reality (loss)

Let’s say the **true answer = 1 (they bought)**

Error:

```
1 - 0.57 = 0.43
```

This error tells the network:

> “You were right, but not confident enough.”

---

# 🔁 Step 6: Backpropagation (learning)

Now the network works **backwards**:

### Output neuron:

- “I need to increase my output”
    
- So it slightly increases weights from h₁ and h₂
    

### Hidden neurons:

- They check how much they contributed to the error
    
- Adjust their incoming weights slightly
    

Example updates:

- h₁ weight: 0.8 → 0.82
    
- h₂ weight: 0.6 → 0.63
    
- biases also slightly adjusted
    

⚠️ Changes are tiny — learning is slow and stable.

---

# 🔄 Step 7: Repeat (training loop)

The entire process repeats for:

- thousands or millions of examples
    

Each cycle:

1. Forward pass
    
2. Measure error
    
3. Backpropagate
    
4. Update weights
    

Over time:

- Good patterns strengthen
    
- Bad ones weaken
    

---

# 🧠 Step 8: After training

Now when the network sees **new people**:

- It no longer guesses randomly
    
- It recognizes patterns it learned
    

Example:

```
Age: 28
Visits: 2
→ Output: 0.91 (very likely to buy)
```

---

# 🔁 Full Loop Summary (Simple View)

```
INPUT →
  weighted sums →
    activation →
      prediction →
        error →
          adjust weights →
            repeat
```

---
