# MNIST Neural Network From Scratch

A simple neural network built to recognize handwritten digits from the MNIST dataset.

This project was completed as part of a **Programming & Computing Society (PCS) Neural Network Workshop**. The goal of the project was to understand the fundamentals behind neural networks by implementing core components manually instead of relying entirely on PyTorch's pre-built neural network layers.

## Overview

The neural network takes a 28×28 pixel image of a handwritten digit and predicts which digit from **0–9** is shown.

Rather than using PyTorch's built-in `nn.Linear` layers, this project implements custom linear layers, ReLU activation, forward propagation, and parameter updates to better understand what happens during neural network training.

## Network Architecture

The model uses three fully connected linear layers:

```text
Input
  │
  ▼
784
  │
  ▼
Linear Layer
  │
  ▼
128
  │
  ▼
ReLU
  │
  ▼
Linear Layer
  │
  ▼
128
  │
  ▼
ReLU
  │
  ▼
Linear Layer
  │
  ▼
10 Outputs
  │
  ▼
Predicted Digit (0–9)
```

### Layer Sizes

- **784 inputs** — one value for each pixel in a 28×28 MNIST image
- **128 neurons** — first hidden layer
- **128 neurons** — second hidden layer
- **10 outputs** — one output for each digit from 0 through 9

## Features

- Custom `Linear` layer
- Custom `ReLU` activation
- Three-layer neural network
- Forward propagation
- Cross-entropy loss
- Backpropagation with PyTorch autograd
- Manual gradient descent
- MNIST dataset loading and preprocessing
- Training and testing
- Accuracy evaluation

## How It Works

### 1. Linear Layers

Each custom linear layer contains its own weights and biases.

The forward pass performs:

```python
x @ weights + bias
```

The weights are initialized randomly and gradients are enabled so they can be updated during training.

### 2. ReLU Activation

The network uses the ReLU activation function between linear layers:

```python
x.clamp(min=0)
```

This keeps positive values while replacing negative values with zero.

### 3. MNIST Data

The project uses the MNIST handwritten-digit dataset provided through `torchvision`.

Each 28×28 image is flattened into a vector containing **784 values** and normalized by dividing the pixel values by `255.0`.

### 4. Forward Pass

The input travels through the network in the following order:

```text
Linear → ReLU → Linear → ReLU → Linear
```

The final layer produces 10 scores corresponding to the digits **0–9**.

### 5. Loss

The network uses cross-entropy loss:

```python
loss = torch.nn.functional.cross_entropy(prediction, target)
```

This measures how far the network's predictions are from the correct labels.

### 6. Backpropagation

PyTorch's autograd system calculates the gradients with:

```python
loss.backward()
```

These gradients show how the model's parameters should change to reduce the loss.

### 7. Gradient Descent

Instead of using a built-in PyTorch optimizer, the parameters are updated manually:

```python
with torch.no_grad():
    for parameter in model.parameters():
        parameter -= learning_rate * parameter.grad
```

After the update, the gradients are cleared before the next training iteration:

```python
for parameter in model.parameters():
    parameter.grad.zero_()
```

## Training Configuration

The current model uses:

```text
Input Size:     784
Hidden Size:    128
Output Size:    10
Learning Rate:  0.1
Epochs:         10
```

During training, the program reports the loss and test accuracy to show how the model is performing.

## Technologies Used

- Python
- PyTorch
- Torchvision
- MNIST

## What I Learned

Through this project, I gained a better understanding of:

- Neural network architecture
- Weights and biases
- Matrix multiplication in neural networks
- Linear layers
- Activation functions
- Forward propagation
- Loss functions
- Backpropagation
- Gradients
- Gradient descent
- Training and evaluating a machine learning model
- How PyTorch autograd assists with neural network training

## Project File

The main implementation can be found in:

```text
mnist_network.py
```

## Acknowledgment

This project was completed through a **Programming & Computing Society (PCS) Neural Network Workshop** as a hands-on introduction to neural networks and machine learning.

The workshop provided the structure and learning material for the project, while I used the exercise to develop a stronger understanding of how neural networks operate internally.
