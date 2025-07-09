# Simple Neural Network Model

## In the code you will find comments to help with understanding

A simple feedforward neural network built with PyTorch to classify Iris flower species based on petal and sepal measurements.

---

## 🚀 Overview

This project demonstrates how to:

- Load and preprocess the Iris dataset using Pandas and Scikit-learn
- Define a custom neural network architecture using `torch.nn`
- Train the model to classify flower species (Setosa, Versicolor, Virginica)
- Evaluate the model’s accuracy on a test set

---

## 🧠 Model Architecture

```python
Input: 4 features (sepal length, sepal width, petal length, petal width)
Hidden Layer 1: 8 neurons
Hidden Layer 2: 9 neurons
Output Layer: 3 neurons (for 3 flower classes)
Activation: ReLU
Loss Function: CrossEntropyLoss
Optimizer: Adam
```
