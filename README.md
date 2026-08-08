# Linear Regression from Scratch

A multiple linear regression model implemented from scratch in NumPy — no `scikit-learn`, no shortcuts. This project was built to deepen my understanding of vectorization, gradient descent, and NumPy internals, and to practice writing clear technical documentation.

## Overview

The model is trained using batch gradient descent to minimize mean squared error (MSE), with all core operations — the prediction function, loss, and gradients — implemented manually using vectorized NumPy operations rather than explicit loops.

## Repository Structure

| File | Description |
|---|---|
| `house_prices.csv` | Dataset from [hugg](https://huggingface.co/datasets/t22000t/house-prices-tabular) |
| `linearreg.py` | Core library: prediction, loss, gradient computation, gradient descent optimizer, and z-score normalization utilities, each documented with docstrings. |
| `test1.ipynb` | First training run and result visualization (`steps=1e5`, `learning_rate=3e-5`). |
| `test2.ipynb` | Second iteration — *in progress*. |

## Features Implemented

- **Vectorized prediction and cost functions** — no per-sample loops
- **Batch gradient descent** with configurable learning rate and step count
- **Z-score normalization** (with inverse transform) for stable, faster convergence
- **Training diagnostics** — loss and weight trajectories tracked and visualized across iterations

## Example Usage

```python
from linearreg import zscore_normalize, optimize_parameters, f

# Normalize features and target
X_norm, X_mean, X_std = zscore_normalize(X_train)
y_norm, y_mean, y_std = zscore_normalize(y_train)

# Train
w, b, J_history, w_history, b_history = optimize_parameters(
    X_norm, y_norm.ravel(),
    w_init=np.zeros(X_norm.shape[1]),
    b_init=0.0,
    steps=100_000,
    alpha=3e-5
)

# Predict
y_pred = f(X_norm, w, b)
```

## Motivation

This project was built as a learning exercise, focused on:
- Understanding gradient descent mechanics beyond the level of a library call
- Practicing NumPy vectorization and array broadcasting
- Building comfort with Matplotlib for training diagnostics
- Writing clear, maintainable documentation

## Status

Actively in progress — `test2.ipynb` extends the initial experiments with further training diagnostics and refinements.

## Result section

In progress
