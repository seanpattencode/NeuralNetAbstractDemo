#!/usr/bin/env python3
"""Debug script to find the difference between implementations"""

import numpy as np

np.random.seed(42)

# Create simple dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Initialize weights
layer_sizes = [2, 3, 1]
weights = []
for i in range(len(layer_sizes) - 1):
    W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
    b = np.zeros(layer_sizes[i+1])
    weights.append((W, b))

# Save initial weights
initial_w1 = [(W.copy(), b.copy()) for W, b in weights]
initial_w2 = [(W.copy(), b.copy()) for W, b in weights]

σ = lambda x: 1/(1+np.exp(-x))

# Implementation 1 (simplified from train2)
def impl1(X, Y, ws, α=0.5, epochs=1):
    for _ in range(epochs):
        A = [X]
        for w, b in ws:
            A.append(σ(A[-1] @ w + b))
        δ = (A[-1] - Y) * A[-1] * (1 - A[-1])
        for i in range(len(ws)-1, -1, -1):
            ws[i] = (ws[i][0] - α * A[i].T @ δ, ws[i][1] - α * δ.sum(0))
            if i:
                δ = (δ @ ws[i][0].T) * A[i] * (1 - A[i])
    return ws

# Implementation 2 (simplified from train4)
def impl2(X, Y, ws, α=0.5, epochs=1):
    for _ in range(epochs):
        A = [X]
        for w, b in ws:
            A.append(σ(A[-1] @ w + b))
        δ = (A[-1] - Y) * A[-1] * (1 - A[-1])
        for i in range(len(ws) - 1, -1, -1):
            w, b = ws[i]
            ws[i] = (w - α * A[i].T @ δ, b - α * δ.sum(0))
            if i > 0:
                δ = (δ @ w.T) * A[i] * (1 - A[i])
    return ws

# Run one epoch with each
print("Testing one epoch...")
ws1 = impl1(X, Y, initial_w1, α=0.5, epochs=1)
ws2 = impl2(X, Y, initial_w2, α=0.5, epochs=1)

print("\nWeights after 1 epoch:")
print("Implementation 1 - Layer 0 weights[0,0]:", ws1[0][0][0,0])
print("Implementation 2 - Layer 0 weights[0,0]:", ws2[0][0][0,0])
print("Difference:", ws1[0][0][0,0] - ws2[0][0][0,0])

print("\nImplementation 1 - Layer 1 weights[0,0]:", ws1[1][0][0,0])
print("Implementation 2 - Layer 1 weights[0,0]:", ws2[1][0][0,0])
print("Difference:", ws1[1][0][0,0] - ws2[1][0][0,0])

# Now check where the difference comes from
print("\n" + "="*50)
print("Detailed trace...")

# Reset
ws1 = [(W.copy(), b.copy()) for W, b in weights]
ws2 = [(W.copy(), b.copy()) for W, b in weights]

# Forward pass (should be identical)
A1 = [X]
A2 = [X]
for w, b in ws1:
    A1.append(σ(A1[-1] @ w + b))
for w, b in ws2:
    A2.append(σ(A2[-1] @ w + b))

print("Forward pass outputs match?", np.allclose(A1[-1], A2[-1]))

# Initial gradient (should be identical)
δ1 = (A1[-1] - Y) * A1[-1] * (1 - A1[-1])
δ2 = (A2[-1] - Y) * A2[-1] * (1 - A2[-1])
print("Initial gradients match?", np.allclose(δ1, δ2))

# Layer 1 update (output layer)
i = 1
print(f"\nUpdating layer {i}...")
print("δ shape:", δ1.shape)
print("A[i] shape:", A1[i].shape)

# Impl 1 update
old_w1 = ws1[i][0].copy()
ws1[i] = (ws1[i][0] - 0.5 * A1[i].T @ δ1, ws1[i][1] - 0.5 * δ1.sum(0))
print(f"Impl1: Using OLD weights for backprop? (ws[{i}][0] is old_w1)", np.array_equal(ws1[i][0], old_w1))

# Impl 2 update
w2, b2 = ws2[i]
old_w2 = w2.copy()
ws2[i] = (w2 - 0.5 * A2[i].T @ δ2, b2 - 0.5 * δ2.sum(0))
print(f"Impl2: Using OLD weights for backprop? (w2 is old_w2)", np.array_equal(w2, old_w2))

# Propagate gradient
if i > 0:
    δ1_new = (δ1 @ ws1[i][0].T) * A1[i] * (1 - A1[i])  # Uses UPDATED weights!
    δ2_new = (δ2 @ w2.T) * A2[i] * (1 - A2[i])  # Uses OLD weights!
    print(f"\nPropagated gradients match?", np.allclose(δ1_new, δ2_new))
    print(f"Difference in propagated gradients:", np.max(np.abs(δ1_new - δ2_new)))