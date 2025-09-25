#!/usr/bin/env python3
"""
Neural Network Implementations at Different Abstraction Levels
Demonstrating that all implementations produce identical results
"""

import numpy as np
from functools import reduce
import sys

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# IMPLEMENTATION 1: True APL One-Liner (Python translation)
# =============================================================================
# APL: train←{⍵-0.01×{⊃⌽{(⊂⍵,⍺),⍨(⊃⍺)(-∘(×∘⍉)¨(⊃⌽⍺))¨⍵}⍣(≢⍵)⊂(((⊃⌽A)(-∘×)(⊢-Y))×(⊃⌽A)×1-⊃⌽A),A←{⍵,⊂σ(⊃⌽⍵)+.×⍺}/⍺,⊂X}⍣1000⊢⍺}
train1=lambda X,Y,ws,α=0.01,n=100:(lambda σ:[(lambda A,δ:[((w:=ws[i][0],b:=ws[i][1]),ws.__setitem__(i,(w-α*A[i].T@δ,b-α*δ.sum(0))),i and(δ:=(δ@w.T)*A[i]*(1-A[i])))for i in range(len(ws)-1,-1,-1)])(reduce(lambda A,wb:A+[σ(A[-1]@wb[0]+wb[1])],ws,[X]),(reduce(lambda A,wb:A+[σ(A[-1]@wb[0]+wb[1])],ws,[X])[-1]-Y)*reduce(lambda A,wb:A+[σ(A[-1]@wb[0]+wb[1])],ws,[X])[-1]*(1-reduce(lambda A,wb:A+[σ(A[-1]@wb[0]+wb[1])],ws,[X])[-1]))for _ in range(n)]and ws)(lambda x:1/(1+np.exp(-x)))

# =============================================================================
# IMPLEMENTATION 2: Ultra-Short (2-3 lines)
# =============================================================================
σ=lambda x:1/(1+np.exp(-x))
def train2(X,Y,ws,α=.01,n=100):
 for _ in[0]*n:A=[X];[A.append(σ(A[-1]@w+b))for w,b in ws];δ=(A[-1]-Y)*A[-1]*(1-A[-1]);[((w:=ws[i][0],b:=ws[i][1]),ws.__setitem__(i,(w-α*A[i].T@δ,b-α*δ.sum(0))),i and(δ:=(δ@w.T)*A[i]*(1-A[i])))for i in range(len(ws)-1,-1,-1)]
 return ws

# Alternative APL-Style (more readable but still functional)
def train2_alt(X, Y, ws, α=0.01, epochs=100):
    σ = lambda x: 1/(1+np.exp(-x))
    for _ in range(epochs):
        A = [X]
        for w, b in ws: A.append(σ(A[-1] @ w + b))
        δ = (A[-1] - Y) * A[-1] * (1 - A[-1])
        for i in range(len(ws)-1, -1, -1):
            w, b = ws[i]
            ws[i] = (w - α * A[i].T @ δ, b - α * δ.sum(0))
            if i: δ = (δ @ w.T) * A[i] * (1 - A[i])
    return ws

# =============================================================================
# IMPLEMENTATION 3: NumPy APL Implementation (Matrix-focused)
# =============================================================================
def train3(X, Y, weights, lr=0.01, epochs=100):
    def sigmoid(x): return 1 / (1 + np.exp(-x))

    for _ in range(epochs):
        # Forward pass with activation collection
        acts = [X]
        for W, b in weights:
            acts.append(sigmoid(acts[-1] @ W + b))

        # Backward pass with gradient updates
        grad = (acts[-1] - Y) * acts[-1] * (1 - acts[-1])
        for i in reversed(range(len(weights))):
            W, b = weights[i]
            weights[i] = (W - lr * acts[i].T @ grad,
                         b - lr * grad.sum(0))
            if i > 0:
                grad = (grad @ W.T) * acts[i] * (1 - acts[i])

    return weights

# =============================================================================
# IMPLEMENTATION 4: Ultra-Readable Implementation
# =============================================================================
def train4(training_inputs, training_outputs, network_weights,
          learning_rate=0.01, num_epochs=100):
    """
    Trains a neural network using backpropagation.
    Clear variable names and explicit steps for maximum readability.
    """

    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    for epoch in range(num_epochs):
        # === Forward Propagation ===
        layer_activations = [training_inputs]
        current_input = training_inputs

        for weight_matrix, bias_vector in network_weights:
            linear_combination = current_input @ weight_matrix + bias_vector
            activated_output = sigmoid(linear_combination)
            layer_activations.append(activated_output)
            current_input = activated_output

        # === Calculate Output Error ===
        # Using same formula as other implementations
        error_gradient = (layer_activations[-1] - training_outputs) * layer_activations[-1] * (1 - layer_activations[-1])

        # === Backward Propagation ===
        for layer_index in range(len(network_weights) - 1, -1, -1):
            weight_matrix, bias_vector = network_weights[layer_index]
            previous_activation = layer_activations[layer_index]

            # Update weights and biases
            network_weights[layer_index] = (
                weight_matrix - learning_rate * previous_activation.T @ error_gradient,
                bias_vector - learning_rate * error_gradient.sum(0)
            )

            # Propagate error to previous layer
            if layer_index > 0:
                error_gradient = (error_gradient @ weight_matrix.T) * layer_activations[layer_index] * (1 - layer_activations[layer_index])

    return network_weights

# =============================================================================
# IMPLEMENTATION 5: English Algorithm (Manual Calculation)
# =============================================================================
def train5_english(X, Y, weights, learning_rate=0.01, num_epochs=100):
    """
    This is the English algorithm translated to Python for verification.
    We'll demonstrate hand calculations for the first iteration.
    """
    print("\n" + "="*80)
    print("ENGLISH ALGORITHM - HAND CALCULATION DEMONSTRATION")
    print("="*80)

    # We'll do hand calculations for the first epoch only
    print("\nInitial Setup:")
    print(f"  Input shape: {X.shape}")
    print(f"  Target shape: {Y.shape}")
    print(f"  Number of layers: {len(weights)}")
    print(f"  Learning rate: {learning_rate}")

    for epoch in range(num_epochs):
        show_calc = (epoch == 0)  # Show calculations only for first epoch

        if show_calc:
            print("\n" + "-"*40)
            print("EPOCH 1 - DETAILED HAND CALCULATION")
            print("-"*40)
            print("\nStep 1: FORWARD PASS")
            print("--------------------")

        # FORWARD PASS
        activations = [X]

        if show_calc:
            print(f"\nLayer 0 (Input):")
            print(f"  Activation shape: {X.shape}")
            print(f"  Sample values: {X[0,:3]}...")

        for layer_idx, (W, b) in enumerate(weights):
            prev_activation = activations[-1]

            # Linear combination: z = x*W + b
            z = prev_activation @ W + b

            # Activation: a = sigmoid(z)
            a = 1 / (1 + np.exp(-z))
            activations.append(a)

            if show_calc:
                print(f"\nLayer {layer_idx + 1}:")
                print(f"  Previous activation shape: {prev_activation.shape}")
                print(f"  Weight matrix shape: {W.shape}")
                print(f"  Bias vector shape: {b.shape}")
                print(f"  Linear combination (z = prev_act @ W + b):")
                print(f"    z[0,0] = {prev_activation[0,:3]} @ {W[:3,0]} + {b[0]}")
                print(f"    z[0,0] = {z[0,0]:.6f}")
                print(f"  Activation (a = 1/(1 + exp(-z))):")
                print(f"    a[0,0] = 1/(1 + exp(-{z[0,0]:.6f})) = {a[0,0]:.6f}")
                print(f"  Full activation shape: {a.shape}")

        if show_calc:
            print("\n\nStep 2: BACKWARD PASS")
            print("---------------------")

        # BACKWARD PASS
        # Calculate output error and initial gradient
        gradient = (activations[-1] - Y) * activations[-1] * (1 - activations[-1])

        if show_calc:
            output = activations[-1]
            error = output - Y
            print(f"\nOutput Layer Error Calculation:")
            print(f"  Prediction: {output[0,:3]}...")
            print(f"  Target: {Y[0,:3]}...")
            print(f"  Error = Prediction - Target:")
            print(f"    error[0,0] = {output[0,0]:.6f} - {Y[0,0]:.6f} = {error[0,0]:.6f}")
            print(f"  Gradient = Error * Activation * (1 - Activation):")
            print(f"    grad[0,0] = {error[0,0]:.6f} * {output[0,0]:.6f} * {1-output[0,0]:.6f}")
            print(f"    grad[0,0] = {gradient[0,0]:.6f}")

        # Update weights layer by layer (backwards)
        for layer_idx in range(len(weights) - 1, -1, -1):
            W, b = weights[layer_idx]
            prev_activation = activations[layer_idx]

            if show_calc and layer_idx == len(weights) - 1:
                print(f"\nLayer {layer_idx + 1} Weight Update:")
                print(f"  Weight gradient = prev_activation.T @ gradient")
                print(f"    Shape: {prev_activation.T.shape} @ {gradient.shape} = {(prev_activation.T @ gradient).shape}")
                print(f"  Bias gradient = sum(gradient, axis=0)")
                print(f"    Shape: {gradient.sum(0).shape}")
                print(f"  Weight update = learning_rate * weight_gradient")
                print(f"  New weights = old_weights - weight_update")

            # Calculate gradients
            W_grad = prev_activation.T @ gradient
            b_grad = gradient.sum(axis=0)

            # Update weights and biases
            weights[layer_idx] = (W - learning_rate * W_grad,
                                  b - learning_rate * b_grad)

            # Propagate gradient to previous layer (must use old W)
            if layer_idx > 0:
                gradient = (gradient @ W.T) * activations[layer_idx] * (1 - activations[layer_idx])

                if show_calc:
                    print(f"\nBackpropagating gradient to layer {layer_idx}:")
                    print(f"  gradient = (gradient @ W.T) * activation * (1 - activation)")
                    print(f"  New gradient shape: {gradient.shape}")

        if show_calc:
            print("\n" + "="*40)
            print("END OF HAND CALCULATION DEMONSTRATION")
            print("Continuing training for remaining epochs...")
            print("="*40)

    return weights

# =============================================================================
# Helper Functions
# =============================================================================
def forward_pass(X, weights):
    """Forward pass through network to get predictions"""
    sigmoid = lambda x: 1/(1+np.exp(-x))
    activation = X
    for W, b in weights:
        activation = sigmoid(activation @ W + b)
    return activation

def initialize_weights(layer_sizes):
    """Initialize weights for all implementations"""
    weights = []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
        b = np.zeros(layer_sizes[i+1])
        weights.append((W, b))
    return weights

def copy_weights(weights):
    """Deep copy weights for fair comparison"""
    return [(W.copy(), b.copy()) for W, b in weights]

# =============================================================================
# Main Demonstration
# =============================================================================
def main():
    print("="*80)
    print("NEURAL NETWORK IMPLEMENTATIONS AT DIFFERENT ABSTRACTION LEVELS")
    print("Demonstrating that all implementations produce identical results")
    print("="*80)

    # Create simple dataset (XOR problem)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=np.float32)

    Y = np.array([[0],
                  [1],
                  [1],
                  [0]], dtype=np.float32)

    # Network architecture: 2 inputs, 3 hidden, 1 output
    layer_sizes = [2, 3, 1]

    # Initialize weights (same for all implementations)
    initial_weights = initialize_weights(layer_sizes)

    print("\nDataset: XOR Problem")
    print("---------------------")
    print("Inputs (X):")
    print(X)
    print("\nTargets (Y):")
    print(Y)

    print(f"\nNetwork Architecture: {layer_sizes}")
    print(f"Initial weights initialized with seed 42 for reproducibility")

    # Train with each implementation
    print("\n" + "="*80)
    print("TRAINING WITH EACH IMPLEMENTATION")
    print("="*80)

    # Implementation 1: True APL one-liner
    print("\n1. TRUE APL ONE-LINER:")
    print("-" * 30)
    weights1 = copy_weights(initial_weights)
    weights1 = train1(X, Y, weights1, α=0.5, n=100)
    pred1 = forward_pass(X, weights1)
    print(f"Final predictions:\n{pred1}")

    # Implementation 2: Ultra-short
    print("\n2. ULTRA-SHORT (2 lines):")
    print("-" * 30)
    weights2 = copy_weights(initial_weights)
    weights2 = train2(X, Y, weights2, α=0.5, n=100)
    pred2 = forward_pass(X, weights2)
    print(f"Final predictions:\n{pred2}")

    # Implementation 2b: APL-style alternative
    print("\n2b. APL-STYLE (READABLE):")
    print("-" * 30)
    weights2b = copy_weights(initial_weights)
    weights2b = train2_alt(X, Y, weights2b, α=0.5, epochs=100)
    pred2b = forward_pass(X, weights2b)
    print(f"Final predictions:\n{pred2b}")

    # Implementation 3: NumPy APL
    print("\n3. NUMPY APL IMPLEMENTATION:")
    print("-" * 30)
    weights3 = copy_weights(initial_weights)
    weights3 = train3(X, Y, weights3, lr=0.5, epochs=100)
    pred3 = forward_pass(X, weights3)
    print(f"Final predictions:\n{pred3}")

    # Implementation 4: Ultra-readable
    print("\n4. ULTRA-READABLE:")
    print("-" * 30)
    weights4 = copy_weights(initial_weights)
    weights4 = train4(X, Y, weights4, learning_rate=0.5, num_epochs=100)
    pred4 = forward_pass(X, weights4)
    print(f"Final predictions:\n{pred4}")

    # Implementation 5: English with hand calculation
    weights5 = copy_weights(initial_weights)
    weights5 = train5_english(X, Y, weights5, learning_rate=0.5, num_epochs=100)
    pred5 = forward_pass(X, weights5)
    print(f"\n5. ENGLISH ALGORITHM RESULT:")
    print("-" * 30)
    print(f"Final predictions:\n{pred5}")

    # Verify all implementations produce the same result
    print("\n" + "="*80)
    print("VERIFICATION: COMPARING ALL IMPLEMENTATIONS")
    print("="*80)

    implementations = [
        ("True APL one-liner", pred1),
        ("Ultra-short", pred2),
        ("APL-style", pred2b),
        ("NumPy APL", pred3),
        ("Ultra-readable", pred4),
        ("English", pred5)
    ]

    print("\nPrediction comparison (all should be nearly identical):")
    print("-" * 50)
    for name, pred in implementations:
        print(f"{name:15} : {pred.flatten()}")

    # Check if all predictions are close
    all_close = True
    reference = pred1
    for name, pred in implementations[1:]:
        if not np.allclose(reference, pred, atol=1e-4):
            all_close = False
            print(f"WARNING: {name} differs from ultra-short!")
            print(f"  Max difference: {np.max(np.abs(reference - pred))}")

    if all_close:
        print("\n✓ SUCCESS: All implementations produce identical results!")
        print("  Maximum difference: < 1e-4")
    else:
        print("\n✗ FAILURE: Implementations produce different results")

    # Show code length comparison
    print("\n" + "="*80)
    print("CODE LENGTH COMPARISON")
    print("="*80)

    code_lengths = [
        ("True APL one-liner", 1, "Direct APL translation in a single Python lambda"),
        ("Ultra-short", 2, "Extreme compression using lambdas and list comprehensions"),
        ("APL-style", 8, "Compact functional style inspired by APL"),
        ("NumPy APL", 14, "Matrix-focused NumPy implementation"),
        ("Ultra-readable", 35, "Clear variable names and explicit steps"),
        ("English", 50, "Algorithmic description in plain English")
    ]

    for name, lines, description in code_lengths:
        print(f"{name:15} : {lines:3} lines - {description}")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()