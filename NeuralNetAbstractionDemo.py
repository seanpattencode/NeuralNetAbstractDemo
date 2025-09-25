
pythonimport numpy as np
σ = lambda x: 1/(1+np.exp(-x))
def train(X, Y, ws, α=0.01, epochs=1000):
    for _ in range(epochs):
        A = [X]
        for w, b in ws: A.append(σ(A[-1] @ w + b))
        δ = (A[-1] - Y) * A[-1] * (1 - A[-1])
        for i in range(len(ws)-1, -1, -1):
            ws[i] = (ws[i][0] - α * A[i].T @ δ, ws[i][1] - α * δ.sum(0))
            if i: δ = (δ @ ws[i][0].T) * A[i] * (1 - A[i])
    return ws
2. Ultra-compressed (2 lines):
pythonimport numpy as np;σ=lambda x:1/(1+np.exp(-x))
train=lambda X,Y,ws,α=.01,n=1000:[(A:=[X],[A.append(σ(A[-1]@w+b))for w,b in ws],δ:=(A[-1]-Y)*A[-1]*(1-A[-1]),[ws.__setitem__(i,(ws[i][0]-α*A[i].T@δ,ws[i][1]-α*δ.sum(0)))or(i and setattr('δ',δ,(δ@ws[i][0].T)*A[i]*(1-A[i])))for i in range(len(ws)-1,-1,-1)])for _ in[0]*n]and ws
3. NumPy concise readable (8 lines):
pythonimport numpy as np

def neural_net(X, Y, layers, lr=0.01, epochs=1000):
    σ = lambda x: 1/(1+np.exp(-x))
    weights = [(np.random.randn(i,o)/i**.5, np.zeros(o)) for i,o in zip(layers[:-1], layers[1:])]
    
    for _ in range(epochs):
        # Forward: collect activations
        acts = [X]
        for W, b in weights:
            acts.append(σ(acts[-1] @ W + b))
        
        # Backward: gradient descent
        grad = (acts[-1] - Y) * acts[-1] * (1-acts[-1])
        for i in reversed(range(len(weights))):
            weights[i] = (weights[i][0] - lr * acts[i].T @ grad,
                         weights[i][1] - lr * grad.sum(0))
            if i > 0:
                grad = (grad @ weights[i][0].T) * acts[i] * (1-acts[i])
    
    return lambda x: σ(x @ weights[0][0] + weights[0][1]) if len(weights)==1 else neural_net.forward(x, weights)
4. Ultra readable (20+ lines):
pythonimport numpy as np

def train_neural_network(training_inputs, training_outputs, layer_sizes, 
                         learning_rate=0.01, num_epochs=1000):
    """
    Trains a neural network using backpropagation.
    
    Args:
        training_inputs: Input data (samples × features)
        training_outputs: Target outputs (samples × classes)
        layer_sizes: List of neurons per layer [input, hidden..., output]
        learning_rate: Step size for weight updates
        num_epochs: Number of training iterations
    """
    
    # Initialize weights and biases randomly
    network_weights = []
    for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        weight_matrix = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        bias_vector = np.zeros(output_size)
        network_weights.append((weight_matrix, bias_vector))
    
    # Sigmoid activation function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(activated_output):
        return activated_output * (1 - activated_output)
    
    # Training loop
    for epoch in range(num_epochs):
        
        # Forward propagation: compute all layer activations
        layer_activations = [training_inputs]
        current_input = training_inputs
        
        for weight_matrix, bias_vector in network_weights:
            linear_combination = current_input @ weight_matrix + bias_vector
            activated_output = sigmoid(linear_combination)
            layer_activations.append(activated_output)
            current_input = activated_output
        
        # Calculate output error
        output_error = layer_activations[-1] - training_outputs
        error_gradient = output_error * sigmoid_derivative(layer_activations[-1])
        
        # Backward propagation: update weights layer by layer
        for layer_index in reversed(range(len(network_weights))):
            weight_matrix, bias_vector = network_weights[layer_index]
            previous_activation = layer_activations[layer_index]
            
            # Calculate weight and bias updates
            weight_update = learning_rate * previous_activation.T @ error_gradient
            bias_update = learning_rate * error_gradient.sum(axis=0)
            
            # Update this layer's parameters
            new_weights = weight_matrix - weight_update
            new_biases = bias_vector - bias_update
            network_weights[layer_index] = (new_weights, new_biases)
            
            # Propagate error to previous layer (if not input layer)
            if layer_index > 0:
                error_gradient = (error_gradient @ weight_matrix.T) * \
                                sigmoid_derivative(layer_activations[layer_index])
    
    return network_weights             







Python

import numpy as np; from functools import reduce; σ=lambda x:1/(1+np.exp(-x)); fit=lambda X,Y,ws,α=.01,n=1000:reduce(lambda w, _:(A:=[X],[A.append(σ(A[-1]@i[0]+i[1]))for i in w],δ:=(A[-1]-Y)*A[-1]*(1-A[-1]),[i and(δ:=(δ@w[i][0].T)*A[i]*(1-A[i]))or 1 for i in range(len(w)-1,-1,-1)],[(w[i][0]-α*A[i].T@δ,w[i][1]-α*δ.sum(0))for i in range(len(w))])[4], [0]*n, ws)
2. Original Fixed and Runnable
This is the user-provided code, corrected for a typo (Σ changed to σ) and with necessary imports added.

Python

import numpy as np
from functools import reduce

σ = lambda x: 1/(1+np.exp(-x))
fw = lambda X, ws: reduce(lambda x, w: σ(x@w[0]+w[1]), ws, X)

def fit(X, Y, ws, α=.01, n=1000):
 for _ in [0]*n: A=[X]; [A.append(σ(A[-1]@w+b)) for w,b in ws]; δ=(A[-1]-Y)*A[-1]*(1-A[-1]); [(ws.__setitem__(i,(ws[i][0]-α*A[i].T@δ,ws[i][1]-α*δ.sum(0))), i and(δ:=(δ@ws[i][0].T)*A[i]*(1-A[i]))) for i in range(len(ws)-1,-1,-1)]
 return ws
3. Concise but Readable Python
This version uses standard Python loops and meaningful variable names. It's what a proficient programmer might write for a personal project.

Python

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fit(X, Y, weights_biases, learning_rate=0.01, epochs=1000):
    for _ in range(epochs):
        # Forward pass
        activations = [X]
        for w, b in weights_biases:
            activations.append(sigmoid(activations[-1] @ w + b))

        # Backward pass
        # Calculate output layer error delta
        error = activations[-1] - Y
        delta = error * activations[-1] * (1 - activations[-1])

        # Propagate error and update weights
        for i in range(len(weights_biases) - 1, -1, -1):
            w, b = weights_biases[i]
            
            # Update weights and biases
            prev_activations = activations[i].T
            weights_biases[i] = (w - learning_rate * prev_activations @ delta,
                                 b - learning_rate * delta.sum(axis=0))

            # Propagate delta to the previous layer
            if i > 0:
                delta = (delta @ w.T) * activations[i] * (1 - activations[i])
                
    return weights_biases
4. Ultra-Readable Python (Object-Oriented)
This version uses a class to encapsulate the network's logic, with highly descriptive variable names and comments, making it suitable for teaching or inclusion in a larger project.

Python

import numpy as np

class NeuralNetwork:
    """A simple, highly-readable neural network implementation."""
    def __init__(self, layer_sizes):
        # Note: Weights are tuples of (weight_matrix, bias_vector)
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1])
            b = np.random.randn(1, layer_sizes[i+1])
            self.weights.append((w, b))

    def _sigmoid(self, x):
        """Activation function."""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, sigmoid_output):
        """Derivative of the sigmoid function."""
        return sigmoid_output * (1 - sigmoid_output)

    def fit(self, X, Y, learning_rate=0.01, epochs=1000):
        """Train the neural network using backpropagation."""
        for epoch in range(epochs):
            # === 1. Forward Pass ===
            # Store all activations, starting with the input layer
            layer_activations = [X]
            for w, b in self.weights:
                z = layer_activations[-1] @ w + b
                activation = self._sigmoid(z)
                layer_activations.append(activation)

            # === 2. Backward Pass ===
            # Start by calculating the error for the final output layer
            prediction = layer_activations[-1]
            output_error = prediction - Y
            output_delta = output_error * self._sigmoid_derivative(prediction)
            
            deltas = [output_delta]

            # Propagate the error backward through the hidden layers
            for i in range(len(self.weights) - 2, -1, -1):
                # Get the delta from the next layer (closer to the output)
                forward_delta = deltas[-1]
                # Get the weights from the next layer
                forward_weights = self.weights[i+1][0].T
                
                hidden_error = forward_delta @ forward_weights
                hidden_delta = hidden_error * self._sigmoid_derivative(layer_activations[i+1])
                deltas.append(hidden_delta)
            
            # Reverse deltas to match the order of the layers
            deltas.reverse()

            # === 3. Update Weights and Biases ===
            for i in range(len(self.weights)):
                w, b = self.weights[i]
                input_to_layer = layer_activations[i].T
                delta_for_layer = deltas[i]

                # Calculate gradients
                weight_gradient = input_to_layer @ delta_for_layer
                bias_gradient = np.sum(delta_for_layer, axis=0, keepdims=True)

                # Update using gradient descent
                w -= learning_rate * weight_gradient
                b -= learning_rate * bias_gradient
                self.weights[i] = (w, b)
5. Plain English Procedural Description
This describes the algorithm as if English were a programming language.

Program Start

Required Items:

Input Data: A table of initial numbers.

Target Outputs: A table of correct final numbers.

Network Layers: A list describing the weights and biases for each layer.

Learning Rate: A small number that controls how fast we adjust.

Repetitions: The number of times we will repeat the training cycle.

Procedure: Train The Network

Repeat the following main loop for the specified number of Repetitions:

Part A: The Forward Pass (Make a Prediction)

Create a list called Layer Activations. Add the Input Data to it as the first item.

For each Layer in Network Layers, from the first to the last:
a. Take the most recent activation from the Layer Activations list.
b. Multiply it by the current Layer's weights.
c. Add the current Layer's biases.
d. Apply the sigmoid function (1 divided by 1 plus e to the power of the negative result) to this sum.
e. Add the final result to the end of the Layer Activations list.

Part B: The Backward Pass (Learn From The Error)

Get the very last item from Layer Activations. This is the network's Prediction.

Calculate the Initial Error by subtracting the Target Outputs from the Prediction.

Calculate the Current Delta: Multiply the Initial Error by the Prediction and also by (1 minus the Prediction).

Now, for each Layer in Network Layers, starting from the last layer and working backward to the first:
a. Get the activation of the layer that fed into the current Layer.
b. Calculate the Weight Change: Multiply the transpose of the feeder-layer's activation by the Current Delta.
c. Calculate the Bias Change: Sum up all the rows of the Current Delta.
d. Update the current Layer's weights: Subtract the (Weight Change multiplied by the Learning Rate) from the current weights.
e. Update the current Layer's biases: Subtract the (Bias Change multiplied by the Learning Rate) from the current biases.
f. If this is not the very first layer, you must calculate the Current Delta for the next step of the loop (the previous layer):
i. Multiply the Current Delta by the transpose of the current Layer's weights.
ii. Multiply that result by the feeder-layer's activation.
iii. Multiply that result by (1 minus the feeder-layer's activation).
iv. This becomes the new Current Delta.

After all Repetitions are complete, the Network Layers are now trained.

Program End


Version 1: Ultra-Short (Pushing Limits)
python
import numpy as np
from functools import reduce
σ=lambda x:1/(1+np.exp(-x));f=lambda X,w:reduce(lambda x,w:σ(x@w[0]+w[1]),w,X)
def t(X,Y,w,a=.01,n=1000):
 for _ in[0]*n:A=[X];[A.append(σ(A[-1]@w+b))for w,b in w];d=(A[-1]-Y)*A[-1]*(1-A[-1]);[w.__setitem__(i,(w[i][0]-a*A[i].T@d,w[i][1]-a*d.sum(0)))or i and(d:=(d@w[i][0].T)*A[i]*(1-A[i]))for i in range(len(w)-1,-1,-1)]
 return w
Reduction: 5 lines → 3 lines by combining statements and using or for sequencing

Version 2: Original (Fixed)
python
import numpy as np
from functools import reduce

σ = lambda x: 1/(1+np.exp(-x))
forward = lambda X, weights: reduce(lambda x, w: σ(x @ w[0] + w[1]), weights, X)

def train(X, Y, weights, α=0.01, iterations=1000):
    for _ in [0] * iterations:
        # Forward pass: store all layer activations
        activations = [X]
        [activations.append(σ(activations[-1] @ w + b)) for w, b in weights]
        
        # Output error * sigmoid derivative
        δ = (activations[-1] - Y) * activations[-1] * (1 - activations[-1])
        
        # Backward pass through layers
        [(weights.__setitem__(i, (weights[i][0] - α * activations[i].T @ δ, 
                                 weights[i][1] - α * δ.sum(0))),
          i and (δ := (δ @ weights[i][0].T) * activations[i] * (1 - activations[i])))
         for i in range(len(weights)-1, -1, -1)]
    
    return weights
Fixed: Added proper imports and fixed variable naming

Version 3: Concise NumPy
python
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))

def forward_pass(X, layers):
    a = X
    for weights, biases in layers:
        a = sigmoid(a @ weights + biases)
    return a

def train_network(X, Y, layers, lr=0.01, epochs=1000):
    for _ in range(epochs):
        # Forward pass storing all activations
        activations = [X]
        for weights, biases in layers:
            activations.append(sigmoid(activations[-1] @ weights + biases))
        
        # Backward pass
        δ = (activations[-1] - Y) * activations[-1] * (1 - activations[-1])
        
        for i in reversed(range(len(layers))):
            # Update current layer weights/biases
            layers[i] = (
                layers[i][0] - lr * activations[i].T @ δ,
                layers[i][1] - lr * δ.sum(axis=0)
            )
            # Propagate error backward if not input layer
            if i > 0:
                δ = (δ @ layers[i][0].T) * activations[i] * (1 - activations[i])
    
    return layers
Version 4: Ultra-Readable
python
import numpy as np
from typing import List, Tuple

def sigmoid_activation(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation function element-wise"""
    return 1.0 / (1.0 + np.exp(-x))

def neural_network_forward(input_data: np.ndarray, 
                          layers: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Perform forward pass through the neural network
    
    Args:
        input_data: Input features (samples × features)
        layers: List of (weights, biases) for each layer
    
    Returns:
        Final layer outputs
    """
    current_activation = input_data
    for layer_weights, layer_biases in layers:
        # Linear transformation: input × weights + biases
        linear_output = current_activation @ layer_weights + layer_biases
        # Non-linear activation
        current_activation = sigmoid_activation(linear_output)
    return current_activation

def train_neural_network(training_inputs: np.ndarray,
                        target_outputs: np.ndarray,
                        layers: List[Tuple[np.ndarray, np.ndarray]],
                        learning_rate: float = 0.01,
                        training_epochs: int = 1000) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Train neural network using backpropagation and gradient descent
    
    Args:
        training_inputs: Input data matrix
        target_outputs: Target output matrix  
        layers: Initial network layers (weights, biases)
        learning_rate: Step size for gradient updates
        training_epochs: Number of training iterations
    
    Returns:
        Trained network layers
    """
    
    for epoch in range(training_epochs):
        # FORWARD PASS: Compute activations for all layers
        layer_activations = [training_inputs]  # Store all activations starting with input
        
        for layer_weights, layer_biases in layers:
            # Compute this layer's activation from previous layer's activation
            previous_activation = layer_activations[-1]
            linear_combination = previous_activation @ layer_weights + layer_biases
            current_activation = sigmoid_activation(linear_combination)
            layer_activations.append(current_activation)
        
        # BACKWARD PASS: Compute errors and update weights
        network_output = layer_activations[-1]
        
        # Calculate output layer error (derivative of loss * activation derivative)
        output_error = (network_output - target_outputs) * network_output * (1 - network_output)
        current_error = output_error
        
        # Process layers in reverse order (output to input)
        for layer_index in range(len(layers) - 1, -1, -1):
            current_activation = layer_activations[layer_index]
            
            # Calculate weight gradients and update weights
            weight_gradient = current_activation.T @ current_error
            bias_gradient = current_error.sum(axis=0)
            
            # Update current layer weights and biases
            old_weights, old_biases = layers[layer_index]
            new_weights = old_weights - learning_rate * weight_gradient
            new_biases = old_biases - learning_rate * bias_gradient
            layers[layer_index] = (new_weights, new_biases)
            
            # Propagate error to previous layer (if not input layer)
            if layer_index > 0:
                # Error for previous layer = current error × weights^T × activation derivative
                previous_layer_activation = layer_activations[layer_index]
                error_backpropagated = current_error @ old_weights.T
                activation_derivative = previous_layer_activation * (1 - previous_layer_activation)
                current_error = error_backpropagated * activation_derivative
    
    return layers
Version 5: English as Programming Language
text
PROGRAM: Neural Network Training

IMPORTS:
- Numerical computation library
- Function reduction utility

DEFINE:
- Sigmoid function: 1 divided by (1 plus exponential of negative input)

FUNCTION forward_pass(input_data, network_layers):
    START with input_data as current_value
    FOR EACH layer in network_layers:
        APPLY: current_value = sigmoid(current_value × layer_weights + layer_biases)
    RETURN final current_value

FUNCTION train_network(inputs, targets, initial_layers, learning_rate=0.01, iterations=1000):
    
    REPEAT for specified number of iterations:
        
        // FORWARD PASS: Compute all layer activations
        LET activations = list containing [inputs]
        FOR EACH (weights, biases) in initial_layers:
            CALCULATE: new_activation = sigmoid(last_activation × weights + biases)
            APPEND new_activation to activations list
        
        // BACKWARD PASS: Update weights using gradient descent
        CALCULATE output_error = (final_activation - targets) × final_activation × (1 - final_activation)
        SET current_error = output_error
        
        // Process layers in reverse order (output to input)
        FOR layer_index FROM last_layer DOWN TO first_layer:
            
            // Update current layer weights and biases
            CALCULATE weight_update = learning_rate × (previous_activation^T × current_error)
            CALCULATE bias_update = learning_rate × sum(current_error across samples)
            
            UPDATE layer_weights = old_weights - weight_update
            UPDATE layer_biases = old_biases - bias_update
            
            // If not the first layer, propagate error backward
            IF layer_index > 0:
                CALCULATE: current_error = (current_error × weights^T) × previous_activation × (1 - previous_activation)
    
    RETURN updated network_layers

END PROGRAM