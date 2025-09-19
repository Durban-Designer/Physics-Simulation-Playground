"""
A simple two-layer neural network for binary classification, using only NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Generate some data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
np.random.seed(42)
input_neurons = 2
hidden_neurons = 4
output_neurons = 1

wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
w_out = np.random.uniform(size=(hidden_neurons, output_neurons))
b_out = np.random.uniform(size=(1, output_neurons))

learning_rate = 0.1
epochs = 10000
errors = []

for i in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, wh) + bh
    hidden_layer_activation = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_activation, w_out) + b_out
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(w_out.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)

    # Update weights and biases
    w_out += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
    b_out += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden_layer) * learning_rate
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    if i % 100 == 0:
        errors.append(np.mean(np.abs(error)))

print("Training finished.")
print("Final predictions:")
print(predicted_output)

# Plot the error over time
plt.figure(figsize=(10, 6))
plt.plot(errors)
plt.title("Neural Network Training Error")
plt.xlabel("Epochs (x100)")
plt.ylabel("Mean Absolute Error")
plt.grid(True)
plt.savefig("neural_network_training.png")
print("Neural network training plot saved to neural_network_training.png")
