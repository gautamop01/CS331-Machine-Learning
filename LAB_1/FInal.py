import numpy as np

# Randomly generated data
np.random.seed(100)
input_data = np.random.rand(1, 3)  # One sample with three features
target_output = np.random.rand(1, 1)  # Target output for demonstration

# Initialize weights and biases
weights_input_hidden = np.random.rand(3, 2)  # Weights connecting input to hidden layer
bias_hidden = np.random.rand(1, 2)  # Bias for the hidden layer

weights_hidden_output = np.random.rand(2, 1)  # Weights connecting hidden to output layer
bias_output = np.random.rand(1, 1)  # Bias for the output layer

# Forward pass
hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
hidden_layer_output = hidden_layer_input  # No sigmoid activation function

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
predicted_output = output_layer_input  # No sigmoid activation function

# Calculate the forward pass loss
forward_pass_loss = 0.5 * np.sum((predicted_output - target_output) ** 2)

# Print the forward pass loss
print("Forward Pass Loss:", forward_pass_loss)

# Backward pass (Gradient Descent)
output_error = predicted_output - target_output
output_gradient = hidden_layer_output.T.dot(output_error)
bias_output_gradient = np.sum(output_error, axis=0, keepdims=True)

hidden_error = output_error.dot(weights_hidden_output.T) * hidden_layer_output
hidden_gradient = input_data.T.dot(hidden_error)
bias_hidden_gradient = np.sum(hidden_error, axis=0, keepdims=True)

# Update weights and biases using gradient descent (you can do this as part of the training loop)
learning_rate = 0.01
weights_hidden_output -= learning_rate * output_gradient
bias_output -= learning_rate * bias_output_gradient

weights_input_hidden -= learning_rate * hidden_gradient
bias_hidden -= learning_rate * bias_hidden_gradient

# Forward pass after one iteration
hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
hidden_layer_output = hidden_layer_input  # No sigmoid activation function

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
predicted_output = output_layer_input  # No sigmoid activation function

# Calculate the backward pass loss
backward_pass_loss = 0.5 * np.sum((predicted_output - target_output) ** 2)

# Print the backward pass loss
print("Backward Pass Loss:", backward_pass_loss)

# Compare forward pass loss and backward pass loss
print("Comparison: Forward Pass Loss == Backward Pass Loss?", np.allclose(forward_pass_loss, backward_pass_loss))

# Print the gradients
print("Gradient Difference (Backward Pass - Forward Pass):", forward_pass_loss - backward_pass_loss)
