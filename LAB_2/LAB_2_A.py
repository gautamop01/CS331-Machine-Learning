# Gautam Kumar Mahar 
# 2103114 
# Machine Learning LAB 2, BONUS
#------------------

import numpy as np

np.random.seed(100)
U = np.random.rand(1, 10)
X = U.flatten()

true_a = 2.0
true_b = 1.0

noise = np.random.normal(0, 1, size=len(X))
Y = true_a * X + true_b + noise

data = np.column_stack((X, Y))

weights = np.random.rand(1)
bias = np.random.rand(1)

learning_rate = 0.001
# Number of approaches 
epochs = 1000

for epoch in range(epochs):
    # y = wx + b
    predicted_output = X * weights + bias

    # (y' - y)**2 / N;
    # Forward Pass Loss
    loss = 0.5 * np.mean((predicted_output - Y) ** 2)
    
    dL_dw = np.mean(X * (predicted_output - Y))
    dL_db = np.mean(predicted_output - Y)
    
    weights -= learning_rate * dL_dw
    bias -= learning_rate * dL_db
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

print("\nTrue Weights and Bias:")
print("True Slope (a):", true_a)
print("True Intercept (b):", true_b)

print("\nLearned Weights and Bias:")
print("Learned Slope (a):", weights[0])
print("Learned Intercept (b):", bias[0])













