from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Boston dataset
boston = load_boston()
X, y = boston.data, boston.target

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias term to the input
X = np.c_[X, np.ones(X.shape[0])]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize weights and bias
W = np.random.randn(X.shape[1], 1)
b = np.zeros(1)

# Stochastic Gradient Descent function
def stochastic_gradient_descent(X, y, W, b, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        for i in range(len(X)):
            # Forward pass
            pred = np.dot(X[i], W) + b
            loss = MeanSquaredLossLayer().forward(pred, y[i])

            # Backward pass
            grad_loss = MeanSquaredLossLayer().backward()
            grad_X, grad_W = MatrixMultiplyLayer().backward(grad_loss)
            grad_X_bias, grad_b = BiasAdditionLayer().backward(grad_X)

            # Update weights and bias
            W -= learning_rate * grad_W
            b -= learning_rate * grad_b

    return W, b

# Train the model
W_trained, b_trained = stochastic_gradient_descent(X_train, y_train.reshape(-1, 1), W, b)

# Test the model
predictions = np.dot(X_test, W_trained) + b_trained
mse = MeanSquaredLossLayer().forward(predictions, y_test)
print("Mean Squared Error on Test Set:", mse)
