import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic data with blobs for binary classification
np.random.seed(42)
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Plot the generated blobs
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Data Points')
plt.title('Generated Blobs for Binary Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Initialize weights
w = np.random.rand(X.shape[1])

check = True

while check:
    check = False

    for i in range(len(y)):
        # Compute the dot product w @ X[i].T
        dot_product = np.dot(w, X[i])

        # Check misclassification and update weights
        if (y[i] == 0 and dot_product > 0) or (y[i] == 1 and dot_product < 0):
            check = True

          # Update weights based on perceptron learning rule
            # w_{t+1} = w_{t} + (y_i - prediction) * x_i
            w = w + (y[i] - dot_product) * X[i]

# Plot the decision boundary
x_values = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y_values = -(w[0] * x_values) / w[1]

plt.plot(x_values, y_values, label='Decision Boundary', color='blue')
plt.legend()
plt.show()
