import numpy as np
import matplotlib.pyplot as plt

# Assuming N = 1500 for demonstration purposes
N = 1500

# Generate D = {(x_i, z_i)}_i=1^N
np.random.seed(42)  # Set seed for reproducibility

# Generate clusters with different covariances
cluster1 = np.random.multivariate_normal(mean=[6, 6], cov=np.array([[2, 0], [0, 1]]), size=N)
cluster2 = np.random.multivariate_normal(mean=[0, 0], cov=np.array([[0.2, 0], [0, 4]]), size=N)
cluster3 = np.random.multivariate_normal(mean=[6, 0], cov=np.array([[2, 1], [1, 2]]), size=N)

# Combine clusters to form D
D = np.vstack((cluster1, cluster2, cluster3))
np.random.shuffle(D)  # Shuffle the data

# Plot the results with different colors for each covariance matrix
plt.scatter(cluster1[:, 0], cluster1[:, 1], label='Cluster 1', color='red', alpha=0.5)
plt.scatter(cluster2[:, 0], cluster2[:, 1], label='Cluster 2', color='green', alpha=0.5)
plt.scatter(cluster3[:, 0], cluster3[:, 1], label='Cluster 3', color='blue', alpha=0.5)

plt.title('Question 2a - Generating D and Scatter Plot')
plt.legend()
plt.show()

# Save the generated data D for later use
np.save('generated_data.npy', D)
