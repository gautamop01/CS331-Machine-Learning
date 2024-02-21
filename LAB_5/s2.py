import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set a random seed for reproducibility
np.random.seed(42)

# Define covariance matrices
cov1 = np.array([[2, 0], [0, 1]])
cov2 = np.array([[0.2, 0], [0, 4]])
cov3 = np.array([[2, 1], [1, 2]])

covariances = np.stack([cov1, cov2, cov3])

def Calculate_post(data, miu, cov, pi, num_clusters):
    num_points = len(data)
    num_clusters = len(miu)
    # posterior probabilities (gamma)
    gamma = np.zeros((num_points, num_clusters))
    
    # Loop over Cluster
    for j in range(num_clusters):
        mvn = multivariate_normal(mean=miu[j], cov=cov[j])
        gamma[:, j] = pi[j] * mvn.pdf(data)
    # Normalize Gamma
    gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
    return gamma

# Assuming N = 1000 for better separation
N = 1500

# Assume true theta clustering for infinite'
true_theta_cluster1 = np.random.multivariate_normal(mean=[3, 3], cov=cov1, size=N)
true_theta_cluster2 = np.random.multivariate_normal(mean=[0, 0], cov=cov2, size=N)
true_theta_cluster3 = np.random.multivariate_normal(mean=[3, 0], cov=cov3, size=N)

# Combine true theta clusters
infinite_theta = np.vstack((true_theta_cluster1, true_theta_cluster2, true_theta_cluster3))
np.random.shuffle(infinite_theta)  # Shuffle the data

# Calculate means (miu) for each cluster
true_miu = [np.mean(cluster, axis=0) for cluster in [true_theta_cluster1, true_theta_cluster2, true_theta_cluster3]]
true_miu = np.array(true_miu)
pi = [1/3, 1/3, 1/3]

gamma = Calculate_post(infinite_theta, true_miu, covariances, pi, 3) 
z = np.argmax(gamma, axis=1, keepdims=True)
print(np.unique(z, return_counts=True))

# Plot the results with different colors for each covariance matrix
plt.scatter(infinite_theta[:, 0], infinite_theta[:, 1], label='Cluster 3 (True Theta)', c=z.flatten(), alpha=0.5)
plt.title('Question 2b - True Theta Clustering and Scatter Plot')
plt.legend()
plt.show()

# Save the means for later use in EM algorithm
np.save('true_miu.npy', true_miu)
