import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define covariance matrices
cov1 = np.array([[2, 0], [0, 1]])
cov2 = np.array([[0.2, 0], [0, 4]])
cov3 = np.array([[2, 1], [1, 2]])

covariances = np.stack([cov1, cov2, cov3])

N = 1500

# Assume true theta clustering for infinite'
true_theta_cluster1 = np.random.multivariate_normal(mean=[3, 3], cov=cov1, size=N)
true_theta_cluster2 = np.random.multivariate_normal(mean=[0, 0], cov=cov2, size=N)
true_theta_cluster3 = np.random.multivariate_normal(mean=[3, 0], cov=cov3, size=N)

# Calculate means (miu) for each cluster
true_miu = [np.mean(cluster, axis=0) for cluster in [true_theta_cluster1, true_theta_cluster2, true_theta_cluster3]]
true_miu = np.array(true_miu)
pi = [1/3, 1/3, 1/3]

# Combine true theta clusters
infinite_theta = np.vstack((true_theta_cluster1, true_theta_cluster2, true_theta_cluster3))
np.random.shuffle(infinite_theta)  # Shuffle the data

def initialize_centroids(data, k):
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]
    return centroids

def kmeans_algorithm(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(centroids, new_centroids, rtol=1e-4):
            break
        centroids = new_centroids
    return centroids, labels

def initialize_parameters(k, data):
    pi = np.ones(k) / k
    miu, labels = kmeans_algorithm(data, k)
    cov = [np.cov(data[labels == j].T) for j in range(k)]
    return pi, miu, cov, labels

# Computes the responsibilities (gamma) for each data point and cluster.
def e_step(data, miu, cov, pi):
    num_points = len(data)
    num_clusters = len(miu)
    gamma = np.zeros((num_points, num_clusters))
    
    for j in range(num_clusters):
        mvn = multivariate_normal(mean=miu[j], cov=cov[j])
        gamma[:, j] = pi[j] * mvn.pdf(data)

    gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
    return gamma

# Updates parameters based on current responsibilities.
def m_step(data, gamma):
    num_points, num_clusters = gamma.shape
    pi = np.sum(gamma, axis=0) / num_points
    miu = np.dot(gamma.T, data) / np.sum(gamma, axis=0, keepdims=True).T

    cov = [np.dot((data - miu[j]).T, (gamma[:, j][:, np.newaxis] * (data - miu[j]))) / np.sum(gamma[:, j]) for j in range(num_clusters)]
    
    return pi, miu, cov

# Computes the log likelihood of the data given the parameters.
def log_likelihood(data, pi, miu, cov):
    num_clusters = len(miu)
    num_points = len(data)
    
    likelihood = np.zeros((num_points, num_clusters))
    for j in range(num_clusters):
        mvn = multivariate_normal(mean=miu[j], cov=cov[j])
        likelihood[:, j] = pi[j] * mvn.pdf(data)

    log_likelihood = np.sum(np.log(np.sum(likelihood, axis=1)))

    return log_likelihood

def em_algorithm(data, k, max_iterations=100, tol=1e-4):
    pi, miu, cov, labels = initialize_parameters(k, data)
    log_likelihoods = []

    for iteration in range(max_iterations):
        gamma = e_step(data, miu, cov, pi)
        pi, miu, cov = m_step(data, gamma)

        # Calculate the log-likelihood
        log_likelihood_value = log_likelihood(data, pi, miu, cov)
        log_likelihoods.append(log_likelihood_value)

        # Print log likelihood at each iteration
        print(f"Iteration {iteration + 1}, Log Likelihood: {log_likelihood_value}")

        # Print sorted values for better visibility
        print("Pi:", np.sort(pi))
        print("Miu:", np.sort(miu, axis=0))
        print("Covariance matrices:")
        for i, cov_matrix in enumerate(cov):
            print(f"Cluster {i + 1}:\n{np.sort(cov_matrix)}")
        print("------------------------------")

        # Check for convergence
        if iteration > 0 and np.abs(log_likelihoods[iteration] - log_likelihoods[iteration - 1]) < tol:
            break

    return pi, miu, cov, gamma, log_likelihoods, labels

# def em_algorithm(data, k, max_iterations=100, tol=1e-4):
#     pi, miu, cov, labels = initialize_parameters(k, data)
#     log_likelihoods = []

#     for iteration in range(max_iterations):
#         gamma = e_step(data, miu, cov, pi)
#         pi, miu, cov = m_step(data, gamma)

#         # Calculate the log-likelihood
#         log_likelihood_value = log_likelihood(data, pi, miu, cov)
#         log_likelihoods.append(log_likelihood_value)

#         # Check for convergence
#         if iteration > 0 and np.abs(log_likelihoods[iteration] - log_likelihoods[iteration - 1]) < tol:
#             break

#     return pi, miu, cov, gamma, log_likelihoods, labels

# Load Old Faithful dataset skipping the header line
data = infinite_theta

# Part 3a: K-means clustering
k_means_centroids, k_means_labels = kmeans_algorithm(data, k=3)

# Plot the results of K-means
for i in range(3):
    plt.scatter(data[k_means_labels == i, 0], data[k_means_labels == i, 1], label=f'Cluster {i + 1}')
plt.scatter(k_means_centroids[:, 0], k_means_centroids[:, 1], marker='X', s=200, color='black', label='Centroids')
plt.title('K-means Clustering')
plt.legend()
plt.show()

# Save the K-means results for later use in EM algorithm
np.save('k_means_centroids.npy', k_means_centroids)
np.save('k_means_labels.npy', k_means_labels)

# Part 3b: EM algorithm
k = 3
pi_em, miu_em, cov_em, gamma_em, log_likelihoods_em, em_labels = em_algorithm(data, k=k)

# Plot the log likelihood graph
plt.plot(log_likelihoods_em, marker='o')
plt.title('EM Algorithm Log Likelihood')
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood')
plt.show()


# Plot the results of EM algorithm
for i in range(3):
    plt.scatter(data[gamma_em[:, i] > 0.5, 0], data[gamma_em[:, i] > 0.5, 1], label=f'Cluster {i + 1}')
plt.scatter(miu_em[:, 0], miu_em[:, 1], marker='X', s=200, color='black', label='Updated Centroids')
plt.title('EM Algorithm Clustering')
plt.legend()
plt.show()

# Save the EM algorithm results
np.save('pi_em.npy', pi_em)
np.save('miu_em.npy', miu_em)
np.save('cov_em.npy', cov_em)
np.save('gamma_em.npy', gamma_em)
np.save('log_likelihoods_em.npy', log_likelihoods_em)
np.save('em_labels.npy', em_labels)

Data1 = np.load('final_centroids.npy')