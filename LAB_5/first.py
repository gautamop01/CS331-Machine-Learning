import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to initialize centroids
def initialize_centroids(data, k):
    # Randomly select k data points as initial centroids
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data.iloc[indices].to_numpy()
    return centroids

# Function to perform K-means clustering
def kmeans_algorithm(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(data.values[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # Update centroids based on mean of assigned data points
        new_centroids = np.array([data.values[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.allclose(centroids, new_centroids, rtol=1e-4):
            break

        centroids = new_centroids

    # Plot the results
    for i in range(k):
        plt.scatter(data.values[labels == i, 0], data.values[labels == i, 1], label=f'Cluster {i + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='black', label='Centroids')
    plt.title('K-means Clustering')
    plt.legend()
    plt.show()

    return centroids

# Load Old Faithful dataset
data = pd.read_csv('OldFaithful.csv')

# Initialize centroids using K-means
k = 2
final_centroids = kmeans_algorithm(data, k)

# Save the final centroids for later use in EM algorithm
np.save('final_centroids.npy', final_centroids)
