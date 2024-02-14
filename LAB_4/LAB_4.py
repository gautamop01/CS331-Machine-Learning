# Gautam Kumar Mahar
# 2103114
# LAB 4 

import numpy as np
import matplotlib.pyplot as plt

def e_step(data, pii, pi):
    # Calculate the gamma_i_j (gammaij) for each coin and each sample
    gamma_i_j = np.zeros((len(data), len(pii)))
    
    for i in range(len(data)):
        likelihood_heads = np.power(pii, data[i]) * np.power(1 - pii, 10 - data[i])
        gamma_i_j[i, :] = pi * likelihood_heads / np.sum(pi * likelihood_heads)
    
    return gamma_i_j

def m_step(data, gamma_i_j):
    # Update parameters pii and pi based on the observed data and gamma_i_j
    N = len(data)
    
    # Convert data to a numpy array
    data_array = np.array(data)
    
    # Update pii for each coin
    pii = np.sum(gamma_i_j * data_array[:, np.newaxis], axis=0) / (np.sum(gamma_i_j, axis=0) * 10)
    
    # Update pi for each coin
    pi = np.sum(gamma_i_j, axis=0) / N
    
    return pii, pi

def expectation_maximization(data, n_coins, n_samples, n_iterations):
    # Initialize parameters randomly
    pii = np.random.rand(n_coins)
    pi = np.ones(n_coins) / n_coins
    
    for _ in range(n_iterations):
        # E-step
        gamma_i_j = e_step(data, pii, pi)
        
        # M-step
        pii, pi = m_step(data, gamma_i_j)
    
    return pii, pi

# Example usage
np.random.seed(42)
# Simulate data with 3 coins and 100 samples using binomial distribution
true_pii = np.array([0.3, 0.05,0.1, 0.35, 0.9]) # probability of each coin 
true_pi = np.array([0.2,  0.2, 0.1, 0.3, 0.2])  # probability of chossing coin

# Define the number of samples
n_samples = 1000
n_coins = len(true_pii)

# Sample from true_pi to get data2
data2 = np.random.choice(true_pii, n_samples, p=true_pi)

# Use binomial distribution based on data2 to generate data
data = [np.random.binomial(10, i) for i in data2]

# Run EM algorithm
estimated_pii, estimated_pi = expectation_maximization(data, n_coins=n_coins, n_samples=n_samples, n_iterations=100)

# print("Generated data (data):", data)
# print("Sampled data2 (data2):", data2)
print("True pi:", true_pi)
print("Estimated pi:", estimated_pi)
print("True pii:", true_pii)
print("Estimated pii:", estimated_pii)
