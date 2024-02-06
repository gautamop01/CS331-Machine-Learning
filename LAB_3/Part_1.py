import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.stats import norm

# a. Generate 100 points i.i.d. from p=0.4
np.random.seed(42)
points = np.random.choice([0, 1], size=100, p=[0.6, 0.4])

# b. L(p) as a function of p
def likelihood(p, data):
    return np.prod([p if x == 1 else (1 - p) for x in data])

# c. l(p) as a function of p
# logarithm of likelihoods
def log_likelihood(p, data):
    return np.sum([np.log(p) if x == 1 else np.log(1 - p) for x in data])

# d. Indicate ^p
maximum_likelihood_p = np.sum(points) / len(points)

# part 1
# Plotting L(p) and l(p)
p_values = np.linspace(0, 1, 100)
L_values = [likelihood(p, points) for p in p_values]
l_values = [log_likelihood(p, points) for p in p_values]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(p_values, L_values)
plt.title('Likelihood Function (L(p))')

plt.subplot(1, 2, 2)
plt.plot(p_values, l_values)
plt.title('Log-Likelihood Function (l(p))')

plt.show()

# Print the maximum likelihood estimate ^p
print(f"Maximum Likelihood Estimate (^p): {maximum_likelihood_p}")

num_points = 100
actual_data = np.random.choice([0, 1], size=num_points, p=[0.9, 0.1])

probabilities = np.linspace(0, 1, num_points)
min_error = float('inf')
best_probability = None
errors = [] # list

for prob in probabilities:
    new_data = np.random.choice([0, 1], size=num_points, p=[1 - prob, prob])
    error = np.mean(np.abs(new_data - actual_data))
    errors.append(error)
    if error < min_error:
        min_error = error
        best_probability = prob

print(f"Least Error: {min_error}")
print(f"Corresponding Probability: {best_probability}")


plt.plot(probabilities, errors, label='Error vs Probability')
plt.xlabel('Probability')
plt.ylabel('Error')
plt.title('Error vs Probability for Varying Probabilities')
plt.legend()
plt.show()


# Part 3n 

# Given probabilities
p0 = 0.1
p1 = 0.9

# Generate y using p0 and p1
y = np.random.choice([0, 1], size=100, p=[p0, p1])


# Generate f0(x) and f1(x) using y
f0_x = np.random.normal(loc=-1, scale=1, size=len(y))
f1_x = np.random.normal(loc=1, scale=1, size=len(y))

# Generate q_i(x) for i = 0, 1
qi_0_x = np.exp(-0.5 * (f0_x**2)) / np.sqrt(2 * np.pi)
qi_1_x = np.exp(-0.5 * (f1_x**2)) / np.sqrt(2 * np.pi)

# Generate q1(0), q0(0), and other sets
q1_0 = np.exp(-0.5 * (0 - 1)**2) / np.sqrt(2 * np.pi)
q0_0 = np.exp(-0.5 * (0 + 1)**2) / np.sqrt(2 * np.pi)

# Generate h_B binary sequence based on qi(x)
h_B = np.where(qi_1_x > qi_0_x, 1, 0)

# Calculate error
error = np.mean(np.abs(y - h_B))

# Print results
print(f"Generated y: {y}")
print(f"Generated h_B: {h_B}")
print(f"Error: {error}")
