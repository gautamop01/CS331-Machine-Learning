import numpy as np

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
