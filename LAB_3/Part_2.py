import numpy as np
import matplotlib.pyplot as plt

num_points = 100
actual_data = np.random.choice([0, 1], size=num_points, p=[0.9, 0.1])

probabilities = np.linspace(0, 1, num_points)
errors = [] # list

for prob in probabilities:
    new_data = np.random.choice([0, 1], size=num_points, p=[1 - prob, prob])
    error = np.mean(np.abs(new_data - actual_data))
    errors.append(error)

plt.plot(probabilities, errors, label='Error vs Probability')
plt.xlabel('Probability')
plt.ylabel('Error')
plt.title('Error vs Probability for Varying Probabilities')
plt.legend()
plt.show()
