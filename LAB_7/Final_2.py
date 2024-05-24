from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def perform_regression(X, y, degree):
    """
    Perform polynomial regression with the specified degree.
    """
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    coefficients = model.coef_
    coefficients[0] = model.intercept_

    return coefficients

def add_intercept_and_polynomial_terms(x, degree):
    """
    Add a constant term (intercept) and polynomial terms to the input features matrix.
    """
    x = x.reshape(-1, 1)
    x_with_intercept = np.insert(x, 0, 1, axis=1)

    for d in range(2, degree + 1):
        x_poly = np.power(x, d)
        x_with_intercept = np.concatenate((x_with_intercept, x_poly), axis=1)

    return x_with_intercept

def predict_y(coefficients, x, degree):
    """
    Calculate the predicted y given coefficients and input features x.
    """
    if degree == 0:
        return coefficients[0]

    x_with_intercept_and_poly = add_intercept_and_polynomial_terms(x, degree)
    y_pred = np.dot(x_with_intercept_and_poly, coefficients)

    return y_pred

# Constants and functions for generating data
num_data = 200
num_samples = 2
num_test = 20
sigma_noise = 1

def f(x):
    return np.sin(x)

def getSample():
    X = []
    Y = []

    for j in range(num_samples):
        epsilon = np.random.normal(loc=0, scale=sigma_noise)
        x = np.random.uniform(-1, 1)
        y = f(x) + epsilon
        X.append([x])
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# H0: Single Valued Functions
g_bar = 0
degree = 0

for i in range(num_data):
    X, Y = getSample()
    coefficients = perform_regression(X, Y, degree)
    g_bar += coefficients

g_bar /= num_data
print("-------------------------------------------")
print("G_bar:", g_bar)

Variance = 0
Bias = 0

for i in range(num_test):
    X, Y = getSample()
    coefficients = perform_regression(X, Y, degree)

    var = 0
    b = 0
    for k in range(num_samples):
        var += (predict_y(coefficients, X[k], degree) - predict_y(g_bar, X[k], degree))**2
        b += (f(X[k]) - predict_y(g_bar, X[k], degree))**2

    Bias += b / num_samples
    Variance += var / num_samples

Bias /= num_data
Variance /= num_data

print("H0:\nBias:", Bias)
print("Variance:", Variance)
print("Eout:", Bias + Variance + sigma_noise**2)
arg_min_h0 = Bias + Variance + sigma_noise**2
print("----------------------------------")

# H1: Linear Functions
g_bar = 0
degree = 1

for i in range(num_data):
    X, Y = getSample()
    coefficients = perform_regression(X, Y, degree)
    g_bar += coefficients

g_bar /= num_data
print("G_bar:", g_bar)

Variance = 0
Bias = 0

for i in range(num_test):
    X, Y = getSample()
    coefficients = perform_regression(X, Y, degree)

    var = 0
    b = 0
    for k in range(num_samples):
        var += (predict_y(coefficients, X[k], degree) - predict_y(g_bar, X[k], degree))**2
        b += (f(X[k]) - predict_y(g_bar, X[k], degree))**2

    Bias += b / num_samples
    Variance += var / num_samples

Bias /= num_data
Variance /= num_data

print("H1:\nBias:", Bias)
print("Variance:", Variance)
print("Eout:", Bias + Variance + sigma_noise**2)
arg_min_h1 = Bias + Variance + sigma_noise**2
print("---------------------------------------")

# H2: Quadratic Functions
g_bar = 0
degree = 2

for i in range(num_data):
    X, Y = getSample()
    coefficients = perform_regression(X, Y, degree)
    g_bar += coefficients

g_bar /= num_data
print("G_bar:", g_bar)

Variance = 0
Bias = 0

for i in range(num_test):
    X, Y = getSample()
    coefficients = perform_regression(X, Y, degree)

    var = 0
    b = 0
    for k in range(num_samples):
        var += (predict_y(coefficients, X[k], degree) - predict_y(g_bar, X[k], degree))**2
        b += (f(X[k]) - predict_y(g_bar, X[k], degree))**2

    Bias += b / num_samples
    Variance += var / num_samples

Bias /= num_data
Variance /= num_data

print("H2:\nBias:", Bias)
print("Variance:", Variance)
print("Eout:", Bias + Variance + sigma_noise**2)
arg_min_h2 = Bias + Variance + sigma_noise**2

print("---------------------------------------")
print("---------------------------------------")
print("argmin :")
variables = {'arg_min_h0': arg_min_h0, 'arg_min_h1': arg_min_h1, 'arg_min_h2': arg_min_h2}

# Find the variable with the minimum value
min_var_name = min(variables, key=variables.get)
min_value = variables[min_var_name]

print("Minimum:", min_var_name, "=", min_value)