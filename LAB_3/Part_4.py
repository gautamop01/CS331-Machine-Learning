import numpy as np
import random as rm

class NaiveBayesClassifier:
    def fit(self, X, y): # this -
        self.classes = np.unique(y)
        self.class_probs = {}
        for c in self.classes:
            self.class_probs[c] = np.sum(y == c) / len(y)

        self.feature_probs = {}
        
        for c in self.classes:
            class_indices = []
            class_indices = np.where(y == c)[0]

            class_X = []
            for i in class_indices:
                class_X.append(X[i])

            self.feature_probs[c] = {
                'mean': np.mean(class_X, axis=0),
                'std': np.std(class_X, axis=0) + 1e-10  # Adding a small epsilon to avoid division by zero
            }

    def calculate_probability(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2 + 1e-10)))
        return (1 / (np.sqrt(2 * np.pi) * (std + 1e-10))) * exponent

    def predict(self, X):
        predictions = []

        for x in X:
            class_probs = {}

            for c in self.classes:
                prior = np.log(self.class_probs[c])
                likelihood = np.sum(np.log(self.calculate_probability(x, self.feature_probs[c]['mean'], self.feature_probs[c]['std'] + 1e-10)))
                class_probs[c] = prior + likelihood

            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)

        return predictions
    
def normal_distribution(x, mean, std): # std is standard daviation which is sigma
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-(x - mean)**2 / (2 * std**2))


p = 0.1
y_train = []
X_train = []
for i in range(1000):
    y = np.random.choice([0, 1], p=[1-p, p])
    if y == 0:
        random_integer = rm.randint(-1000, 1000)
        x = normal_distribution(random_integer, -1, 1)
    else:
        random_integer = rm.randint(-1000, 1000)
        x = normal_distribution(random_integer, 1, 1)
    X_train.append(x)
    y_train.append(y)  # Extracting the actual value from the array


# Instantiate the classifier
nb_classifier = NaiveBayesClassifier()

# Train the classifier
nb_classifier.fit(X_train, y_train)
# # Assuming you want to generate 100 samples from a normal distribution with mean -1 and std deviation 1
X_test = X_train

# # Make predictions on the test set
predictions = nb_classifier.predict(X_test)  
print(predictions)