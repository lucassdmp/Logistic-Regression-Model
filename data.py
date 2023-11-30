import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the Iris class
class Iris:
    def __init__(self, id, sepal_length, sepal_width, petal_length, petal_width, species):
        self.id = int(id)
        self.sepal_length = float(sepal_length)
        self.sepal_width = float(sepal_width)
        self.petal_length = float(petal_length)
        self.petal_width = float(petal_width)
        self.species = species

    def __str__(self):
        return f'{self.sepal_length}, {self.sepal_width}, {self.petal_length}, {self.petal_width}, {self.species}'

# Load data from CSV
data = pd.read_csv('Iris.csv')

# Create a list of Iris objects
iris_list = [Iris(*row[1:]) for row in data.itertuples()]

# Extract features and labels
X = np.array([[iris.id, iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width] for iris in iris_list])
y = np.array([1 if iris.species == 'Iris-versicolor' else 0 for iris in iris_list])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add a bias term to the features
X_train = np.hstack((np.ones((len(X_train), 1)), X_train))
X_test = np.hstack((np.ones((len(X_test), 1)), X_test))

# Initialize parameters
theta = np.zeros(X_train.shape[1])

# Set hyperparameters
learning_rate = 0.01
iterations = 2000

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the log-likelihood loss function
def log_likelihood(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    
    return -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Define the gradient descent optimization
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * X.T @ (h - y)
        theta -= learning_rate * gradient
    return theta

# Train the model
theta = gradient_descent(X_train, y_train, theta, learning_rate, iterations)

# Make predictions
def predict(X, theta):
    h = sigmoid(X @ theta)
    return (h >= 0.5).astype(int)

y_pred = predict(X_test, theta)

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
