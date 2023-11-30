import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * X.T @ (h - y)
        theta -= learning_rate * gradient
    return theta

def predict(X, theta):
    h = sigmoid(X @ theta)
    return np.argmax(h, axis=1)