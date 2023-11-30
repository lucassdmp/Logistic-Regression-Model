import numpy as np

class Standizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        standardized_X = (X - self.mean) / self.std
        return standardized_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)