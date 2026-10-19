import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.01, batch_size = 2000):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.W = None
        self.b = None

    def fit(self, x, y):
        self.W = np.random.randn(1, x.shape[1]) / np.sqrt(x.shape[1])
        self.b = 0
        costs = []
        for i in range(self.batch_size):
            z = self.W @ x.T + self.b
            a = ActivationFunctions.sigmoid(z)
            costs.append(np.sum(self.loss_function(a, y)))
            a_pred = a >= 0.5
            acc = np.mean(a_pred == y) * 100
            print(f"batch {i}, cost: {costs[-1]}, accuracy: {acc}")
            self.gradient_descent(self, a, y, x)
    
    @staticmethod
    def loss_function(a, y):
        return  - (y * np.log(a) + (1 - y) * np.log(1 - a))

    @staticmethod
    def gradient_descent(self, a, y, x):
        dz = a - y
        dw = np.dot(dz, x) / x.shape[0]
        db = np.sum(dz, axis=0) / x.shape[0]
        self.W -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, x):
        z = self.W @ x.T + self.b[:x.shape[0]]
        a = ActivationFunctions.sigmoid(z)
        return a >= 0.5

class ActivationFunctions:
        def __init__(self):
            self

        def sigmoid(z):
             return 1 / (1 + np.exp(-z))