import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


class FtLogisticRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, max_iter=1000, learning_rate=0.1):
        X_cpy = X.copy()
        rows, cols = X_cpy.shape
        self.weights = np.zeros(cols + 1)
        X_cpy = np.concatenate((np.ones((rows, 1)), X_cpy), axis=1)
        for it in range(max_iter):
            p = self._sigmoid(self._logit(X_cpy))
            grad = (1.0 / rows) * np.dot(X_cpy.T, (p - y))
            self.weights -= learning_rate * grad

    def predict_proba(self, X):
        rows = X.shape[0]
        X_cpy = np.concatenate((np.ones((rows, 1)), X), axis=1)
        return self._sigmoid(self._logit(X_cpy))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.weights

    def _sigmoid(self, t):
        print(t)
        return 1.0 / (1 + np.exp(-t))

    def _logit(self, X):
        return np.dot(X, self.weights)

    def _log_loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


df = pd.read_csv('datasets/dataset_train.csv')
y = df['Hogwarts House']
X = df.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y_train_gryf = (y_train == 'Gryffindor')
clf = FtLogisticRegression()
clf.fit(X_train, y_train_gryf)
pred = clf.predict(X_test)
