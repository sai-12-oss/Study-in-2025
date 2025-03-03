import numpy as np
from cvxopt import matrix, solvers

from utils import *


np.random.seed(2023)
X, y = get_classification_dataset2()
y[y == 0] = -1

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)


class SVM:
    def __init__(self, kernel='linear', gamma=1.0):
        self.alphas = None
        self.X = None
        self.y = None
        self.b = None
        self.mode = kernel
        self.gamma = gamma

    def kernel(self, x1, x2):
        if self.mode == 'linear':
            return x1.T @ x2
        elif self.mode == 'poly':
            return (x1.T @ x2 + 1) ** 2
        elif self.mode == 'rbf':
            return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.gamma ** 2))

    def fit(self, X, y):
        y = y.reshape(-1, 1) * 1.0
        # objective
        # P = matrix((X * y) @ (X * y).T)
        P = np.zeros((X.shape[0], X.shape[0]), dtype=np.double)
        for i in range(X.shape[0]):
            for j in range(i, X.shape[0]):
                P[i, j] = P[j, i] = y[i] * y[j] * self.kernel(X[i], X[j])
        P = matrix(P)
        q = matrix(-np.ones((X.shape[0], 1), dtype=np.double))
        # inequality constraints
        G = matrix(-np.eye(X.shape[0], dtype=np.double))
        h = matrix(np.zeros((X.shape[0]), dtype=np.double))
        # equality constraints
        A = matrix(y.T)
        b = matrix(0.0)

        solvers.options['show_progress'] = True
        solvers.options['abstol'] = 1e-8
        solvers.options['reltol'] = 1e-8
        solvers.options['feastol'] = 1e-8

        solver = solvers.qp(P, q, G, h, A, b)
        if solver['status'] != 'optimal':
            print('Optimization failed')
            return

        self.alphas = np.array(solver['x'])
        idxs = np.where(self.alphas > 1e-4)[0]
        self.alphas = self.alphas[idxs]
        self.X = X[idxs]
        self.y = y[idxs]

        self.b = 0
        for i in range(len(self.alphas)):
            if self.alphas[i] < 1:
                continue
            self.b += self.y[i]
            for j in range(len(self.alphas)):
                self.b -= self.alphas[j] * self.y[j] *\
                          self.kernel(self.X[j], self.X[i])
        self.b /= len(np.where(self.alphas > 1)[0])

    def predict(self, X, raw=False):
        assert self.alphas is not None, 'Model not trained yet'
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        y_preds = []
        for x in X:
            y_pred = 0
            for i in range(len(self.alphas)):
                y_pred += self.alphas[i] * self.y[i] *\
                          self.kernel(x.T, self.X[i])
            y_pred += self.b
            y_preds.append(y_pred)
        if raw:
            return np.array(y_preds)
        return np.array([np.sign(y_pred) for y_pred in y_preds])


clf = SVM(kernel='rbf', gamma=1.0)
clf.fit(X_train, y_train)
plot_decision_boundary(clf, X_train, y_train, partition='Train', margin=True)
plot_decision_boundary(clf, X_test, y_test, partition='Test', margin=True)
