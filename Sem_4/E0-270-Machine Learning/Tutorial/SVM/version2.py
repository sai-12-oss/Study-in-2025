import numpy as np
from cvxopt import matrix, solvers

from utils import *


np.random.seed(2023)
X, y = get_classification_dataset(var=1.0)
y[y == 0] = -1
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)


class SVM:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y, c=1.0):
        X = np.hstack((X, np.zeros((X.shape[0], 1))))
        X2 = np.zeros_like(X)
        X2[:, -1] = 1
        y = y.reshape(-1, 1)
        # objective
        P = matrix(np.eye(X.shape[1] + X.shape[0], dtype=np.double))
        P[X.shape[0] - 1:, X.shape[0] - 1:] = 0
        q = c * np.ones((X.shape[1] + X.shape[0], 1), dtype=np.double)
        q[:X.shape[1]] = 0
        q = matrix(q)
        # inequality constraints
        G = np.concatenate([y * X + y * X2, np.eye(X.shape[0])], axis=1)
        G = matrix(-G)
        h = matrix(-np.ones_like(y, dtype=np.double))

        solvers.options['show_progress'] = True
        solvers.options['abstol'] = 1e-8
        solvers.options['reltol'] = 1e-8
        solvers.options['feastol'] = 1e-8

        solver = solvers.qp(P, q, G, h)
        if solver['status'] != 'optimal':
            print('Optimization failed')
            return

        self.w = np.array(solver['x'])
        self.b = self.w[X.shape[1]-1]
        self.w = self.w[:X.shape[1]-1]

    def predict(self, X, raw=False):
        assert self.w is not None, 'Model not trained yet'
        if raw:
            return X @ self.w + self.b
        return np.sign(X @ self.w + self.b)


clf = SVM()
clf.fit(X_train, y_train, c=.5)
plot_decision_boundary(clf, X_train, y_train, partition='Train', margin=True)
plot_decision_boundary(clf, X_test, y_test, partition='Test', margin=True)
