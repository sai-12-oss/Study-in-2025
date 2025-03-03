import numpy as np
import matplotlib.pyplot as plt


class ArgumentStorage:
    def __init__(self, args):
        self.__dict__.update(args)


def tanh(x):
    return np.tanh(x)


def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 10 * h, X[:, 1].max() + 10 * h
    xs, ys = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = []
    for x, y in zip(xs.ravel(), ys.ravel()):
        Z.append(clf([x, y]) > 0.5)
    Z = np.array(Z).reshape(xs.shape)

    plt.figure(figsize=(5, 5))
    plt.contourf(xs, ys, Z, cmap=cmap, alpha=0.25)
    plt.contour(xs, ys, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, edgecolors='k')
    plt.show()
