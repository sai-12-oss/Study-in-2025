import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(clf, X, Y, partition="train", cmap='Paired_r'):
    """
    Plots the decision boundary of the model on the dataset.
    """
    # Create a meshgrid of points
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = []
    for x, y in zip(xx.ravel(), yy.ravel()):
        Z.append(clf(np.array([x, y])))
    Z = np.array(Z).reshape(xx.shape)

    # Plot the contour and training examples
    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, edgecolors='k')
    plt.title(f"Decision boundary for {partition} set")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

