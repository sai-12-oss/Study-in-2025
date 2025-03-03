import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union


def get_classification_dataset(
        var: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate a dataset with two classes 1 and 0.

    Args:
        var: used in the covariance of the gaussians

    Returns:
        X: the dataset
        y: the labels
    '''
    mean1 = [-2, 2]
    mean2 = [1, -1]
    cov = [[var, 0], [0, var]]
    cluster1 = np.random.multivariate_normal(mean1, cov, 100)
    cluster2 = np.random.multivariate_normal(mean2, cov, 100)
    X = np.vstack((cluster1, cluster2))
    y = np.concatenate((np.ones(100, dtype=int), np.zeros(100, dtype=int)))
    return X, y


def get_regression_dataset() -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate a regression dataset that is generated from a quadratic polynomial.

    Returns:
        X: the dataset
        y: the labels
    '''
    X = np.linspace(-2, 1, 200).reshape(-1, 1)
    y = - 2 * X * X + np.random.randn(*X.shape) * 0.33
    return X, y


def plot_classification_datapoints(
        X: np.ndarray,
        y: np.ndarray,
        x: Optional[Union[np.ndarray, list]] = None,
        circle: bool = False
) -> None:
    '''
    Plot the classification datapoints.

    Args:
        X: the dataset
        y: the labels
        x: the datapoint to plot
        circle: if True, plot a circle, else plot a dot

    Returns:
        None
    '''
    cluster1 = X[y == 1]
    cluster2 = X[y == 0]
    plt.figure(figsize=(8, 6))
    plt.scatter(cluster1[:, 0], cluster1[:, 1], color='green', label='Class 1', marker='+')
    plt.scatter(cluster2[:, 0], cluster2[:, 1], color='red', label='Class 0', marker='x')
    if x is not None:
        if circle:
            plt.scatter(x[0], x[1], marker='o', facecolor='none', edgecolor='blue', s=100)
        else:
            plt.scatter(x[0], x[1], marker='.', color='blue')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_regression_datapoints(X: np.ndarray, y: np.ndarray) -> None:
    '''
    Plot the regression datapoints.

    Args:
        X: the dataset
        y: the labels

    Returns:
        None
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', marker='.')
    plt.xlabel('Feature')
    plt.ylabel('Label')
    plt.grid(True)
    plt.show()


def plot_decision_boundary(
        clf: object,
        X: np.ndarray,
        Y: np.ndarray,
        partition: str = '',
        cmap: str = 'Paired_r'
) -> None:
    '''
    Plot decision boundary of a classification model.

    Args:
        clf: model
        X: input data
        Y: target data
        partition: partition of data
        cmap: colormap

    Returns:
        None
    '''
    h = 0.02
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 10 * h, X[:, 1].max() + 10 * h
    xs, ys = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = []
    for x, y in zip(xs.ravel(), ys.ravel()):
        Z.append(clf.predict(np.asarray([x, y]).reshape(1, -1)) > 0.5)
    Z = np.array(Z).reshape(xs.shape)

    plt.figure(figsize=(8, 8))
    plt.contourf(xs, ys, Z, cmap=cmap, alpha=0.25)
    plt.contour(xs, ys, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, edgecolors='k')
    plt.title('%s decision boundary' % partition)
    plt.show()


def plot_fit(
        clf: object,
        X: np.ndarray,
        Y: np.ndarray,
        partition: str = ''
) -> None:
    '''
    Plot fit of a regression model.

    Args:
        clf: model
        X: input data
        Y: target data
        partition: partition of data

    Returns:
        None
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', marker='.')
    # plt.scatter(X, clf.predict(X), color='red')
    X_test = np.linspace(-2, 1, 200).reshape(-1, 1)
    plt.plot(X_test, clf.predict(X_test), color='red')
    plt.xlabel('Feature')
    plt.ylabel('Label')
    plt.title('%s fit' % partition)
    plt.grid(True)
    plt.show()


def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        seed=2025
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split the dataset into train and test sets.

    Args:
        X: the dataset
        y: the labels
        test_size: the size of the test set
        seed: the seed for the random number generator

    Returns:
        X_train: the train set
        y_train: the train labels
        X_test: the test set
        y_test: the test labels
    '''
    assert 0.0 <= test_size <= 1.0
    indices = np.arange(X.shape[0])
    np.random.RandomState(seed).shuffle(indices)
    X_train = X[indices[:-int(test_size * X.shape[0])]]
    y_train = y[indices[:-int(test_size * y.shape[0])]]
    X_test = X[indices[-int(test_size * X.shape[0]):]]
    y_test = y[indices[-int(test_size * y.shape[0]):]]
    return X_train, y_train, X_test, y_test
