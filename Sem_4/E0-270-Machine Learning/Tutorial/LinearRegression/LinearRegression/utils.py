import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from model import LinearModel


def load_data(seed: int = 2024) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Create a toy dataset for regression demonstration

    Args:
        seed: int - the seed for the random number generator

    Returns:
        Tuple[np.ndarray, np.ndarray] - the data and the labels
    '''

    X = np.linspace(-2, 1, 200).reshape(-1, 1)
    y = -2 * X ** 2 + np.random.RandomState(seed).randn(*X.shape) * 0.33

    return X, y


def split_data(
        X: np.ndarray,
        y: np.ndarray,
        split_ratio: list = [0.7, 0.1, 0.2],
        random_state: int = 2024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split data into train, validation, and test sets. (default: 70%, 10%, 20%)

    Args:
        X: input data
        y: target data
        split_ratio: list of split ratios
        random_state: random seed

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    '''

    np.random.RandomState(random_state).shuffle(X)
    np.random.RandomState(random_state).shuffle(y)

    X_train = X[:int(X.shape[0] * split_ratio[0])]
    y_train = y[:int(y.shape[0] * split_ratio[0])]
    X_val = X[
        int(X.shape[0] * split_ratio[0]):int(X.shape[0] * (
                split_ratio[0] + split_ratio[1]))]
    y_val = y[
        int(y.shape[0] * split_ratio[0]):int(y.shape[0] * (
                split_ratio[0] + split_ratio[1]))]
    X_test = X[int(X.shape[0] * split_ratio[0] + split_ratio[1]):]
    y_test = y[int(y.shape[0] * split_ratio[0] + split_ratio[1]):]
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_performance(
        model: LinearModel,
        X: np.ndarray,
        y: np.ndarray,
        loss_name: str,
        metric_name: str,
        batch_size=32
) -> Tuple[float, float]:
    '''
    Get performance of model on data according to specified loss and metric.

    Args:
        model: model
        X: input data
        y: target data
        loss_name: loss function (e.g., mse, mae)
        metric_name: metric function (e.g., mse, mae, mape, etc.)
        batch_size: batch size

    Returns:
        loss, metric
    '''
    num_batches = X.shape[0] // batch_size + (X.shape[0] % batch_size > 0)

    if loss_name == 'mse':
        loss_fn = lambda y, y_pred: np.mean((y - y_pred) ** 2)
    elif loss_name == 'mae':
        loss_fn = lambda y, y_pred: np.mean(np.abs(y - y_pred))
    else:
        raise ValueError('Unknown loss function: %s' % loss_name)

    if metric_name == 'mse':
        metric_fn = lambda y, y_pred: np.mean((y - y_pred) ** 2)
    elif metric_name == 'mae':
        metric_fn = lambda y, y_pred: np.mean(np.abs(y - y_pred))
    else:
        raise ValueError('Unknown metric function: %s' % metric_name)

    loss, metric, total_samples = 0, 0, 0
    for i in range(num_batches):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        y_batch = y[i * batch_size:(i + 1) * batch_size]

        y_pred = model(X_batch)
        loss += loss_fn(y_batch, y_pred) * X_batch.shape[0]
        metric += metric_fn(y_batch, y_pred) * X_batch.shape[0]
        total_samples += X_batch.shape[0]
    loss /= total_samples
    metric /= total_samples

    return loss, metric


def plot_metrics(
        train_metrics: np.ndarray,
        val_metrics: np.ndarray,
        metric_name: str
) -> None:
    '''
    Plot training and validation metrics.

    Args:
        train_metric: training metric
        val_metric: validation metric
        metric_name: name of metric

    Returns:
        None
    '''
    plt.plot(train_metrics, label='train')
    plt.plot(val_metrics, label='val')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()


def plot_fit(
        X: np.ndarray,
        y: np.ndarray,
        model: LinearModel,
        kernel_mode: bool = False,
        partition: str = 'train'
) -> None:
    '''
    Plot data and model fit.

    Args:
        X: input data
        y: target data
        model: model
        kernel_mode: whether to use kernel
        partition: partition of data

    Returns:
        None
    '''
    plt.scatter(X[:, 0], y, label='data')
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    Xs = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    if kernel_mode:
        Xs = np.hstack([Xs, Xs ** 2])
    Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
    plt.plot(Xs[:, 0], model(Xs), label='fit', color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('%s %s' % (partition, 'fit'))
    plt.legend()
    plt.show()
