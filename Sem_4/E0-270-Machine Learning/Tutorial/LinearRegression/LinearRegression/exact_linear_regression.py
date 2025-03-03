from matplotlib import pyplot as plt

from utils import *
from model import LinearModel


def train(
        model: LinearModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
) -> None:
    '''
    Train the model using the training data (exact solution).

    Args:
        model: LinearModel object
        X_train: training data
        y_train: training labels

    Returns:
        None
    '''
    model.W = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train


def main(kernel_mode: bool = False) -> None:
    '''
    Main function.

    Args:
        kernel_mode: bool, whether to use a polynomial kernel.

    Returns:
        None
    '''
    # Load data
    X, y = load_data()
    if kernel_mode:
        X = np.concatenate([X, X ** 2], axis=1)

    # Preprocess the data
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    y = y.reshape(-1, 1)
    X_train, y_train, _, _, X_test, y_test = split_data(X, y)

    # Create and train the model
    model = LinearModel(input_dim=X_train.shape[0])
    train(model, X_train, y_train)

    # Predict
    y_pred = model(X)

    # Evaluate
    train_loss, train_metric = get_performance(model, X_train, y_train, 'mse', 'mae')
    test_loss, test_metric = get_performance(model, X_test, y_test, 'mse', 'mae')
    print(f'Train MSE: {train_loss}, Test MSE: {test_loss}')
    print(f'Train MAE: {train_metric}, Test MAE: {test_metric}')

    # Plot the fit
    plot_fit(X_train, y_train, model, kernel_mode=kernel_mode, partition='train')
    plot_fit(X_test, y_test, model, kernel_mode=kernel_mode, partition='test')


if __name__ == '__main__':
    main(kernel_mode=True)
