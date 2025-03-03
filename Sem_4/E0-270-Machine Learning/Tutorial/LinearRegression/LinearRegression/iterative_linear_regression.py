from matplotlib import pyplot as plt

from utils import *
from model import LinearModel


def train(
        model: LinearModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train the model using the training data (exact solution).

    Args:
        model: LinearModel object
        X_train: training data
        y_train: training labels
        X_val: validation data
        y_val: validation labels
        batch_size: int, batch size
        num_epochs: int, number of epochs
        learning_rate: float, learning rate

    Returns:
        train_losses: np.ndarray, training loss
        train_metrics: np.ndarray, training metric
        val_losses: np.ndarray, validation loss
        val_metrics: np.ndarray, validation metric
    '''
    loss_fn = lambda y, y_pred: np.mean((y - y_pred) ** 2)  # MSE
    metric_fn = lambda y, y_pred: np.mean(np.abs(y - y_pred))  # MAE

    train_losses, train_metrics, val_losses, val_metrics = [], [], [], []

    num_train_batches =\
        X_train.shape[0] // batch_size + (X_train.shape[0] % batch_size != 0)
    num_val_batches =\
        X_val.shape[0] // batch_size + (X_val.shape[0] % batch_size != 0)

    for epoch in range(num_epochs):
        # Train
        train_loss_epoch, train_metric_epoch, total_train_samples = 0, 0, 0
        shuffled_idxs = np.random.permutation(X_train.shape[0])
        for i in range(num_train_batches):
            batch_idxs =\
                shuffled_idxs[i * batch_size:(i + 1) * batch_size]
            X_batch = X_train[batch_idxs]
            y_batch = y_train[batch_idxs]

            y_batch_pred = model(X_batch)
            loss = loss_fn(y_batch, y_batch_pred)

            grad_W = 2 * (X_batch.T @ X_batch @ model.W - X_batch.T @ y_batch)
            model.W -= learning_rate * grad_W  # SGD

            train_loss_epoch += loss * X_batch.shape[0]
            train_metric_epoch += metric_fn(y_batch, y_batch_pred)\
                                * X_batch.shape[0]
            total_train_samples += X_batch.shape[0]
        train_loss_epoch /= total_train_samples
        train_metric_epoch /= total_train_samples
        train_losses.append(train_loss_epoch)
        train_metrics.append(train_metric_epoch)

        # Validate
        # Note: We don't need to compute the gradients here
        val_loss_epoch, val_metric_epoch, total_val_samples = 0, 0, 0
        for i in range(num_val_batches):
            X_batch = X_val[i * batch_size:(i + 1) * batch_size]
            y_batch = y_val[i * batch_size:(i + 1) * batch_size]

            y_batch_pred = model(X_batch)
            loss = loss_fn(y_batch, y_batch_pred)

            val_loss_epoch += loss * X_batch.shape[0]
            val_metric_epoch += metric_fn(y_batch, y_batch_pred)\
                                * X_batch.shape[0]
            total_val_samples += X_batch.shape[0]
        val_loss_epoch /= total_val_samples
        val_metric_epoch /= total_val_samples
        val_losses.append(val_loss_epoch)
        val_metrics.append(val_metric_epoch)

        # TODO: Write Early Stopping code here based on the val_metrics (histories)
    train_losses = np.array(train_losses)
    train_metrics = np.array(train_metrics)
    val_losses = np.array(val_losses)
    val_metrics = np.array(val_metrics)
    return train_losses, train_metrics, val_losses, val_metrics

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
    X_train, y_train,X_val, y_val, X_test, y_test = split_data(X, y)

    # Create and train the model
    model = LinearModel(input_dim=X_train.shape[1])
    train_losses, train_metrics, val_losses, val_metrics =\
        train(model, X_train, y_train, X_val, y_val)

    # Plot the training and validation loss
    plot_metrics(train_losses, val_losses, 'Loss')
    plot_metrics(train_metrics, val_metrics, 'Metric')

    # Test the model
    test_loss, test_metric = get_performance(model, X_test, y_test, 'mse', 'mae')


    # Plot the fit
    plot_fit(X_train, y_train, model, kernel_mode=kernel_mode, partition='train')
    plot_fit(X_test, y_test, model, kernel_mode=kernel_mode, partition='test')


if __name__ == '__main__':
    main(kernel_mode=True)
