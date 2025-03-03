import numpy as np

from utils import plot_decision_boundary


class Perceptron:
    def __init__(self, inp_dim: int, thresh: float = 0.0) -> None:
        self.inp_dim = inp_dim
        self.thresh = thresh
        self.weights = np.zeros(inp_dim)
        self.bias = 0.0
    
    def predict(self, x: np.ndarray) -> int:
        return 2 * int((np.dot(self.weights, x) + self.bias) >= self.thresh) - 1
    
    def __call__(self, x: np.ndarray):
        return self.predict(x)
    
    def update(self, x: np.ndarray, y: int, y_hat: int, lr: float = 0.2):
        # update weights and bias for -1/1 labels
        loss = y - y_hat
        self.weights += lr * loss * x  # update only when loss != 0
        self.bias += lr * loss  # update only when loss != 0


def train(
        model: Perceptron, inputs: np.ndarray, labels: np.ndarray,
        lr: float, iterations: int
):
    for i in range(iterations):
        # After every 10 iterations, print the predictions
        if i % 10 == 0:
            print(f"Epoch: {i}")
            for x, y in zip(inputs, labels):
                y_hat = model(x)
                print(f"Input: {x} Output: {y_hat} Ground Truth: {y}")
        idx = np.random.randint(inputs.shape[0])
        x, y = inputs[idx], labels[idx]
        y_hat = model(x)
        model.update(x, y, y_hat, lr)
    print(f"Final Predictions:")
    for x, y in zip(inputs, labels):
        y_hat = model(x)
        print(f"Input: {x} Output: {y_hat} Ground Truth: {y}")


def main() -> None:
    np.random.seed(2023)
    
    # Create a dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([
        -1,
        -1,
        1,
        1
    ])

    # Create a perceptron model
    model = Perceptron(inp_dim=X.shape[1], thresh=0.0)

    # Visualize the decision boundary before training
    plot_decision_boundary(model, X, y)
    
    # Train the model
    train(model, X, y, lr=0.1, iterations=100)

    # Visualize the decision boundary after training
    plot_decision_boundary(model, X, y)


if __name__ == "__main__":
    main()

