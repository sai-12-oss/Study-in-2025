import numpy as np

from utils import tanh


class Layer:
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
    ) -> None:
        self.data = np.array([0.0])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights =\
            np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)
        self.bias = np.zeros(output_dim)
        self.grad_w = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.data = x
        return np.dot(x, self.weights) + self.bias

    def backward(
            self,
            upstream_grad: np.ndarray,
    ) -> np.ndarray:
        self.grad_w = self.data.T @ upstream_grad
        self.grad_b = upstream_grad.sum(axis=0)
        grad_x = upstream_grad @ self.weights.T
        return grad_x

    def update(
            self,
            lr: float,
    ) -> None:
        self.weights -= lr * self.grad_w
        self.bias -= lr * self.grad_b


class Network:
    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
    ) -> None:
        self.tanh_x = None
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer = Layer(num_inputs, num_hidden)
        self.output_layer = Layer(num_hidden, num_outputs)

    def forward(
            self,
            x: np.ndarray,
    ) -> np.ndarray:
        x = self.hidden_layer.forward(x)
        x = tanh(x)
        self.tanh_x = x
        x = self.output_layer.forward(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def zero_grad(self):
        self.hidden_layer.grad_w.fill(0)
        self.hidden_layer.grad_b.fill(0)
        self.output_layer.grad_w.fill(0)
        self.output_layer.grad_b.fill(0)

    def backward(
            self,
            grad_L: np.ndarray,
    ) -> None:
        # calculate gradient for output layer
        grad_o = self.output_layer.backward(grad_L)

        # calculate gradient for non-linearity
        grad_tanh = (1 - self.tanh_x ** 2) * grad_o

        # calculate gradient for hidden layer
        self.hidden_layer.backward(grad_tanh)

    def update(
            self,
            lr: float,
    ) -> None:
        self.hidden_layer.update(lr)
        self.output_layer.update(lr)
