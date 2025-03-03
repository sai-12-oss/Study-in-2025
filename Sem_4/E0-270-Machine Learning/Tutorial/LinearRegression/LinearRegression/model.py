import numpy as np


class LinearModel:
    '''
    Linear Model class.
    '''
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 1,
    ) -> None:
        '''
        args:
            input_dim: int, input dimension.
            output_dim: int, output dimension.

        Attributes:
            W: np.ndarray, weights.
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim)\
                * (1 / np.sqrt(input_dim))  # this is a vector when out_dim = 1

    def forward(self, X: np.ndarray) -> np.ndarray:
        '''
        Forward pass.

        args:
            X: np.ndarray, input data.

        return:
            y_pred: np.ndarray, predicted output.
        '''
        y_pred = X @ self.W  # Reference: Matrix Cookbook
        return y_pred

    def __call__(self, X: np.ndarray) -> np.ndarray:
        '''
        Forward pass.
        '''
        return self.forward(X)
