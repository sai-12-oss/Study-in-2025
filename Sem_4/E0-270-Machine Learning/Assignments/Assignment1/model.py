import numpy as np
from scipy import sparse as sp


class MultinomialNaiveBayes:
    """
    A Multinomial Naive Bayes model
    """
    def __init__(self, alpha=0.01) -> None:
        """
        Initialize the model
        :param alpha: float
            The Laplace smoothing factor (used to handle 0 probs)
            Hint: add this factor to the numerator and denominator
        """
        self.alpha = alpha
        self.priors = None
        self.means = None
        self.i = 0  # to keep track of the number of examples seen

    def fit(self, X: sp.csr_matrix, y: np.ndarray, update=False) -> None:
        """
        Fit the model on the training data
        :param X: sp.csr_matrix
            The training data
        :param y: np.ndarray
            The training labels
        :param update: bool
            Whether to the model is being updated with new data
            or trained from scratch
        :return: None
        """
        raise NotImplementedError
        self.i += X.shape[1]

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        """
        Predict the labels for the input data
        :param X: sp.csr_matrix
            The input data
        :return: np.ndarray
            The predicted labels
        """
        assert self.priors.shape[0] == self.means.shape[0]
        preds = []
        for i in range(X.shape[0]):
            raise NotImplementedError
        return np.array(preds)
