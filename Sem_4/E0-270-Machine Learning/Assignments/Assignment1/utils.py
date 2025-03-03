import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from scipy import sparse as sp


class Vectorizer:
    """
    A vectorizer class that converts text data into a sparse matrix
    """
    def __init__(self, max_vocab_len=10_000) -> None:
        """
        Initialize the vectorizer
        """
        self.vocab = None
        # TODO: Add more class variables if needed

    def fit(self, X_train: np.ndarray) -> None:
        """
        Fit the vectorizer on the training data
        :param X_train: np.ndarray
            The training sentences
        :return: None
        """
        # TODO: count the occurrences of each word
        # TODO: sort the words based on frequency
        # TODO: retain the top 10k words
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> sp.csr_matrix:
        """
        Transform the input sentences into a sparse matrix based on the
        vocabulary obtained after fitting the vectorizer
        ! Do NOT return a dense matrix, as it will be too large to fit in memory
        :param X: np.ndarray
            Input sentences (can be either train, val or test)
        :return: sp.csr_matrix
            The sparse matrix representation of the input sentences
        """
        assert self.vocab is not None, "Vectorizer not fitted yet"
        # TODO: convert the input sentences into vectors
        raise NotImplementedError


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_data(
        path: str,
        seed: int
) -> Tuple[np.ndarray, np.ndarray, np.array, np.ndarray]:
    """
    Load twitter sentiment data from csv file and split into train, val and
    test set. Relabel the targets to -1 (for negative) and +1 (for positive).

    :param path: str
        The path to the csv file
    :param seed: int
        The random state for reproducibility
    :return:
        Tuple of numpy arrays - (data, labels) x (train, val) respectively
    """
    # load data
    df = pd.read_csv(path, encoding='utf-8')

    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # split into train, val and test set
    train_size = int(0.8 * len(df))  # ~1M for training, remaining ~250k for val
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    x_train, y_train =\
        train_df['stemmed_content'].values, train_df['target'].values
    x_val, y_val = val_df['stemmed_content'].values, val_df['target'].values
    return x_train, y_train, x_val, y_val
