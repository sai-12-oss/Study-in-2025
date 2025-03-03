# get data mnist data

import os
from torchvision import datasets, transforms


class ArgsStorage:
    def __init__(self, args):
        self.__dict__.update(args)


def get_data():
    data_path = os.path.join(os.path.dirname(__file__), '../PytorchNN/data')
    mnist_train = datasets.MNIST(
        data_path, train=True, download=False, transform=transforms.ToTensor())
    return mnist_train
    
