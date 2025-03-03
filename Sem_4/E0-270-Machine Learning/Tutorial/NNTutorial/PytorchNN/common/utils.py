from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data(
        dataset_name: str,
        batch_size: int = 32,
        transform: transforms = transforms.ToTensor(),
) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == 'mnist':
        train_data = datasets.MNIST(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset_name == 'fashion_mnist':
        train_data = datasets.FashionMNIST(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        train_data = datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform)
    else:
        raise ValueError('Dataset not supported')

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, test_loader
