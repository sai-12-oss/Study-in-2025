import torch
import torch.nn as nn
import torch.optim as optim

from common.utils import *
from common.train_utils import *


class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding="same")
        self.conv2 = nn.Conv2d(64, 128, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(128, 256, 3, 1, padding="same")
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))  # 14 x 14 x 64
        x = torch.relu(F.max_pool2d(self.conv2(x), 2))  # 7 x 7 x 128
        x = torch.relu(F.max_pool2d(self.conv3(x), 2))  # 3 x 3 x 256
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x


def main() -> None:
    # Load the data
    train_loader, test_loader = get_data('fashion_mnist', batch_size=64)

    # Create a model
    model = Net()
    print("Model Parameter Count:", sum(p.numel() for p in model.parameters()))

    # Create an optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    train(model, train_loader, optimizer, epochs=25)

    # Evaluate the model
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
