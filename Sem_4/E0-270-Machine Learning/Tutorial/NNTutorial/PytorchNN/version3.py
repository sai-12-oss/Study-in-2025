import torch
import torch.nn as nn
import torch.optim as optim

from common.utils import *
from common.train_utils import *


class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 16, 5, 1)
        self.fc1 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))  # 12 x 12 x 16
        x = torch.relu(F.max_pool2d(self.conv2(x), 2))  # 4 x 4 x 16
        x = x.view(x.size(0), -1)
        x = torch.log_softmax(self.fc1(x), dim=1)
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
