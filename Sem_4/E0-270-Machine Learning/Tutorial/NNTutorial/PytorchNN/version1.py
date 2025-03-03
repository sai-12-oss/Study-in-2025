import torch
import torch.nn as nn
import torch.optim as optim

from common.utils import *
from common.train_utils import *


class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x


def main() -> None:
    # Load the data
    train_loader, test_loader = get_data('mnist', batch_size=64)

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
