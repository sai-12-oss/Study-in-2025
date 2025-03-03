import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm


def train_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)):
        output = model(data)
        loss = F.nll_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_correct += (output.argmax(1) == target).sum().item()
        total_samples += data.size(0)
    return total_loss / total_samples, total_correct / total_samples


def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
                enumerate(data_loader), total=len(data_loader), desc="Testing"):
            output = model(data)
            loss = F.nll_loss(output, target)

            total_loss += loss.item() * data.size(0)
            total_correct += (output.argmax(1) == target).sum().item()
            total_samples += data.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
        epochs: int = 10,
) -> None:
    print("Training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, data_loader, optimizer)
        print(f"Epoch {epoch + 1} / {epochs} | " +
              f"Train Loss: {train_loss:.4f} | " +
              f"Train Acc: {train_acc:.4f}")
    print("Training complete!")
