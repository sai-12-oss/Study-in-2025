import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, hid_dim=16):
        super(Generator, self).__init__()
        self.hid_dim = hid_dim
        self.conv1 = nn.ConvTranspose2d(hid_dim, 2 * hid_dim, 2)
        self.bn1 = nn.BatchNorm2d(2 * hid_dim)
        self.conv2 = nn.ConvTranspose2d(2 * hid_dim, 200, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(200)
        self.conv3 = nn.ConvTranspose2d(200, 64, 3, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 1, 5, stride=2, output_padding=1)
    
    def forward(self, x):
        x = x.view(-1, self.hid_dim, 1, 1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.bn3(x)
        x = self.conv4(x)
        return torch.tanh(x)


class Discriminator(nn.Module):
    def __init__(self, hid_dim=16):
        super(Discriminator, self).__init__()
        self.hid_dim = hid_dim
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 2 * hid_dim, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(2 * hid_dim)
        self.conv4 = nn.Conv2d(2 * hid_dim, hid_dim, 2)
        self.fc = nn.Linear(hid_dim, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = x.view(-1, self.hid_dim)
        return self.fc(x)
