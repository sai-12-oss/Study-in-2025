import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hid_dim, stgs):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.stgs = stgs
        self.conv1 = nn.Conv2d(1, 6, 5, stride=2) # 6x12x12
        self.conv2 = nn.Conv2d(6, 16, 3, stride=2) # 16x5x5
        self.conv3 = nn.Conv2d(16, 2 * hid_dim, 3, stride=2) # 32x2x2
        self.conv4 = nn.Conv2d(2 * hid_dim, hid_dim, 2) # 16x1x1
        if not self.stgs:
            self.conv4b = nn.Conv2d(2 * hid_dim, hid_dim, 2) # 16x1x1
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        if not self.stgs:
            mu = self.conv4(x)
            logstd = self.conv4b(x)
            return mu, logstd
        else:
            x = self.conv4(x)
            return x.view(-1, self.hid_dim)


class Decoder(nn.Module):
    def __init__(self, hid_dim=16, cond=False):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.cond = cond
        self.conv1 = nn.ConvTranspose2d(
            hid_dim + (10 if cond else 0), 2 * hid_dim, 2)
        self.conv2 = nn.ConvTranspose2d(2 * hid_dim, 16, 3, stride=2)
        self.conv3 = nn.ConvTranspose2d(16, 6, 3, stride=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(6, 1, 5, stride=2, output_padding=1)
    
    def forward(self, z, y=None):
        assert self.cond == (y is not None), 'y should be provided for conditional generation'
        z = z.view(-1, self.hid_dim, 1, 1)
        if self.cond:
            # convert y to one-hot
            yz = torch.nn.functional.one_hot(y, num_classes=10).reshape(-1, 10, 1, 1)
            yz = yz.float().to(z.device)
            # concatenate z and y
            z = torch.cat([z, yz], dim=1)
        z = self.conv1(z)
        z = torch.relu(z)
        z = self.conv2(z)
        z = torch.relu(z)
        z = self.conv3(z)
        z = torch.relu(z)
        z = self.conv4(z)
        return torch.sigmoid(z)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=2) # 6x12x12
        self.conv2 = nn.Conv2d(6, 16, 3, stride=2) # 16x5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Network(nn.Module):
    def __init__(self, hid_dim=16, stgs=False, cond=False):
        super(Network, self).__init__()
        self.stgs = stgs
        self.cond = cond
        self.encoder = Encoder(hid_dim, stgs)
        self.decoder = Decoder(hid_dim, cond)

    def forward(self, x, y=None):
        assert self.cond == (y is not None), 'y should be provided for conditional generation'
        mu, logstd = self.encoder(x)
        z = self.sample(mu, logstd)
        out = self.decoder(z, y)
        return out, mu, logstd
    
    def encode(self, x):
        mu, logstd = self.encoder(x)
        return mu, logstd
    
    def sample(self, mu, logstd):
        std = torch.exp(logstd)
        z = mu + std * torch.randn_like(std)
        return z
    
    def decode(self, z, y=None):
        assert self.cond == (y is not None), 'y should be provided for conditional generation'
        out = self.decoder(z, y)
        return out
