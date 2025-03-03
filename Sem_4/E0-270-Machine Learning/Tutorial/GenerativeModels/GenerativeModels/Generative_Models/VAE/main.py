import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import *
from model import Network


def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        #print(data.shape)
        # batch, 48, 48, 1
        reconstructions, mus, logvars = model(data)

        loss = (reconstructions - data).pow(2).sum() #MSE
        # regularization
        kl_divergence = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
        
        loss += kl_divergence
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.4f}'
            )
    model.eval()


def main(args: ArgsStorage):
    # Load data
    mnist_dataset = get_data()

    # Create model
    model = Network(hid_dim=args.latent_dim).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    random_input = torch.randn(64, args.latent_dim).to(args.device)
    
    # Sample
    with torch.no_grad():
        sample = model.decoder(random_input)
        save_image(
            sample.view(64, 1, 28, 28),
            f'results/sample_0.png', nrow=8)

    for epoch in range(1, args.epochs + 1):
        # Train
        dataloader = DataLoader(
            mnist_dataset, batch_size=args.batch_size, shuffle=True)
        train(model, args.device, dataloader, optimizer, epoch)

        # Sample
        with torch.no_grad():
            sample = model.decoder(random_input)
            save_image(
                sample.view(64, 1, 28, 28),
                f'results/sample_{epoch}.png', nrow=8)
        
        # Save model
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'models/model_{epoch}.pth')


if __name__ == '__main__':
    args = ArgsStorage({
        'gpu_id': 0,
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 30,
        'latent_dim': 16,
    })
    args.device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    main(args)
