import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import *
from model import Generator, Discriminator


def train(models, device, train_loader, optimizers, epoch, log_interval=100):
    generator, discriminator = models
    generator_optimizer, discriminator_optimizer = optimizers
    generator.train()
    discriminator.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        real_data = 2 * data - 1

        # for i in range(3):
        real_logits = discriminator(real_data)
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))

        fake_data = generator(torch.randn(data.size(0), args.latent_dim).to(device))
        fake_logits = discriminator(fake_data)
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))

        disc_loss = real_loss + fake_loss
        
        discriminator_optimizer.zero_grad()
        disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
        discriminator_optimizer.step()

        fake_data = generator(torch.randn(data.size(0), args.latent_dim).to(device))
        fake_logits = discriminator(fake_data)    
        gen_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

        generator_optimizer.zero_grad()
        gen_loss.backward()
        generator_optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'G_Loss: {gen_loss.item():.4f}\t'
                f'D_Loss: {disc_loss.item():.4f}\t'
            )
    generator.eval()
    discriminator.eval()


def main(args: ArgsStorage):
    # Load data
    mnist_dataset = get_data()

    # Create model
    generator = Generator(args.latent_dim).to(args.device)
    discriminator = Discriminator(args.latent_dim).to(args.device)
    generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)

    random_input = torch.randn(64, args.latent_dim).to(args.device)
    
    # Sample
    with torch.no_grad():
        sample = (1 + generator(random_input)) / 2
        save_image(
            sample.view(64, 1, 28, 28), f'results/sample_0.png', nrow=8)

    for epoch in range(1, args.epochs + 1):
        # Train
        dataloader = DataLoader(
            mnist_dataset, batch_size=args.batch_size, shuffle=True)
        train(
            [generator, discriminator], args.device, dataloader,
            [generator_optimizer, discriminator_optimizer], epoch)

        # Sample
        with torch.no_grad():
            sample = (1 + generator(random_input)) / 2
            save_image(
                sample.view(64, 1, 28, 28),
                f'results/sample_{epoch}.png', nrow=8)
        
        # Save model
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f'models/model_{epoch}.pth')


if __name__ == '__main__':
    args = ArgsStorage({
        'gpu_id': 0,
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 3,
        'latent_dim': 50,
    })
    args.device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    main(args)
