import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os 
import matplotlib.pyplot as pyplot


os.makedirs('demo_gan', exist_ok=True)
os.makedirs('demo_gan_model', exist_ok=True)
os.makedirs('demo_gan_losses', exist_ok=True)
            

file_losses = open('demo_gan_losses/losses.txt', 'w')
file_losses.write('Epoch,Batch,D_loss_real,D_loss_fake,G_loss\n')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()

        self.fc_label = nn.Sequential(
            nn.Embedding(num_classes, 50),  # Embedding layer for labels
            nn.Linear(50 , 28*28)
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size = 3, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
        )

        self.post_net = nn.Sequential(
            nn.Flatten(), 
            nn.Dropout(0.4), 
            nn.Linear(128 * 7 * 7, 1), 
            nn.Sigmoid()
        )
    
    def forward(self, image,  label): 
        label = self.fc_label(label).view(-1, 1, 28, 28)
        merge = torch.cat((image, label), dim = 1)
        out = self.conv_layers(merge)
        out = self.post_net(out)
        return out

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.fc_label = nn.Sequential(
            nn.Embedding(num_classes, 50),  # Embedding layer for labels
            nn.Linear(50 , 7 * 7),
        )

        self.fc_z = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7), 
            nn.LeakyReLU(0.2),
        )

        #these layers are not used
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(129, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.post_net = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=7, padding=3)
        )

    def forward(self, z, label):
        label = self.fc_label(label)
        label = label.view((-1, 1, 7, 7))

       
        z = self.fc_z(z)
        z = z.view((-1, 128, 7, 7))

        merge = torch.cat((z, label), dim=1)

        out = self.conv_layers(merge)
        out = self.post_net(out)

        return out

discriminator = Discriminator().to(device)
generator = Generator(100).to(device)
num_classes = 10
latent_dim = 100
learning_rate = 0.0002
betas = (0.5, 0.999)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas = betas)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas = betas)


batch_size = 128

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 100

if __name__ == '__main__':

    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            real_images, real_classes = data

            real_images = real_images.to(device)
            real_classes = real_classes.to(device)

            # Train conditional discriminator with real images
            optimizer_D.zero_grad()
            real_outputs = discriminator(real_images, real_classes)

            real_labels = torch.ones(real_images.size(0), 1).to(device)
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_real.backward()

            # Train conditional discriminator with fake images
            z = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_classes = torch.randint(0, num_classes, (real_images.size(0),)).to(device)  # Random fake labels
            fake_images = generator(z, fake_classes)
            fake_outputs = discriminator(fake_images.detach(), fake_classes)

            fake_labels = torch.zeros(real_images.size(0), 1).to(device)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()

            # Update conditional discriminator weights
            optimizer_D.step()

            # Train generator
            optimizer_G.zero_grad()
            z = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_classes = torch.randint(0, num_classes, (real_images.size(0),)).to(device)  # Random fake labels
            fake_images = generator(z, fake_classes)

            fake_outputs = discriminator(fake_images, fake_classes)
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()

            # Update generator weights
            optimizer_G.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(data_loader)}], '
                    f'D_loss_real: {d_loss_real.item():.4f}, D_loss_fake: {d_loss_fake.item():.4f}, G_loss: {g_loss.item():.4f}')
                file_losses.write(f'{epoch}, {i}, {d_loss_real.item():.4f}, {d_loss_fake.item():.4f}, {g_loss.item():.4f}\n')

        # Generate and save a sample of fake images
        if (epoch + 1) % 10 == 0:
        # if epoch:
            with torch.no_grad():
                plt.figure(figsize=(10, 1))
                for digit in range(10):
                    z = torch.randn(1, latent_dim).to(device)
                    label = torch.tensor([digit]).to(device)
                    generated_image = generator(z, label).view(1, 28, 28)
                    generated_image = 0.5 * (generated_image + 1).cpu().numpy()
                    
                    plt.subplot(1, 10, digit + 1)
                    plt.imshow(generated_image[0], cmap='gray')
                    plt.axis('off')
                    plt.title(f'Digit {digit}')
                plt.savefig(f'demo_gan/gan_generated_epoch_{epoch + 1}.png')
                plt.close()

                torch.save(generator.state_dict(), 'demo_gan_model/generator_{epoch + 1}.pth')

    file_losses.close()