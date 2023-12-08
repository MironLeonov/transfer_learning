import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os 

os.makedirs('first', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the generator network
class Generator(nn.Module):
    # def __init__(self, latent_dim=100, image_dim=784, num_classes=10):
    #     super(Generator, self).__init__()
    #     self.latent_dim = latent_dim
    #     self.num_classes = num_classes
        
    #     # Embedding layer to convert digit labels to embeddings
    #     self.embedding = nn.Embedding(num_classes, latent_dim)

    #     self.fc = nn.Sequential(
    #         nn.Linear(latent_dim * 2, 256),
    #         nn.LeakyReLU(0.2),
    #         nn.Linear(256, 512),
    #         nn.LeakyReLU(),
    #         nn.Linear(512, 1024),
    #         nn.ReLU(),
    #         nn.Linear(1024, 2048),
    #         nn.ReLU(),
    #         nn.Linear(2048, image_dim),
    #         nn.Tanh()
    #     )

    # def forward(self, z, labels):
    #     # Generate an embedding for the desired digit label
    #     label_embedding = self.embedding(labels)
    #     # print(label_embedding.shape)
    #     # print(z.shape)
        
    #     # Concatenate the latent vector and label embedding
    #     z = torch.cat((z, label_embedding), dim=1)
        
    #     return self.fc(z)

    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding layer to convert digit labels to embeddings
        self.embedding = nn.Embedding(num_classes, latent_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 128 * 7 * 7),  # Adjust output size based on desired image size
            nn.LeakyReLU(0.2),
        )
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Generate an embedding for the desired digit label
        label_embedding = self.embedding(labels)
        
        # Concatenate the latent vector and label embedding
        z = torch.cat((z, label_embedding), dim=1)
        
        z = self.fc(z)
        z = z.view(z.size(0), 128, 7, 7)  # Reshape to match the deconvolutional layers
        z = self.deconv_layers(z)
        z = z.view(z.size(0) , -1)
        
        # return self.deconv_layers(z)
        return z

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, image_dim=784):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
    
class ConditionalDiscriminator(nn.Module):
    def __init__(self, image_dim=784, num_classes=10):
        super(ConditionalDiscriminator, self).__init__()
        self.fc_image = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )
        self.fc_label = nn.Sequential(
            nn.Embedding(num_classes, 128),  # Embedding layer for labels
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + 128, 1),
            nn.Sigmoid()
        )

    def forward(self, image, label):
        # print(image.shape)
        image = self.fc_image(image)
        # print(image.shape)
        label = self.fc_label(label)
        # print(label.shape)
        label = label.view(label.size(0), -1)  # Ensure label has the right shape
        # print(label.shape)
        combined = torch.cat((image, label), dim=1)
        # print(combined.shape)
        return self.fc_combined(combined)


# Hyperparameters
batch_size = 64
latent_dim = 100
image_dim = 28 * 28
num_epochs = 400
learning_rate = 0.0002
learning_rate_g = 0.0002
num_classes = 10

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)

# Initialize generator and discriminator
generator = Generator(latent_dim, image_dim).to(device)
discriminator = Discriminator(image_dim).to(device)
conditional_discriminator = ConditionalDiscriminator(image_dim=image_dim, num_classes=num_classes).to(device)
optimizer_conditional_D = optim.Adam(conditional_discriminator.parameters(), lr=learning_rate, weight_decay=0.0001)

# Define loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_g)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)



if __name__ == '__main__': 
    # print(data_loader[0])
    # Training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            real_images, real_labels = data
            # print(real_images.shape)
            real_images = real_images.view(-1, image_dim).to(device)
            real_labels = real_labels.to(device)
            # print(real_labels)

            # Train conditional discriminator with real images
            optimizer_conditional_D.zero_grad()
            real_outputs = conditional_discriminator(real_images, real_labels)
            # print(real_outputs)
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_real.backward()

            # Train conditional discriminator with fake images
            z = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_labels = torch.randint(0, num_classes, (real_images.size(0),)).to(device)  # Random fake labels
            # fake_labels = fake_labels.long()  # Convert fake_labels to Long data type
            # fake_labels = fake_labels.view(-1) 
            fake_images = generator(z, fake_labels)
            # print(fake_images.shape)
            # fake_labels = torch.zeros(batch_size, 1).to(device)
            # fake_labels = fake_labels.long()  # Convert fake_labels to Long data type
            # fake_labels = fake_labels.view(-1) 
            fake_outputs = conditional_discriminator(fake_images.detach(), fake_labels)
            # print(fake_outputs)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()

            # Update conditional discriminator weights
            optimizer_conditional_D.step()

            # Train generator
            optimizer_G.zero_grad()
            z = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_labels = torch.randint(0, num_classes, (real_images.size(0),)).to(device)  # Random fake labels
            fake_images = generator(z, fake_labels)
            fake_outputs = conditional_discriminator(fake_images, fake_labels)
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()

            # Update generator weights
            optimizer_G.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(data_loader)}], '
                    f'D_loss_real: {d_loss_real.item():.4f}, D_loss_fake: {d_loss_fake.item():.4f}, G_loss: {g_loss.item():.4f}')

        # Generate and save a sample of fake images
        # if (epoch + 1) % 10 == 0:
            # with torch.no_grad():
            #     z = torch.randn(16, latent_dim).to(device)
            #     print(z)
            #     fake_images = generator(z).to('cpu').detach().reshape(-1, 1, 28, 28)
            #     fake_images = 0.5 * (fake_images + 1)
            #     plt.figure(figsize=(4, 4))
            #     for j in range(16):
            #         plt.subplot(4, 4, j + 1)
            #         plt.imshow(fake_images[j][0], cmap='gray')
            #         plt.axis('off')
            #     plt.savefig(f'gan_generated_epoch_{epoch + 1}.png')
            #     plt.close()
        if (epoch + 1) % 10 == 0:
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
                plt.savefig(f'first/gan_generated_epoch_{epoch + 1}.png')
                plt.close()
            # with torch.no_grad():
            #     plt.figure(figsize=(10, 1))
            #     generated_images = []
                
            #     for digit in range(10):
            #         z = torch.randn(1, latent_dim).to(device)
            #         label = torch.tensor([digit]).to(device)
            #         generated_image = generator(z, label).view(1, 28, 28)
            #         generated_image = 0.5 * (generated_image + 1).cpu().numpy()
            #         generated_images.append(generated_image)
                
            #     generated_images = np.concatenate(generated_images, axis=2)
                
            #     plt.imshow(generated_images[0], cmap='gray')
            #     plt.axis('off')
            #     plt.show()

    # Save the trained generator model
    torch.save(generator.state_dict(), 'generator.pth')
