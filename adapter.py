import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from gan1 import Generator
from classifier import Classifier

import os

import functools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs('adapter', exist_ok=True)
os.makedirs('critic', exist_ok=True)


class Adapter(nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()

        # Initial convolution layer
        self.initial_conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm2d(64)
        self.initial_relu = nn.ReLU(inplace=True)

        # Additional convolution layers
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
       
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)

        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, kernel_size=4, stride=2, padding=1)
        )
        self.fc = nn.Linear(3 * 3, 1)

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = x.view(-1, 3 * 3)
        x = self.fc(x)
        return x


generator = Generator(100)
generator.load_state_dict(torch.load('second_gan_model\generator_100.pth'))
generator.eval()


classifier = Classifier()
classifier.load_state_dict(torch.load('classifier_model\model-fold-3.pth'))
classifier.eval()

adapter = Adapter().to(device)
critic = Critic().to(device)

lr = 1e-4
lambda_gp = 10
n_epochs = 100 

# critic_optimizer = optim.RMSprop(critic.parameters(), lr=lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr, betas = (0.0, 0.9))

# adapter_optimizer = optim.RMSprop(adapter.parameters(), lr=lr)
adapter_optimizer = optim.Adam(adapter.parameters(), lr=lr, betas = (0.0, 0.9))

batch_size = 100

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)

def get_base_layers(model_children: list) -> list:

    base_layers = list() 

    for layer in model_children: 
        if isinstance(layer, nn.Sequential): 
            for child_layer in layer: 
                base_layers.append(child_layer)
        else: 
            base_layers.append(layer)

    return base_layers


def pass_value_throgh_generator(z, label, generator_base_layers) -> dict: 
    x = generator_base_layers[0](label) #nn.Embedding(num_classes, 50)
    x = generator_base_layers[1](x) #nn.Linear(50 , 7 * 7)
    x = x.view((-1, 1, 7, 7))
    
    y = generator_base_layers[2](z) #nn.Linear(latent_dim, 128 * 7 * 7)
    y = generator_base_layers[3](y) #nn.LeakyReLU(0.2)
    y = y.view((-1, 128, 7, 7))

    # c 8го
    merge = torch.cat((y, x), dim=1)

    out = generator_base_layers[8](merge)
    # res['conv2d_trans_1'] = out

    out = generator_base_layers[9](out) #l_relu_1

    return out


def pass_value_throgh_classifier(image, classifier_base_layers) -> dict: 

    x = classifier_base_layers[0](image)
    # res['conv2d_1'] = x

    x = classifier_base_layers[1](x)
    # res['relu_1'] = x

    x = classifier_base_layers[2](x)
    # res['pool_2d_1'] = x

    x = classifier_base_layers[3](x)
    # res['conv2d_2'] = x

    x = classifier_base_layers[4](x)
    # res['relu_2'] = x

    x = classifier_base_layers[5](x)
    # res['conv2d_3'] = x

    x = classifier_base_layers[6](x)
    # res['relu_3'] = x

    return x





if __name__   == '__main__': 

    generator_children = list(generator.children())

    generator_base_layers = get_base_layers(generator_children)


    classifier_children = list(classifier.children())

    classifier_base_layers = get_base_layers(classifier_children)

    for epoch in range(n_epochs):
        generator_values = list() 
        classifier_values = list()

        for batch_idx, data in enumerate(data_loader):

            with torch.no_grad(): 
                real_images, real_classes = data

                z = torch.randn(batch_size, 100)

                generator_out =  pass_value_throgh_generator(z, real_classes, generator_base_layers)

                classifier_in = pass_value_throgh_classifier(real_images, classifier_base_layers)

            real_data = classifier_in.to(device)
            batch_size = real_data.size(0)
            
            critic.zero_grad()
            real_output = critic(real_data)
            
            # Generate fake data
            z = generator_out.to(device)
            fake_data = adapter(z)
            # print(fake_data.shape)
            fake_output = critic(fake_data.detach())
             
            # Compute the Wasserstein distance (negative loss)
            critic_loss = -(torch.mean(real_output) - torch.mean(fake_output))
            
            # Compute the gradient penalty
            alpha = torch.rand(batch_size, 1, 1, 1).to(device)
            interpolates = (alpha * real_data + (1 - alpha) * fake_data.detach()).requires_grad_(True)
            d_interpolates = critic(interpolates)
            gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                            grad_outputs=torch.ones(d_interpolates.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            # Update the critic loss with gradient penalty
            critic_loss += gradient_penalty
            critic_loss.backward(retain_graph=True)
            critic_optimizer.step()
            
            # Train the adapter
            if batch_idx % 5 == 0:
                adapter.zero_grad()
                fake_output = critic(fake_data)
                adapter_loss = -torch.mean(fake_output)
                adapter_loss.backward()
                adapter_optimizer.step()
            
            # Print training progress
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{n_epochs}] Batch [{batch_idx}/{len(data_loader)}] "
                    f"critic Loss: {critic_loss.item()} "
                    f"adapter Loss: {adapter_loss.item()}")
            

        if epoch % 10 == 0: 
            torch.save(adapter.state_dict(), f'adapter/adapter_{epoch}.pth')
            torch.save(critic.state_dict(), f'critic/critic_{epoch}.pth')

    torch.save(adapter.state_dict(), 'adapter/adapter_final.pth')
    torch.save(critic.state_dict(), 'critic/critic_final.pth')
