import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=True)

from gan1 import Generator

generator = Generator(100)
generator.load_state_dict(torch.load('transfer_learning\second_gan_model\generator_100.pth'))
generator.eval()

# Load your pre-trained generator model
# generator = YourGeneratorModel().to(device)
# generator.load_state_dict(torch.load("path_to_your_model.pth"))
# generator.eval()

# Function to generate images using your model
def generate_image(digit, generator):
    z = torch.randn(1, 100)
    return generator(z, torch.tensor([digit]))
    # Create noise vector and digit input
    # noise = torch.randn(1, noise_dim).to(device)
    # digit_input = torch.tensor([digit]).to(device)

    # Generate image
    # generated_image = generator(noise, digit_input)
    # return generated_image

    # Placeholder return, replace with the above code
    # return torch.randn(1, 1, 28, 28)  # Random noise image, replace with actual generation
fig, axes = plt.subplots(10, 2, figsize=(8, 40))

# axes[0, 0].set_title("Изображение из набора данных")
# axes[0, 1].set_title("Сгенерированные изображения")

# Compare original and generated images for each digit
for digit in range(10):
    # Find an original image of the current digit
    for image, label in mnist_loader:
        if label.item() == digit:
            original_image = image
            break

    # Generate an image
    generated_image = generate_image(digit, generator)

    # Plotting
    # Plot original image
    # Plot original image
    axes[digit, 0].imshow(original_image.squeeze(), cmap='gray')
    axes[digit, 0].set_ylabel(f"Digit {digit}")
    axes[digit, 0].axis('off')

    # Plot generated image
    axes[digit, 1].imshow(generated_image.squeeze().detach().numpy(), cmap='gray')
    axes[digit, 1].axis('off')

plt.tight_layout()
plt.show()