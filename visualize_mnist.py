import torch
import torchvision
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)

# Create a list to store one example for each digit
examples = [None] * 10

# Iterate through the dataset and find an example for each digit
for image, label in mnist_train:
    if examples[label] is None:
        examples[label] = image
    if all(examples):  # Check if all digits have been found
        break

# Plot the images in a 2x5 grid
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, image in enumerate(examples):
    row, col = divmod(i, 5)
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].set_title(f'Цифра: {i}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
