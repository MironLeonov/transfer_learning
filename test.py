import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from gan1 import Generator


# model = torch.load('second_gan_model\generator_100.pth')
model = Generator(100)
model.load_state_dict(torch.load('second_gan_model\generator_100.pth'))
model.eval()

batch_size = 1

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    for i, data in enumerate(data_loader):
        real_images, real_classes = data
        print(real_images.shape)
        print(real_images[0][0])
        # plt.imshow(real_images[0][0].numpy(), cmap='gray')
        # plt.show()

        break 
    print('generate')
    z = torch.randn(1, 100)
    label = torch.tensor([3])
    generated_image = model(z, label).view(1, 28, 28)
    print(generated_image.shape)
    print(generated_image[0])
    # generated_image = 0.5 * (generated_image + 1).detach().numpy()
    # print(generated_image[0])
    # generated_image = generated_image.detach().numpy()
    # # print(generated_image)
    
    # plt.imshow(generated_image[0], cmap='gray')
    # plt.show()