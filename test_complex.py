import torch 
from gan1 import Generator
from classifier import Classifier

generator = Generator(100)
generator.load_state_dict(torch.load('second_gan_model\generator_100.pth'))
generator.eval()


classifier = Classifier()
classifier.load_state_dict(torch.load('classifier_model\model-fold-3.pth'))
classifier.eval()

if __name__ == '__main__': 
    z = torch.randn(1, 100)
    label = torch.tensor([7])
    generated_image = generator(z, label).view(1, 28, 28)
    print(generated_image.shape)

    generated_image = torch.unsqueeze(generated_image, dim=0)
    print(generated_image.shape)

    out = classifier(generated_image)

    print(out)
    _, predicted = torch.max(out.data, 1)
    print(predicted)