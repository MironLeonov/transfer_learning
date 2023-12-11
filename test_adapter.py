import torch 
import torch.nn as nn
from gan1 import Generator
from classifier import Classifier
from adapter import Adapter

generator = Generator(100)
generator.load_state_dict(torch.load('second_gan_model\generator_100.pth'))
generator.eval()


classifier = Classifier()
classifier.load_state_dict(torch.load('classifier_model\model-fold-3.pth'))
classifier.eval()

adapter = Adapter()
adapter.load_state_dict(torch.load('adapter\\adapter_final.pth'))

# w

def get_base_layers(model_children: list) -> list:

    base_layers = list() 

    for layer in model_children: 
        if isinstance(layer, nn.Sequential): 
            for child_layer in layer: 
                base_layers.append(child_layer)
        else: 
            base_layers.append(layer)

    return base_layers


def pass_value_throgh_generator(z, label, generator_base_layers): 
    x = generator_base_layers[0](label) #nn.Embedding(num_classes, 50)
    x = generator_base_layers[1](x) #nn.Linear(50 , 7 * 7)
    x = x.view((-1, 1, 7, 7))
    
    y = generator_base_layers[2](z) #nn.Linear(latent_dim, 128 * 7 * 7)
    y = generator_base_layers[3](y) #nn.LeakyReLU(0.2)
    y = y.view((-1, 128, 7, 7))

    # c 8го
    merge = torch.cat((y, x), dim=1)

    out = generator_base_layers[8](merge)

    out = generator_base_layers[9](out) #l_relu_1

    return out


def pass_value_throgh_classifier(image, classifier_base_layers): 

    x = classifier_base_layers[7](image)

    x = classifier_base_layers[8](x)

    x = classifier_base_layers[9](x)

    x = classifier_base_layers[10](x)

    x = classifier_base_layers[11](x) # nn.Linear(100, 10)

    out = classifier_base_layers[12](x) #nn.Softmax(dim=1)

    return out


if __name__ == '__main__': 

    generator_children = list(generator.children())

    generator_base_layers = get_base_layers(generator_children)


    classifier_children = list(classifier.children())

    classifier_base_layers = get_base_layers(classifier_children)

    z = torch.randn(1, 100)
    label = torch.tensor([9])

    generator_out = pass_value_throgh_generator(z, label, generator_base_layers)

    adapter_out = adapter(generator_out)

    classifier_out = pass_value_throgh_classifier(adapter_out, classifier_base_layers)

    print(classifier_out)

    _, predicted = torch.max(classifier_out.data, 1)
    print(predicted)