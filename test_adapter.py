import random
import torch 
import torch.nn as nn
import numpy as np
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
    # for i in range(10): 
    num_exp = 1000
    
    classic_scores = list()
    adapter_scores = list()
    adapter_noise_scores = list()
    no_adapter_noise_scores = list()
    for j in range(10): 
        classic_score = 0
        adapter_score = 0
        adapter_noise_score = 0
        no_adapter_noise_score = 0
        for i in range(num_exp): 

            random_number = random.randint(1, 9)

            z = torch.randn(1, 100)
            label = torch.tensor([random_number])

            # ADAPTER
            generator_out = pass_value_throgh_generator(z, label, generator_base_layers)

            adapter_out = adapter(generator_out)

            classifier_out = pass_value_throgh_classifier(adapter_out, classifier_base_layers)

            _, adapter_predicted = torch.max(classifier_out.data, 1)

            #  ADAPTER NOISE
            z_test = torch.randn_like(generator_out)

            adapter_out = adapter(z_test)

            classifier_out = pass_value_throgh_classifier(adapter_out, classifier_base_layers)

            _, adapter_noise_predicted = torch.max(classifier_out.data, 1)

            # CLASSIC 
            generated_image = generator(z, label).view(1, 28, 28)

            generated_image = torch.unsqueeze(generated_image, dim=0)

            out = classifier(generated_image)

            _, classic_predicted = torch.max(out.data, 1)

            #NO ADAPTER NOISE 
            z_test = torch.randn_like(generated_image)
            out = classifier(z_test)

            _, no_adapter_noise_predicted = torch.max(out.data, 1)

            if classic_predicted == random_number: 
                classic_score += 1 
            if adapter_predicted == random_number: 
                adapter_score += 1
            if adapter_noise_predicted == random_number: 
                adapter_noise_score += 1
            if no_adapter_noise_predicted == random_number:
                no_adapter_noise_score += 1

        classic_scores.append(classic_score/num_exp)
        adapter_scores.append(adapter_score/num_exp)
        adapter_noise_scores.append(adapter_noise_score/num_exp)
        no_adapter_noise_scores.append(no_adapter_noise_score/num_exp) 

    print(f"Classic accuracy: {np.mean(classic_scores)} +- {np.std(classic_scores)}")
    print(f"Adapter accuracy: {np.mean(adapter_scores)} +- {np.std(adapter_scores)}")
    print(f"Adapter noise accuracy: {np.mean(adapter_noise_scores)} +- {np.std(adapter_noise_scores)}")
    print(f"No adapter noise accuracy: {np.mean(no_adapter_noise_scores)} +- {np.std(no_adapter_noise_scores)}")
