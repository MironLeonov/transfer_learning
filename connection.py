import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from gan1 import Generator
from classifier import Classifier


generator = Generator(100)
generator.load_state_dict(torch.load('second_gan_model\generator_100.pth'))
generator.eval()


classifier = Classifier()
classifier.load_state_dict(torch.load('classifier_model\model-fold-3.pth'))
classifier.eval()

batch_size = 1

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
    res = { 
        'conv2d_trans_1': 0,
        'l_relu_1': 0,
        'conv2d_trans_2': 0,
        'l_relu_2': 0,
        'conv2d_1': 0
    }

    x = generator_base_layers[0](label) #nn.Embedding(num_classes, 50)
    x = generator_base_layers[1](x) #nn.Linear(50 , 7 * 7)
    x = x.view((-1, 1, 7, 7))
    
    y = generator_base_layers[2](z) #nn.Linear(latent_dim, 128 * 7 * 7)
    y = generator_base_layers[3](y) #nn.LeakyReLU(0.2)
    y = y.view((-1, 128, 7, 7))

    # c 8го
    merge = torch.cat((y, x), dim=1)

    out = generator_base_layers[8](merge)
    res['conv2d_trans_1'] = out

    out = generator_base_layers[9](out)
    res['l_relu_1'] = out

    out = generator_base_layers[10](out)
    res['conv2d_trans_2'] = out

    out = generator_base_layers[11](out)
    res['l_relu_2'] = out

    out = generator_base_layers[12](out)
    res['conv2d_1'] = out

    return res


def pass_value_throgh_classifier(image, classifier_base_layers) -> dict: 

    res = { 
        'conv2d_1': 0,
        'relu_1': 0,
        'pool_2d_1': 0,
        'conv2d_2': 0,
        'relu_2': 0,
        'conv2d_3': 0,
        'relu_3': 0,
        'pool_2d_2': 0,
        'flat': 0,
        'linear_1': 0,
        'relu_4': 0 
    }

    x = classifier_base_layers[0](image)
    res['conv2d_1'] = x

    x = classifier_base_layers[1](x)
    res['relu_1'] = x

    x = classifier_base_layers[2](x)
    res['pool_2d_1'] = x

    x = classifier_base_layers[3](x)
    res['conv2d_2'] = x

    x = classifier_base_layers[4](x)
    res['relu_2'] = x

    x = classifier_base_layers[5](x)
    res['conv2d_3'] = x

    x = classifier_base_layers[6](x)
    res['relu_3'] = x

    x = classifier_base_layers[7](x)
    res['pool_2d_2'] = x

    x = classifier_base_layers[8](x)
    res['flat'] = x

    x = classifier_base_layers[9](x)
    res['linear_1'] = x

    x = classifier_base_layers[10](x)
    res['relu_4'] = x

    return res



if __name__   == '__main__': 

    generator_children = list(generator.children())

    generator_base_layers = get_base_layers(generator_children)


    classifier_children = list(classifier.children())

    classifier_base_layers = get_base_layers(classifier_children)

    num_epochs = 1

    for epoch in range(num_epochs):
        generator_values = list() 
        classifier_values = list()

        for i, data in enumerate(data_loader):
            real_images, real_classes = data


            z = torch.randn(batch_size, 100)

            generator_res_batch = pass_value_throgh_generator(z, real_classes, generator_base_layers)
            generator_values.append(generator_res_batch)

            classifier_res_batch = pass_value_throgh_classifier(real_images, classifier_base_layers)
            classifier_values.append(classifier_res_batch)


            
            break
        break

    max_similarity = 0
    most_similar_indexes = None

    for i, dict1 in enumerate(generator_values):
        for j, dict2 in enumerate(classifier_values):
            # print(dict1["conv2d_trans_1"].shape)
            # print(dict2["conv2d_1"].shape)

            print(dict1["conv2d_trans_1"].shape)
            print(dict2["conv2d_2"].shape)
            similarity = wasserstein_distance(dict1["conv2d_trans_1"].detach().numpy(), dict2["conv2d_1"].detach().numpy())
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_indexes = (i, j)

            break
        break

    print(similarity)




    ## LOSS DISCR 
    # E[f(x)] - E[f(y)]


    ## LOSS GER 
    # -E[f(x)]

    # Display the most similar dictionaries and their similarity scores
    # if most_similar_indexes is not None:
    #     index1, index2 = most_similar_indexes
    #     dict1 = list1[index1]
    #     dict2 = list2[index2]

    #     print(f"Most Similar Dictionaries (Indexes): {index1} and {index2}")
    #     print(f"Similarity Score (Wasserstein Distance): {max_similarity:.2f}")
    #     print(f"Dictionary 1: {dict1}")
    #     print(f"Dictionary 2: {dict2}")
    # else:
    #     print("No similar elements found.")


