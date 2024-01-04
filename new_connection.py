import itertools
import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from gan1 import Generator
from classifier import Classifier

import functools


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

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = {}
    for d in list_of_dicts:
        for key, value in d.items():
            dict_of_lists.setdefault(key, []).append(value)
    return dict_of_lists

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
        'pool_2d_2': 0
        # 'flat': 0,
        # 'linear_1': 0,
        # 'relu_4': 0 
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

    # x = classifier_base_layers[8](x)
    # res['flat'] = x

    # x = classifier_base_layers[9](x)
    # res['linear_1'] = x

    # x = classifier_base_layers[10](x)
    # res['relu_4'] = x

    return res



def calculate_mean_weights(weights: list):
    keys = weights[0].keys() 
    mean_dict = {}

    for key in keys:
        stacked_tensor = torch.stack([d[key] for d in weights])
        
        mean_tensor = torch.mean(stacked_tensor, dim=0)
        
        mean_dict[key] = mean_tensor

    return mean_dict

def generate_sample_from_list_of_tensors(tensors): 
    if not all(tensor.shape == tensors[0].shape for tensor in tensors):
        raise ValueError("All tensors must have the same shape.")

    tensor_shape = tensors[0].shape

    index_dict = {}

    for index in itertools.product(*[range(dim) for dim in tensor_shape]):
        index_dict[index] = [tensor[index].item() for tensor in tensors]

    return index_dict


def generate_samples_to_compare(sample1, sample2): 
    res = dict()

    sample1_index_dict = generate_sample_from_list_of_tensors(sample1)
    sample2_index_dict = generate_sample_from_list_of_tensors(sample2)

    for key1, dist1 in sample1_index_dict.items():
        for key2, dist2 in sample2_index_dict.items():
            # Calculate the Wasserstein distance
            distance = wasserstein_distance(dist1, dist2)

            res[f'{key1}, {key2}'] = distance

    return res

def calculate_similarity_with_rnd_samples(smaller_tensor, rnd_samples): 
    similarities = list()

    for sample in rnd_samples: 
        similarities.append(wasserstein_distance(smaller_tensor.detach().numpy(), sample.detach().numpy()))


    return functools.reduce(lambda a, b: a + b, similarities) / len(similarities) 
    

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

            for key, value in generator_res_batch.items(): 
                generator_res_batch[key] = torch.mean(generator_res_batch[key], dim = 0)
                generator_res_batch[key] = torch.mean(generator_res_batch[key], dim = 0)
            generator_values.append(generator_res_batch)

            classifier_res_batch = pass_value_throgh_classifier(real_images, classifier_base_layers)
            for key, value in classifier_res_batch.items(): 
                classifier_res_batch[key] = torch.mean(classifier_res_batch[key], dim = 0)
                classifier_res_batch[key] = torch.mean(classifier_res_batch[key], dim = 0)
            classifier_values.append(classifier_res_batch)

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(data_loader)}]')
            if i + 1 == 300: #mem out))
                break 
            if i > 10: 
                break
        break 

    print('weights collected')

    generator_weights = list_of_dicts_to_dict_of_lists(generator_values)

    print(f"total len: {len(generator_weights['conv2d_trans_1'])}")

    classifier_weights = list_of_dicts_to_dict_of_lists(classifier_values)

    min_similarity = float('inf')
    most_similar_keys = {'gen_key': 'layer', 'clas_key': 'layer'}
    similarities_layers = {}

    f = open('matrix.txt', 'w')

    for gen_key, gen_weights in generator_weights.items():
        for clas_key, clas_weights in classifier_weights.items(): 
            wasserstein_matrix = generate_samples_to_compare(gen_weights, clas_weights)
            f.write(f'\ngen: {gen_key}, clas: {clas_key}\n')
            f.write(str(wasserstein_matrix))
            similarity = sum(wasserstein_matrix.values()) / len(wasserstein_matrix)
            similarities_layers[f'gen: {gen_key}, clas: {clas_key}'] = similarity
            if similarity < min_similarity: 
                min_similarity = similarity
                most_similar_keys['gen_key'] = gen_key
                most_similar_keys['clas_key'] = clas_key
            elif similarity == min_similarity: 
                print('EQUAL!!!')
                print(gen_key, clas_key)
        print(f'Done {gen_key}')
    
    print(min_similarity)
    print(most_similar_keys)


    for k,v in sorted(similarities_layers.items(), key=lambda p:p[1]):
        print(k,v)
    
    f.close()