import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from gan1 import Generator
from classifier import Classifier
import itertools

# Use a function to load models to avoid repetition
generator = Generator(100)
generator.load_state_dict(torch.load('second_gan_model\generator_100.pth'))
generator.eval()


classifier = Classifier()
classifier.load_state_dict(torch.load('classifier_model\model-fold-3.pth'))
classifier.eval()

# Define constants
BATCH_SIZE = 1
LATENT_DIM = 100  # Assuming this is the latent dimension

# Transform and DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset=mnist_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Helper Functions
def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = {}
    for d in list_of_dicts:
        for key, value in d.items():
            dict_of_lists.setdefault(key, []).append(value)
    return dict_of_lists


def get_base_layers(model):
    # return [layer for layer in model.children() if isinstance(layer, nn.Sequential)]
    base_layers = list() 

    for layer in model.children(): 
        if isinstance(layer, nn.Sequential): 
            for child_layer in layer: 
                base_layers.append(child_layer)
        else: 
            base_layers.append(layer)

    return base_layers

def pass_through_model(x, layers):
    results = {}
    for i, layer in enumerate(layers):
        x = layer(x)
        results[f'layer_{i}'] = x
    return results

def mean_tensor_list(tensors):
    return [torch.mean(tensor, dim=0) for tensor in tensors]

def generate_samples_to_compare(sample1, sample2): 
    # Simplified using list comprehension and dict comprehension
    sample1_dict = {index: tensor[index].item() for tensor in sample1 for index in itertools.product(*[range(dim) for dim in tensor.shape])}
    sample2_dict = {index: tensor[index].item() for tensor in sample2 for index in itertools.product(*[range(dim) for dim in tensor.shape])}

    return {f'{key1}, {key2}': wasserstein_distance(dist1, dist2)
            for key1, dist1 in sample1_dict.items()
            for key2, dist2 in sample2_dict.items()}

# Main Loop
if __name__ == '__main__': 
    generator_layers = get_base_layers(generator)
    classifier_layers = get_base_layers(classifier)

    for epoch in range(1):
        gen_values, clas_values = [], []

        for i, (images, classes) in enumerate(data_loader):
            z = torch.randn(BATCH_SIZE, LATENT_DIM)

            gen_res = pass_through_model(z, generator_layers)
            gen_values.extend(mean_tensor_list(gen_res.values()))

            clas_res = pass_through_model(images, classifier_layers)
            clas_values.extend(mean_tensor_list(clas_res.values()))

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/1], Batch [{i + 1}/{len(data_loader)}]')

    print('Weights collected')

    # Further processing and comparisons
    
    generator_weights = list_of_dicts_to_dict_of_lists(gen_values)

    classifier_weights = list_of_dicts_to_dict_of_lists(clas_values)

    min_similarity = float('inf')
    most_similar_keys = {'gen_key': 'layer', 'clas_key': 'layer'}
    similarities_layers = {}

    for gen_key, gen_weights in generator_weights.items():
        for clas_key, clas_weights in classifier_weights.items(): 
            wasserstein_matrix = generate_samples_to_compare(gen_weights, clas_weights)
            print('hi')
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