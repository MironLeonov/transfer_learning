import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
import os 


os.makedirs('classifier_model', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()

        self.fc_layers = nn.Sequential(
            nn.Linear(64*7*7, 100), 
            nn.ReLU(), 
            nn.Linear(100, 10), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


batch_size = 10
k_folds = 5
num_epochs = 5

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset_train_part = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataset_test_part = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

dataset = ConcatDataset([dataset_train_part, dataset_test_part])
  
kfold = KFold(n_splits=k_folds, shuffle=True)

# train_set, val_set = torch.utils.data.random_split(mnist_dataset, [50000, 10000])
# data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

if __name__ == '__main__': 
    # results = {}

    # for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)): 
    #     print(f"FOLD: {fold}")


    #     train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    #     test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
    #     trainloader = torch.utils.data.DataLoader(
    #                     dataset, 
    #                     batch_size=10, sampler=train_subsampler)
    #     testloader = torch.utils.data.DataLoader(
    #                     dataset,
    #                     batch_size=10, sampler=test_subsampler)
        
    #     model = Classifier().to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #     criterion = nn.CrossEntropyLoss()

    #     for epoch in range(0, num_epochs):

    #         print(f'Starting epoch {epoch+1}')

    #         current_loss = 0.0
    #         for i, data in enumerate(trainloader, 0):
                
    #             inputs, targets = data

    #             inputs = inputs.to(device)
    #             targets = targets.to(device)
                
    #             optimizer.zero_grad()
                
    #             outputs = model(inputs)
                
    #             loss = criterion(outputs, targets)
    #             loss.backward()

    #             optimizer.step()
                
    #             current_loss += loss.item()
    #             if i % 500 == 499:
    #                 print('Loss after mini-batch %5d: %.3f' %
    #                     (i + 1, current_loss / 500))
    #                 current_loss = 0.0
            
    #     print('Training process has finished. Saving trained model.')
            
    #     print('Starting testing')
        
    #     save_path = f'classifier_model/model-fold-{fold}.pth'
    #     torch.save(model.state_dict(), save_path)

    #     correct, total = 0, 0
    #     with torch.no_grad():
    #         for i, data in enumerate(testloader, 0):

    #             inputs, targets = data

    #             inputs = inputs.to(device)
    #             targets = targets.to(device)

    #             outputs = model(inputs)

    #             _, predicted = torch.max(outputs.data, 1)
    #             total += targets.size(0)
    #             correct += (predicted == targets).sum().item()

    #     print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
    #     print('--------------------------------')
    #     results[fold] = 100.0 * (correct / total)
    

    # print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    # print('--------------------------------')
    # sum = 0.0
    # for key, value in results.items():
    #     print(f'Fold {key}: {value} %')
    #     sum += value
    # print(f'Average: {sum/len(results.items())} %')
            

    # print(type(train_set))
    model = Classifier()
    data_loader = DataLoader(dataset=dataset_test_part, batch_size=batch_size, shuffle=True)

    for i, data in enumerate(data_loader): 
        real_images, real_classes = data
        print(real_images.shape)
        
        out = model(real_images)
        print(out.shape)
    #     print(out)

        break



# data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)
