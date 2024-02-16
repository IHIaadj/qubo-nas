import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataloaders import * 
from gnn_supernetwork import * 

def load_data(dataset_name):
    if dataset_name == 'CIFAR-10':
        batch_size = 64
    elif dataset_name == 'ImageNet':
        batch_size = 256
    else:
        raise ValueError("Unsupported dataset")

    trainloader, testloader = load_dataset(dataset_name, batch_size=batch_size)
    return trainloader, testloader

def initialize_model(dataset_name, num_layers, num_operations, num_graph_layers):
    gnn_supernetwork = GNNSupernetwork(num_layers, num_operations, num_graph_layers)
    return gnn_supernetwork

def train_model(model, trainloader, dataset_name):
    criterion = nn.CrossEntropyLoss()
    if dataset_name == 'CIFAR-10':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        epochs = 200
    elif dataset_name == 'ImageNet':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        epochs = 200

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

def validate_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total}%')


def train(dataset_name, model):
    trainloader, testloader = load_data(dataset_name)
    train_model(model, trainloader, dataset_name)
    validate_model(model, testloader)

# Example Usage
#main('CIFAR-10')
