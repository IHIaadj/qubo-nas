import torch 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import RandomHorizontalFlip, Compose, Normalize, ToTensor

import numpy as np
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
    
def load_dataset(dataset_name, batch_size, num_workers=2, cutout_length=16):
    if dataset_name == 'CIFAR-10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
            Cutout(cutout_length)
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data_cifar10', train=False, download=True, transform=transform)

    elif dataset_name == 'ImageNet':
        # Note: ImageNet loading requires the dataset to be downloaded manually
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
            Cutout(cutout_length)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = torchvision.datasets.ImageFolder(root='./data_imagenet/train', transform=train_transform)
        testset = torchvision.datasets.ImageFolder(root='./data_imagenet/val', transform=test_transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader
