#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 23:35:51 2021

"""

import os
import sys
import numpy as np
import h5py
import torch
import torchvision
from cifar_train import ResNet18
from cifar_extract import load_model, inv_normalize, get_activations

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    # model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            # x = x.to(device=device)
            # y = y.to(device=device)
            # x = x.view(-1, 512)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
    
            print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
        

if __name__ == '__main__':
    mu = 0
    sigma = float(sys.argv[1])
    # DATASET = sys.argv[1]
    DATASET = 'CIFAR10'
    
    seed = np.random.randint(100)
    g = torch.Generator()
    g.manual_seed(seed)
    
    norm_mean = np.array((0.4914, 0.4822, 0.4465))
    norm_std = np.array((0.2023, 0.1994, 0.2010))
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(norm_mean.tolist(),
                                                                                  norm_std.tolist()),
                                                 AddGaussianNoise(mu, sigma)])

    if DATASET == 'CIFAR10':
        train = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, download=False,
                                             transform=transforms)
        test = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, download=False,
                                            transform=transforms)
        num_classes = 10
    if DATASET == 'CIFAR100':
        train = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=True, download=False,
                                              transform=transforms)
        test = torchvision.datasets.CIFAR100(root='../datasets/cifar100', train=False, download=False,
                                             transform=transforms)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=2048, num_workers=4)
    testloader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=2048, num_workers=4)

    net = load_model(f'../logs/{DATASET}_ResNet18_Custom_Aug/checkpoints/best.pth', num_classes)
    net.eval()
    
    print("checking training accuracy")
    check_accuracy(trainloader, net)
    print("checking test accuracy")
    check_accuracy(testloader, net)


    activation_dir = f'../activations/{DATASET.lower()}/resnet18_custom_aug/full_activations_noises_imgs/'
    
    if not os.path.isdir(activation_dir):
        os.makedirs(activation_dir)

    for label_filter in range(num_classes):
        activations = get_activations(net, trainloader, label_filter, is_sample=False, is_imgs=True)
        print(activations['predictions'].shape)
        with h5py.File(os.path.join(activation_dir, f'label{label_filter}_{sigma}.hdf5'), 'w') as out_file:
            [out_file.create_dataset(layer_name, data=layer_act) for layer_name, layer_act in
             activations.items()]
        del activations

    