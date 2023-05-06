#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 02:47:16 2021

"""

import os
import sys
import numpy as np
import h5py
import torch
import torchvision
from cifar_train import ResNet18
from cifar_extract import load_model, inv_normalize, get_activations

if __name__ == '__main__':
    # DATASET = sys.argv[1]
    DATASET = 'CIFAR10'

    norm_mean = np.array((0.4914, 0.4822, 0.4465))
    norm_std = np.array((0.2023, 0.1994, 0.2010))
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(norm_mean.tolist(),
                                                                                  norm_std.tolist())])

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

    activation_dir = f'../activations/{DATASET.lower()}/resnet18_custom_aug/full_activations/'

    for label_filter in range(num_classes):
        activations = get_activations(net, trainloader, label_filter, is_sample=False, is_imgs=True)
        with h5py.File(os.path.join(activation_dir, f'label{label_filter}.hdf5'), 'w') as out_file:
            [out_file.create_dataset(layer_name, data=layer_act) for layer_name, layer_act in
             activations.items()]
        del activations

    