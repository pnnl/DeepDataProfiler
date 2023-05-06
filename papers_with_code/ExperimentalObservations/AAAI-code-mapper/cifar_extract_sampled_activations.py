#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 03:13:25 2021

"""

import os
import sys
import h5py
import torch
import torchvision
import numpy as np
import pandas as pd
from glob import glob
from cifar_train import ResNet18
from cifar_extract import load_model, inv_normalize, get_activations, read_activation

if __name__ == '__main__':
    DATASET = "CIFAR10"

    norm_mean = np.array((0.4914, 0.4822, 0.4465))
    norm_std = np.array((0.2023, 0.1994, 0.2010))
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(norm_mean.tolist(),
                                                                                  norm_std.tolist())])

    train = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, download=False,
                                             transform=transforms)
    num_classes = 10
    
    trainloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=2048, num_workers=4)

    net = load_model(f'../logs/{DATASET}_ResNet18_Custom_Aug/checkpoints/best.pth', num_classes)
    net.eval()

    activation_dir = f'../activations/{DATASET.lower()}/resnet18_custom_aug/sampled_activations/'
    output_dir = '../datasets/cifar10_single_batch_df/'
    
    if not os.path.isdir(activation_dir):
        os.makedirs(activation_dir)
        
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for label_filter in range(num_classes):
        activations = get_activations(net, trainloader, label_filter, is_sample=True)
        with h5py.File(os.path.join(activation_dir, f'label{label_filter}.hdf5'), 'w') as out_file:
            [out_file.create_dataset(layer_name, data=layer_act) for layer_name, layer_act in
             activations.items()]
        del activations
        
    all_files = glob(os.path.join(activation_dir, '*.hdf5'))
    
    names = ['airplane',
      'automobile',
      'bird',
      'cat',
      'deer',
      'dog',
      'frog',
      'horse',
      'ship',
      'truck']
    
    layers_name = ['layer4.1.bn2', 'layer4.1.bn1', 'layer4.0.bn2', 'layer4.0.bn1', 'layer3.1.bn2', 'layer3.1.bn1',
                   'layer3.0.bn2', 'layer3.0.bn1', 'layer2.1.bn2', 'layer2.1.bn1', 'layer2.0.bn2', 'layer2.0.bn1',
                   'layer1.1.bn2', 'layer1.1.bn1', 'layer1.0.bn2', 'layer1.0.bn1']
        
    
    for idx in range(len(layers_name)):
        print("collection sampled activations for", layers_name[idx])
        
        layer = layers_name[idx]
    
        layer_activations = []
        for i in range(num_classes):
            print(i)
            layer_activations_i = read_activation(os.path.join(activation_dir, 'label'+str(i)+'.hdf5'), layer)
            num_dims = layer_activations_i.shape[1]
            # predictions_i = read_activation(os.path.join(activation_dir, 'label'+str(i)+'.hdf5'), "predictions")
            # predictions_i = np.array(['true' if p == i else 'false' for p in predictions_i]).reshape(-1,1)
            label_i = np.repeat(i, len(layer_activations_i)).reshape(-1,1)
            # layer_activations_i = np.hstack([label_i, predictions_i, layer_activations_i])
            layer_activations_i = np.hstack([label_i, layer_activations_i])
            layer_activations.append(layer_activations_i)
        
        layer_activations = np.vstack([layer_activations_i for layer_activations_i in layer_activations])
        layer_activations_df = pd.DataFrame(layer_activations)
    
        # cols = np.array(['label', 'predictions'])
        cols = np.array(['label'])
        cols = np.concatenate((cols,np.arange(1,num_dims+1).astype("str")))
    
        layer_activations_df.columns = cols 
        layer_activations_df['label'] = [names[int(layer_activations_df['label'].iloc[i])] for i in range(len(layer_activations_df))]
        print(layer_activations_df.shape)
        # print(layer_activations_df.iloc[:20, :20])
        
        # print("prediction accuracy:", np.sum(layer_activations_df['predictions']=='true')/len(layer_activations_df['predictions']))
        layer_activations_df.to_csv(output_dir+"train_single_batch_"+layer+".csv", index=False)
    
    
    