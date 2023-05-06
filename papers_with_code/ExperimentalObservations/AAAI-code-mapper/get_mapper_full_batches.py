#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 02:41:48 2021

"""
import os
import sys
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
from mapper_interactive.mapper_CLI import get_mapper_graph
from get_knn import elbow_eps

def read_activation(filepath, layer):
    with h5py.File(filepath, 'r') as f:
        activation = f[layer][:]
        return activation

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    layer = sys.argv[1]
    interval = int(sys.argv[2])
    overlap = int(sys.argv[3])
    
    DATASET = 'CIFAR10'
    num_classes = 10

    activation_dir = f'../activations/{DATASET.lower()}/resnet18_custom_aug/full_activations'
    
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
    
    print("collection full activations for", layer)
    layer_activations = []
    for i in range(num_classes):
        print(i)
        layer_activations_i = read_activation(os.path.join(activation_dir, 'label'+str(i)+'.hdf5'), layer)
        layer_activations_i = torch.tensor(layer_activations_i)
        layer_activations_i = F.relu(layer_activations_i[:, :, :, :]).numpy()
        num_patches = layer_activations_i.shape[2]
        layer_activations_i_new = []
        for j in range(num_patches):
            for k in range(num_patches):
                layer_activations_i_new.append(layer_activations_i[:,:,j,k])
        layer_activations_i = np.vstack([l for l in layer_activations_i_new])
        label_i = np.repeat(i, len(layer_activations_i)).reshape(-1,1)
        layer_activations_i = np.hstack([label_i, layer_activations_i])
        layer_activations.append(layer_activations_i)
    
    layer_activations = np.vstack([layer_activations_i for layer_activations_i in layer_activations])
    layer_activations_df = pd.DataFrame(layer_activations)
    
    cols = np.array(['label'])
    cols = np.concatenate((cols,np.arange(1,layer_activations.shape[1]).astype("str")))
    
    layer_activations_df.columns = cols 
    layer_activations_df['label'] = [names[int(layer_activations_df['label'].iloc[i])] for i in range(len(layer_activations_df))]
    print(layer_activations_df.shape)

    if layer_activations_df.shape[0] > 800000:
        selected_indices = np.random.choice(layer_activations_df.shape[0], 800000)
        layer_activations_df = layer_activations_df.iloc[selected_indices, :]

    print(layer_activations_df.shape)

    try:
      eps = float(sys.argv[4])
    except:
      eps = elbow_eps(layer_activations_df.iloc[:, 1:])
    print("eps", eps)
            
    min_samples = 5
    
    output_dir = '../mapper_graphs/full_batches/'
    output_fname = 'full_batches_'+layer
    
    get_mapper_graph(layer_activations_df, interval, overlap, eps, min_samples, output_dir, output_fname, is_parallel=False)
    
    

