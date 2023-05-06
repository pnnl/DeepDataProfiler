#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 01:42:33 2021

"""
import os
import sys
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
from cifar_extract import read_activation
from mapper_interactive.mapper_CLI import get_mapper_graph
from matplotlib import pyplot as plt



if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    layer = sys.argv[1]
    eps = float(sys.argv[2])
    interval = int(sys.argv[3])
    overlap = int(sys.argv[4])
    sigma = float(sys.argv[5])
    
    DATASET = 'CIFAR10'
    num_classes = 10

    activation_dir = f'../activations/{DATASET.lower()}/resnet18_custom_aug/full_activations_noises_imgs/'
    
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
    
    ### Gaussian noise parameters ###
    # mu = 0
    # sigma = 0.1
    
    print("collection full activations for", layer)
    layer_activations = []
    for i in range(num_classes):
        print(i)
        layer_activations_i = read_activation(os.path.join(activation_dir, f'label{i}_{sigma}.hdf5'), layer)
        layer_activations_i = torch.tensor(layer_activations_i)
        layer_activations_i = F.relu(layer_activations_i[:, :, :, :]).numpy()
        num_patches = layer_activations_i.shape[2]
        layer_activations_i_new = []
        for j in range(num_patches):
            for k in range(num_patches):
                layer_activations_i_new.append(layer_activations_i[:,:,j,k])
        layer_activations_i = np.vstack([l for l in layer_activations_i_new])
        # noise_i = np.random.normal(mu, sigma, layer_activations_i.shape)
        # layer_activations_i += noise_i
        label_i = np.repeat(i, len(layer_activations_i)).reshape(-1,1)
        layer_activations_i = np.hstack([label_i, layer_activations_i])
        layer_activations.append(layer_activations_i)
    
    layer_activations = np.vstack([layer_activations_i for layer_activations_i in layer_activations])
    layer_activations_df = pd.DataFrame(layer_activations)
    
    # df_std = layer_activations_df.iloc[:,1:].std(axis=0)
    # df_mean = layer_activations_df.iloc[:,1:].mean(axis=0)
    # print(df_std.shape)
    # print(df_mean.shape)
    # plt.hist(df_std)
    # plt.savefig('../figs/activations_std_full.png')
    # plt.close()
    # plt.hist(df_mean)
    # plt.savefig('../figs/activations_mean_full.png')
    # plt.close()
    
    cols = np.array(['label'])
    cols = np.concatenate((cols,np.arange(1,layer_activations.shape[1]).astype("str")))
    
    layer_activations_df.columns = cols 
    layer_activations_df['label'] = [names[int(layer_activations_df['label'].iloc[i])] for i in range(len(layer_activations_df))]
    print(layer_activations_df.shape)
            
    min_samples = 5
    # interval = 40
    # overlap = 25
    
    output_dir = '../mapper_graphs/full_batches_noises/'
    output_fname = 'full_batches_'+layer+'_'+str(sigma)
    
    get_mapper_graph(layer_activations_df, interval, overlap, eps, min_samples, output_dir, output_fname, is_parallel=False)
    
    

