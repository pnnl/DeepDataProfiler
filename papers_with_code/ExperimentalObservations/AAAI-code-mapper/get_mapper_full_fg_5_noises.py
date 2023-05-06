#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:19:50 2021

"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
import collections
from functools import partial
from glob import glob
import torch
import torch.nn.functional as F
from pynndescent import NNDescent
from matplotlib import pyplot as plt
from mapper_interactive.mapper_CLI import get_mapper_graph

if __name__ == '__main__':
    layer = sys.argv[1]
    eps = float(sys.argv[2])
    interval = int(sys.argv[3])
    overlap = int(sys.argv[4])
    sigma = float(sys.argv[5])
    
    df_dir = "../datasets/cifar10_full_fg_5_df_noises/"
    
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
    
    print("collection activations for", layer)
    
    
    # layer_activations_df = pd.read_csv(df_dir+"train_full_fg_1_"+layer+".csv")
    layer_activations_df = pd.read_csv(df_dir+"train_full_fg_5_"+layer+"_"+str(sigma)+".csv")
    # df_std = layer_activations_df.std(axis=0)
    # df_mean = layer_activations_df.mean(axis=0)
    # mu = 0
    # sigma = 0.1
    # noises = np.random.normal(mu, sigma, layer_activations_df.iloc[:, 1:].shape)
    # layer_activations_df.iloc[:, 1:] += noises
    # print(df_std.shape)
    # print(df_mean.shape)
    # plt.hist(df_std)
    # plt.savefig('../figs/activations_std_fg5.png')
    # plt.close()
    # plt.hist(df_mean)
    # plt.savefig('../figs/activations_mean_fg5.png')
    # plt.close()
    print(layer_activations_df.shape)
            
    min_samples = 5
    # interval = 40
    # overlap = 30
    
    output_dir = '../mapper_graphs/full_fg_5_noises/'
    output_fname = 'full_fg_5_'+layer+'_'+str(sigma)
    
    get_mapper_graph(layer_activations_df, interval, overlap, eps, min_samples, output_dir, output_fname, is_parallel=False)
    