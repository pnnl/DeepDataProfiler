#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 20:47:33 2021

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
from get_knn import elbow_eps
from mapper_interactive.mapper_CLI import get_mapper_graph

if __name__ == '__main__':
    layer = sys.argv[1]
    interval = int(sys.argv[2])
    overlap = int(sys.argv[3])
    df_dir = "../datasets/cifar10_single_batch_df/"
    
    print("collection single activations for", layer)
    
    layer_activations_df = pd.read_csv(df_dir+"train_single_batch_"+layer+".csv")
    print(layer_activations_df.shape)

    try:
      eps = float(sys.argv[4])
    except:
      eps = elbow_eps(layer_activations_df.iloc[:, 1:])
    print("eps", eps)
            
    min_samples = 5
    # interval = 40
    # overlap = 30
    
    output_dir = '../mapper_graphs/single_batches/'
    output_fname = 'single_batch_'+layer
    
    get_mapper_graph(layer_activations_df, interval, overlap, eps, min_samples, output_dir, output_fname)
    
    

