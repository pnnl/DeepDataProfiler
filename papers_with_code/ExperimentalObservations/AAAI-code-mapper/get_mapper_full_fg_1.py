#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 07:38:06 2021

"""

import os
import sys
import numpy as np
import pandas as pd
from get_knn import elbow_eps
from mapper_interactive.mapper_CLI import get_mapper_graph

if __name__ == '__main__':
    layer = sys.argv[1]
    interval = int(sys.argv[2])
    overlap = int(sys.argv[3])
    df_dir = "../datasets/cifar10_full_fg_1_df/"
    
    print("collection activations for", layer)
    
    layer_activations_df = pd.read_csv(df_dir+"train_full_fg_1_"+layer+".csv")
    print(layer_activations_df.shape)

    try:
      eps = float(sys.argv[4])
    except:
      eps = elbow_eps(layer_activations_df.iloc[:, 1:])
    print("eps", eps)
            
    min_samples = 5
    
    output_dir = '../mapper_graphs/full_fg_1/'
    output_fname = 'full_fg_1_'+layer
    
    get_mapper_graph(layer_activations_df, interval, overlap, eps, min_samples, output_dir, output_fname)
    