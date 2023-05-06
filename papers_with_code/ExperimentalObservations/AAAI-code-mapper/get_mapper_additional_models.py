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
    model = sys.argv[1]
    layer = sys.argv[2]
    interval = int(sys.argv[3])
    overlap = int(sys.argv[4])

    df_dir = os.path.join("../activations/additional_models/", model)
    
    print("collection activations for", model, layer)
    
    layer_activations_df = pd.read_csv(os.path.join(df_dir, layer+".csv"))
    print(layer_activations_df.shape)

    try:
      eps = float(sys.argv[5])
    except:
      eps = elbow_eps(layer_activations_df.iloc[:, :-3])
    print("eps", eps)

    min_samples = 5
    
    output_dir = '../mapper_graphs/additional_models/'
    output_fname = model+'_'+layer
    
    get_mapper_graph(layer_activations_df, interval, overlap, eps, min_samples, output_dir, output_fname, is_additional=True)
    
    

