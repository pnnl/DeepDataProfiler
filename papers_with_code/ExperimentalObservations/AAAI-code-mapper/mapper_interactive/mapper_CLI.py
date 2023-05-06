#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 02:54:52 2021

"""

import numpy as np
import pandas as pd
import os
from .kmapper import KeplerMapper, Cover
from sklearn.cluster import DBSCAN
import json
import itertools
from os.path import join
from sklearn.preprocessing import MinMaxScaler, normalize

def get_filter_fn(X, f):
    mapper = KeplerMapper()
    filter_fn = mapper.fit_transform(X, projection=f).reshape(-1,1)
    return filter_fn


def mapper_wrapper(X, filter_fn, clusterer, cover, is_parallel=True, **mapper_args):
    mapper = KeplerMapper()
    if is_parallel:
        graph = mapper.map_parallel(filter_fn, X, clusterer=clusterer, cover=cover, use_gpu=True, **mapper_args)
    else:
        graph = mapper.map(filter_fn, X, clusterer=clusterer, cover=cover, **mapper_args)
    return graph


def graph_to_dict(g, **kwargs):
    d = {}
    d['nodes'] = {}
    d['edges'] = {}
    for k in g['nodes']:
        d['nodes'][k] = g['nodes'][k]
    for k in g['links']:
        d['edges'][k] = g['links'][k]
    for k in kwargs.keys():
        d[k] = kwargs[k]
    return d


def normalize_data(X, norm_type):
    if norm_type == "none" or norm_type is None:
        X_prime = X
        pass
    elif norm_type == "0-1":  # axis=0, min-max norm for each column
        scaler = MinMaxScaler()
        X_prime = scaler.fit_transform(X)
    else:
        X_prime = normalize(X, norm=norm_type, axis=0,
                            copy=False, return_norm=False)
    return X_prime


def get_node_id(node):
    interval_idx = node.interval_index
    cluster_idx = node.cluster_index
    node_id = "node"+str(interval_idx)+str(cluster_idx)
    return node_id


def get_mapper_graph(df, interval, overlap, eps, min_samples, output_dir, output_fname, is_parallel=True, is_cifar100=False, is_additional=False):
    print("get mapper for", output_fname)
    if is_additional:
        df_np = df.iloc[:,:-3].to_numpy()
        categorical_cols = ['ground']
    elif is_cifar100:
        df_np = df.iloc[:,2:].to_numpy()
        categorical_cols = ['fine_label', 'coarse_label']
    else:
        # df_np = df.iloc[:,2:].to_numpy()
        # categorical_cols = ['label', 'predictions']
        df_np = df.iloc[:,1:].to_numpy()
        categorical_cols = ['label']
    print(df_np.shape)
    # No normalization
    # df_np = normalize_data(df_np, norm_type=norm) 
    filter_str = "l2norm"
    filter_fn = get_filter_fn(df_np.astype("float"), filter_str)

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    cover = Cover(n_cubes=interval, perc_overlap=overlap / 100)
    g = graph_to_dict(mapper_wrapper(
                df_np, filter_fn, clusterer, cover, is_parallel=is_parallel))
    for node_id in g['nodes']:
        vertices = g['nodes'][node_id]
        node = {}
        node['categorical_cols_summary'] = {}
        node['vertices'] = vertices
        node['avgs'] = {}
        node['avgs']['lens'] = np.mean(filter_fn[vertices])
        for col in categorical_cols:
            data_categorical_i = df[col].iloc[vertices]
            node['categorical_cols_summary'][col] = data_categorical_i.value_counts().to_dict()
        g['nodes'][node_id] = node
    g['categorical_cols'] = list(categorical_cols)
    numerical_col_keys = ['lens']
    g['numerical_col_keys'] = list(numerical_col_keys)  
    
    
    filename = 'mapper_' + str(output_fname) + '_' + str(interval) + '_' + str(overlap) + '_' + str(eps) + '.json'

    with open(join(output_dir, filename), 'w') as fp:
        json.dump(g, fp)
