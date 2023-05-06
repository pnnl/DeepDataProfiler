#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 19:11:48 2021

"""
import kneed
import numpy as np
import pandas as pd
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def elbow_eps(df):
    """
    df: only numerical cols included
    """
    print("get eps for df", df.shape)
    nbrs = NearestNeighbors(n_neighbors=2).fit(df)
    distances, indices = nbrs.kneighbors(df)
    distances = np.sort(distances, axis=0)[::-1]
    kneedle = kneed.KneeLocator(distances[:, 1], np.linspace(0, 1, num=len(distances)), curve='convex', direction='decreasing')
    if kneedle.knee:
        # eps = kneedle.knee * 0.75
        eps = kneedle.knee
    else:
        dists = distances[:,1]
        dists.sort()
        eps = dists[len(dists)//2]
    return np.round(eps, 3)

def get_knn(df, plt_title, is_cifar100=False): 
    if not is_cifar100:
        activations = df.iloc[:, 1:]
        fig_path = "../figs/"+plt_title+".png"
    else:
        activations = df.iloc[:, 1:]
        fig_path = plt_title+".png"
    k=5
    print('Running KNN with activations:', activations.shape)
    
    index = NNDescent(activations, n_neighbors=15, metric='euclidean')
    out = index.query(activations, k=k)
    dist = out[1]
    s_dist=np.sort(dist, axis=0)
    plt.figure(figsize=(5, 5), dpi=500)
    plt.plot(s_dist[:,k-1])
    plt.title(plt_title)
    plt.savefig(fig_path)
