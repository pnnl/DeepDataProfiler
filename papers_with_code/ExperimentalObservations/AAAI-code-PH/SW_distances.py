""" Code for computing SW distances between PDs [1]_ of point cloud summaries of activations

Notes
-----
Relevant section : Experiments with PH

Relevant library : `Persim` [2]_

References
----------
.. [1] Carrière, M.; Cuturi, M.; and Oudot, S. 2017. Sliced Wasserstein Kernel for 
   Persistence Diagrams. In Precup, D.; and Teh, Y. W., eds., Proceedings of the
   34th International Conference on Machine Learning, volume 70 of Proceedings of
   Machine Learning Research, 664–673. PMLR.
.. [2] Saul, N.; and Tralie, C. 2019. Scikit-TDA: Topological Data Analysis for Python.
"""

import pickle
import argparse
import numpy as np

from persim import sliced_wasserstein

# UPDATE THESE TO REFLECT YOUR OWN DIRECTORIES AND FILE NAMING CONVENTIONS:
# path to project directory containing all model and experiment files
projectdir = '/rcfs/projects/blaktop'
# path to experiment directory containing PH results
expdir = f'{projectdir}/resnet_cifar_experiments'
# directory prefix and filename suffix for PH results per model/batch
# e.g., path to PH results for model i on batch b is expdir/prefix_{i}/persistence_batch{b}filesuffix.p
prefix = 'resnet18_cifar_large'
filesuffix = '_1000'

# number of randomly initialized models (used in 'cross' mode, see below)
num_models = 100

def get_layers(PH):
    """ Returns layers from PH dict keys in the correct order
    
    Note
    ----
    Specifically designed for the module names defined in `cifar_resnet.resnet18`
    """
    # use key so that conv1 is first (before all block_seq)
    return sorted(PH, key = lambda x : x.replace('conv',''))

def SW_dist_internal(PH, layers):
    """ Computes SW distance between layers of a model """
    nlayers = len(layers)
    dist = np.zeros((nlayers, nlayers))
    for i, layer_i in enumerate(layers[:-1]):
        for j, layer_j in enumerate(layers[i+1:], start=i+1):
            dist[i][j] = sliced_wasserstein(PH[layer_i]['dgms'][1], PH[layer_j]['dgms'][1])
            dist[j][i] = dist[i][j]
    return dist

def SW_dist_cross_model(PH_i, PH_j, layers):
    """ Computes SW distances between layers for two differently initialized models """
    nlayers = len(layers)
    dist = np.zeros((nlayers, nlayers))
    for i, layer_i in enumerate(layers):
        for j, layer_j in enumerate(layers):
            dist[i][j] = sliced_wasserstein(PH_i[layer_i]['dgms'][1], PH_j[layer_j]['dgms'][1])
    return dist
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SW Distances between PDs of Point Cloud Summaries')
    parser.add_argument('-fs', type=int, help='run index of first random seed ResNet-18 model', default=0)
    parser.add_argument('-ls', type=int, help='run index of last random seed ResNet-18 model (exclusive)', default=1)
    parser.add_argument('-fb', type=int, help='index of first batch', default=0)
    parser.add_argument('-lb', type=int, help='index of last batch (exclusive)', default=1)
    parser.add_argument('-m', type=str, help="mode: either 'int' for single model internal distances, or 'cross' for distances between differently initialized models", default='int')
    args = parser.parse_args()
    
    for b in range(args.fb, args.lb):
        filename = f'persistence_batch{b}{filesuffix}.p'
        for i in range(args.fs, args.ls):
            savepath = f'{expdir}/{prefix}_{i}'
            
            PH = pickle.load(open(f'{savepath}/{filename}','rb'))
            layers = get_layers(PH)
            
            if args.m == 'int':
                dist = SW_dist_internal(PH, layers)
            else:
                dist = []
                for j in range(num_models):
                    if j != i:
                        otherpath = f'{expdir}/{prefix}_{j}'
                        PH_other = pickle.load(open(f'{otherpath}/{filename}','rb'))
                        dist_other = SW_dist_cross_model(PH, PH_other, layers)
                    else:
                        dist_other = SW_dist_internal(PH,layers)
                    dist.append(dist_other)
                dist = np.concatenate(dist, axis=1)
                
            np.save(f'{savepath}/sliced_wasserstein_batch{b}{filesuffix}_{args.m}', dist)


