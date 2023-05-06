""" Code for performing intuitive sensitivity test [2]_ for SW distance between PDs [1]_

Notes
-----
Relevant section : 
    Experiments with PH - Representation similarity metrics - intuitive testing (Sensitivity)

Relevant libraries :
    - `Ripser` [5]_
    - `Persim` [3]_
    - `PyTorch` [4]_

Notation :
 - N = batchsize, i.e., # of images
 - c = # of channels

References
----------
.. [1] Carrière, M.; Cuturi, M.; and Oudot, S. 2017. Sliced Wasserstein Kernel for 
   Persistence Diagrams. In Precup, D.; and Teh, Y. W., eds., Proceedings of the
   34th International Conference on Machine Learning, volume 70 of Proceedings of
   Machine Learning Research, 664–673. PMLR. 
.. [2] Ding, F.; Denain, J.-S.; and Steinhardt, J. 2021. Grounding Representation
   Similarity Through Statistical Testing. In Ranzato, M.; Beygelzimer, A.; Dauphin, Y.;
   Liang, P.; and Vaughan, J. W., eds., Advances in Neural Information Processing Systems,
   volume 34, 1556–1568. Curran Associates, Inc.
.. [3] Saul, N.; and Tralie, C. 2019. Scikit-TDA: Topological Data Analysis for Python.
.. [4] Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; 
   Lin, Z.; Gimelshein, N.; Antiga, L.; Desmaison, A.; Kopf, A.; Yang, E.; DeVito, Z.;
   Raison, M.; Tejani, A.; Chilamkurthy, S.; Steiner, B.; Fang, L.; Bai, J.; and Chintala, S.
   2019. PyTorch: An Imperative Style, High-PerformanceDeep Learning Library. In Wallach, H.;
   Larochelle, H.; Beygelzimer, A.; d'Alché-Buc, F.; Fox, E.; and Garnett, R., eds.,
   Advances in Neural Information Processing Systems 32, 8024–8035. Curran Associates, Inc.
.. [5] Tralie, C.; Saul, N.; and Bar-On, R. 2018. Ripser.py: A Lean Persistent Homology
   Library for Python. The Journal of Open Source Software, 3(29): 925.
"""

import pickle
import argparse
import numpy as np

from ripser import ripser
from persim import sliced_wasserstein
import torch

# UPDATE THESE TO REFLECT YOUR OWN DIRECTORIES AND FILE NAMING CONVENTIONS:
# path to project directory containing all model and experiment files
projectdir = '/rcfs/projects/blaktop'
# path to experiment directory containing PH results
expdir = f'{projectdir}/resnet_cifar_experiments'
# directory prefix and filename suffix for PH results per model/batch
# e.g., path to PH results for model i on batch b is expdir/prefix_{i}/persistence_batch{b}filesuffix.p
prefix = 'resnet18_cifar_large'
filesuffix = '_1000'

projectdir = '/rcfs/projects/blaktop'
expdir = f'{projectdir}/resnet_cifar_experiments'
prefix = 'resnet18_cifar_large'

def get_layers(PH):
    """ Returns layers from PH dict keys in the correct order
    
    Note
    ----
    Specifically designed for the module names defined in `cifar_resnet.resnet18`
    """
    # use key so that conv1 is first (before all block_seq)
    return sorted(PH, key = lambda x : x.replace('conv',''))

def low_rank(A, U, r):
    """ Returns a low-rank approximation
    
    Parameters
    ----------
    A : torch.Tensor
        point cloud, dims=(c, N)
    U : torch.Tensor
        from the SVD of A
    r : int
        rank of the approximation
    
    Returns
    -------
    torch.Tensor
        low-rank approximation point cloud, dims=(N, r)
    """
    return torch.mm(U.T[:r],A).T


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intuitive sensitivity testing of SW distances between layers')
    parser.add_argument('first', type=int, help='index of layer')
    parser.add_argument('last', type=int, help='index of last layer')
    parser.add_argument('-i', type=int, help='run index of random seed ResNet-18 model', default=0)
    parser.add_argument('-b', type=int, help='index of batch', default=0)
    parser.add_argument('-s', type=int, help='used to determine increment interval when removing PCs', default=4)
    args = parser.parse_args()
    
    step = args.s
    
    savepath = f'{expdir}/{prefix}_{args.i}'
    full_suffix = f'batch{args.b}{filesuffix}'
    
    pointclouds = torch.load(f'{savepath}/pointclouds_{full_suffix}.pt')
    PH = pickle.load(open(f'{savepath}/persistence_{full_suffix}.p','rb'))
    layers = get_layers(PH)
    
    for i in range(args.first, args.last):
        lowrank_suffix = f'lowrank_layer{i}_batch{args.b}{filesuffix}'
        
        layer = layers[i]
        A = pointclouds[layer].T
        pc = torch.linalg.svd(A)
        N = A.shape[0]
        khats = np.linspace(step,N,N//step,dtype=int,endpoint=False)
        
        # keyed by # PCs removed
        removed_pcs = {k: low_rank(A, pc.U, N-k) for k in khats}
        torch.save(removed_pcs, f'{savepath}/pointclouds_{lowrank_suffix}')
        
        # keyed by # PCs removed
        PH_lowrank = {k: ripser(removed_pcs[k]) for k in khats}
        pickle.dump(PH_lowrank, open(f'{savepath}/persistence_{lowrank_suffix}.p','wb'))
        
        # first row is SW distance, second row is fraction of PCs removed
        dist = np.zeros((2,len(khats)))
        dist[0] = np.asarray(
            [sliced_wasserstein(PH[layer]['dgms'][1], PH_lowrank[k]['dgms'][1]) for k in khats]
        )
        dist[1] = khats/N
        np.save(f'{savepath}/sliced_wasserstein_{lowrank_suffix}',dist)
        


