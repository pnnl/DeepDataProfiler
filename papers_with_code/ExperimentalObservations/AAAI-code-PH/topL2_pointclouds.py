""" Code for constructing point cloud summaries and computing PH using top $l^2$-norm sampling

Notes
-----
Model architecture : ResNet-18 [1]_
Dataset : CIFAR-10 [2]_

Relevant sections : 
    - Point cloud summaries of activations - Top $l^2$-norm activations
    - Experiments with PH

Relevant libraries :
    - `Torchvision` [3]_
    - `PyTorch` [4]_
    - `Ripser` [5]_

Notation :
 - N = batchsize, i.e., # of images
 - c = # of channels
 - n, m = spatial dimensions, i.e., # of rows and columns, resp.

References
----------
.. [1] He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep Residual Learning for Image Recognition.
   In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
.. [2] Krizhevsky, A.; and Hinton, G. 2009. Learning multiple layers of features from tiny images.
   Technical Report 0, University of Toronto, Toronto, Ontario.
.. [3] Marcel, S.; and Rodriguez, Y. 2010. Torchvision the Machine-Vision Package of Torch.
   In Proceedings of the 18th ACM International Conference on Multimedia, MM’10, 1485–1488.
   New York, NY, USA: Association for Computing Machinery. ISBN 9781605589336.
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

import torch
import torchvision.transforms as tv_transforms 
from torchvision.datasets import CIFAR10
from torchvision.models.feature_extraction import create_feature_extractor

from ripser import ripser

from cifar_resnet import resnet18

torch.manual_seed(0)

# UPDATE THESE TO REFLECT YOUR OWN DIRECTORIES AND FILE NAMING CONVENTIONS:
# path to project directory containing all model and experiment files
projectdir = '/rcfs/projects/blaktop'
# path to trained model directory containing all weights
modeldir = f'{projectdir}/resnet18Models/resnet18_runs2'
# directory prefix and filename for trained model weights
# e.g., path to weights for model i is modeldir/prefix_{i}/weightfile
prefix = 'resnet18_cifar_large'
weightfile = 'final_weights.pt'
# path to experiment directory to save results in
expdir = f'{projectdir}/resnet_cifar_experiments'

def load_cifar(batch_size=2048):
    """ Load batches of CIFAR-10 test images with standard transforms """
    norm_mean = np.asarray((0.4914, 0.4822, 0.4465))
    norm_std = np.asarray((0.2023, 0.1994, 0.2010))
    transforms = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(norm_mean,norm_std)
    ])
    
    testset = CIFAR10(root='./data', train=False, download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=batch_size)
    
    return testloader

def load_model(path):
    """ Load ResNet-18 model with trained weights from path """
    state_dict = torch.load(path)
    model = resnet18()
    model.load_state_dict(state_dict)
    return model.eval()

def hook_conv_layers(model):
    """ Find and hook convolutional layers of model """
    layers = []
    for layer, module in model.named_modules():
        if 'conv' in layer and isinstance(module, torch.nn.Conv2d):
            layers.append(layer)
    
    feature_extractor = create_feature_extractor(model,return_nodes=layers)
    return layers, feature_extractor.eval()

def top_activations(activations):
    """ Build point cloud for each layer using top $l^2$-norm sampling
    
    Described in Section: `Point cloud summaries of activations - Top $l^2$-norm activations`
    
    Parameters
    ----------
    activations : dict of {str: torch.Tensor}
        {layer: activations}; activations dims = (N, c, n, m)
    
    Returns
    -------
    top_idx : dict of {str: torch.Tensor}
        spatial position index (flattened) of each sampled spatial activation vector;
        {layer: spatial indices}; spatial indices dims = (N,)
    top_act : dict of {str: torch.Tensor}
        {layer: point cloud}; point cloud dims = (N, c)
    """
    top_idx = {}
    top_act = {}
    
    for layer, actives in activations.items():
        Y = actives.flatten(start_dim=2)
        top_idx[layer] = torch.linalg.norm(Y, ord=2, dim=1).argmax(dim=-1)
        top_act[layer] = Y[torch.arange(Y.shape[0]),:,top_idx[layer]]
    
    return top_idx, top_act

def compute_persistence(pointclouds):
    """ Compute PH of point clouds
    
    Parameters
    ----------
    pointclouds : dict of {str: torch.Tensor}
        {layer: point cloud}; point cloud dims = (N, c)
    
    Returns
    -------
    PH : dict of {str: dict}
        {layer: PH results computed by `ripser`}
    """
    PH = {layer: ripser(pointclouds[layer]) for layer in pointclouds}
    return PH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet-18 CIFAR-10 Top L2 Spatial Point Cloud Homology')
    parser.add_argument('first', type=int, help='run index of first random seed ResNet-18 model')
    parser.add_argument('last', type=int, help='run index of last random seed ResNet-18 model (exclusive)')
    parser.add_argument('-b', type=int, help='batchsize', default=1000)
    parser.add_argument('-n', type=int, help='number of batches', default=1)
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    
    testloader = load_cifar(batch_size=args.b)
    
    for b, batch in enumerate(testloader):
        if b >= args.n:
            break
            
        images, labels = batch
        
        for i in range(args.first, args.last):
            modelpath = f'{modeldir}/{prefix}_{i}'
            savepath = f'{expdir}/{prefix}_{i}'
            
            # load model with hooks
            model = load_model(f'{modelpath}/{weightfile}')
            layers, fx_model = hook_conv_layers(model)
            
            # grab activations from conv. layers
            activations = fx_model(images)
            
            # construct and save point clouds
            spatial_idx, pointclouds = top_activations(activations)
            torch.save(spatial_idx, f'{savepath}/top_spatial_idx_batch{b}_{args.b}.pt')
            torch.save(pointclouds, f'{savepath}/pointclouds_batch{b}_{args.b}.pt')
            
            # compute and save PH
            PH = compute_persistence(pointclouds)
            pickle.dump(PH, open(f'{savepath}/persistence_batch{b}_{args.b}.p','wb'))


