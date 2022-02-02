import torch
from deep_data_profiler.classes.profile import Profile
from typing import Dict, Callable, Optional
from collections import OrderedDict
from tqdm import tqdm


def model_svd_dict(profiler: Profile) -> Dict[str, torch.Tensor]:
    '''Creates a dictionary of the left-hand singular vectors
    of the weights, keyed by layer'''
    svd_dict = OrderedDict()
    layer_dict = profiler.create_layers()
    layer_ops = profiler.hooks
    for layer, layer_names in tqdm(layer_dict.items()):
        layer_name = layer_names[0][0]
        if (
            layer_name != "x_in"
            and "resnetadd" not in layer_name
            and "weight" in layer_ops[layer_name]._parameters
        ):

            X = layer_ops[layer_name]._parameters["weight"].detach()
            layer_name = layer_names[0][0]

            # get eigenvectors
            if layer_names[-1] == "contrib_linear":
                svd = torch.svd(X, compute_uv=True)
                svd_dict[layer_name] = svd.U

            elif layer_names[-1] == "contrib_conv2d":
                reshape_tens = torch.flatten(X, start_dim=1, end_dim=-1)
                svd = torch.svd(reshape_tens, compute_uv=True)
                svd_dict[layer_name] = svd.U
    return svd_dict


def project_svd(profiler: Profile, device: Optional[torch.device] = None) -> Callable:
    '''A helper function that projects a 4d matrix of activations onto the U-vec weight tensor SVD of the layer.'''
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    svd_dict = model_svd_dict(profiler)

    def inner(activations: Dict[str, torch.Tensor], layer: str) -> torch.Tensor:
        u_vec = svd_dict[layer]
        layer_activations = activations[layer]
        act_shape = layer_activations.shape
        if len(act_shape) == 4:
            b, c, h, w = act_shape
            layer_reshape = layer_activations.view(b, c, -1)
        elif len(act_shape) == 2:
            layer_reshape = layer_activations
        else:
            raise Exception("Activations not implemented for SVD")
        # take SVD projection
        uprojy = torch.matmul(u_vec.T.to(device), layer_reshape.to(device))
        return uprojy.reshape(b, c, h, w)

    return inner

