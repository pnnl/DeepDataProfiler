from deep_data_profiler.classes.profile import Profile
import torch
import torch.nn.functional as F
from collections import OrderedDict
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional, Union, List, Callable, Dict, Tuple
from tqdm import tqdm


class FeatureObjective(ABC):
    """
    An Abstract Base Class for an input feature to optimize w.r.t a PyTorch model.

    Attributes
    ----------
    layer : Union[str, List[str]]
        Layer or layers from which to pull activations.
    coord : torch.Tensor
        The feature visualization object that is optimized
    transform_activations : Optional[Callable[[torch.Tensor], torch.Tensor]]
        A transformation made on the activations during optimization.
        (for example, projection on the left hand singular vectors of the SVD decomposition
        of the weights.)
    """

    def __init__(
        self,
        layer: Union[str, List[str]],
        coord: Union[int, Tuple[int], List[int], List[Tuple[int]]],
        transform_activations: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.layer = layer
        self.coord = coord
        if transform_activations:
            self.transform_activations = transform_activations
        else:
            self.transform_activations = lambda x: x
        super().__init__()

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return lambda activations: self(activations) + other
        elif isinstance(other, FeatureObjective):
            return lambda activations: self(activations) + other(activations)

    def __sub__(self, other):
        return lambda activations: self(activations) + (-1 * other(activations))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return lambda activations: self(activations) * other
        elif isinstance(other, FeatureObjective):
            return lambda activations: self(activations) * other(activations)

    def __neg__(self):
        return lambda activations: -1 * self(activations)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    @abstractmethod
    def __call__(self):
        pass


class ChannelObjective(FeatureObjective):
    def __call__(self, activations: torch.Tensor) -> torch.Tensor:
        return -activations[self.layer][:, self.coord].mean()


class Diversity(FeatureObjective):
    '''Originally from https://distill.pub/2017/feature-visualization/. 
    Uses a gram matrix to define style (see https://ieeexplore.ieee.org/document/7780634).'''
    def __call__(self, activations: torch.Tensor) -> torch.Tensor:
        layer_activations = activations[layer]
        batch, c, h, w = layer_activations.shape
        flattened = layer_activations.view(batch, c, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        return -sum([ sum([ (grams[i]*grams[j]).sum()
               for j in range(batch) if j != i])
               for i in range(batch)]) / batch


def model_svd_dict(profiler: Profile) -> Dict[str, torch.Tensor]:
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


def project_svd(profiler: Profile) -> Callable:
    svd_dict = model_svd_dict(profiler)

    def inner(activations: Dict[str, torch.Tensor], layer: str) -> torch.Tensor:
        u_vec = svd_dict[layer]
        layer_activations = activations[layer]
        b, c, h, w = layer_activations.shape
        layer_reshape = layer_activations.view(b, c, -1)
        # take SVD projection
        uprojy = torch.matmul(u_vec.T, layer_reshape)
        return uprojy.reshape(b, c, h, w)

    return inner


class SVDMean(FeatureObjective):
    def __call__(self, activations: torch.Tensor):
        uprojy = self.transform_activations(activations, self.layer)
        return -uprojy[:, self.coord].mean()


class ChannelMultiLayerCoord(FeatureObjective):
    def __call__(self, activations: torch.Tensor):
        sum_channels_obj = 0.0
        if isinstance(self.layer, list) and isinstance(self.coord, list):
            assert len(self.layer) == len(
                self.coord
            ), f"Length of layers ({len(self.layer)}) and coordinates ({len(self.coord)}) not equal"
            for idx, lyr in enumerate(self.layer):
                sum_channels_obj += -activations[lyr][:, self.coord[idx]].mean()
        elif isinstance(self.layer, list) and (
            isinstance(self.coord, int) or isinstance(self.coord, slice)
        ):
            for _, lyr in enumerate(self.layer):
                sum_channels_obj += -activations[lyr][:, self.coord].mean()
        else:
            raise TypeError(
                f"Expect layer and/or coord to be a list, received {type(self.layer)} and {type(self.coord)} respectively"
            )
        return sum_channels_obj


class SVDMultiLayerCoord(FeatureObjective):
    def __call__(self, activations: torch.Tensor):
        sum_svd_obj = 0.0
        if isinstance(self.layer, list) and isinstance(self.coord, list):
            assert len(self.layer) == len(
                self.coord
            ), f"Length of layers ({len(self.layer)}) and coordinates ({len(self.coord)}) not equal"
            for idx, lyr in enumerate(self.layer):
                uprojy = self.transform_activations(activations, lyr)
                sum_svd_obj += -uprojy[:, self.coord[idx]].mean()
        elif isinstance(self.layer, list) and (
            isinstance(self.coord, int) or isinstance(self.coord, slice)
        ):
            for _, lyr in enumerate(self.layer):
                uprojy = self.transform_activations(activations, lyr)
                sum_svd_obj += -uprojy[:, self.coord].mean()
        else:
            raise TypeError(
                f"Expect layer and/or coord to be a list, received {type(self.layer)} and {type(self.coord)} respectively"
            )
        return sum_svd_obj


class NeuronObjective(FeatureObjective):
    def __call__(self, activations: torch.Tensor):
        layer_activations = activations[self.layer]
        if len(self.coord) == 3:
            chn, x, y = self.coord
        # take center neuron if only channel
        elif len(self.coord) == 1:
            chn = self.coord
            *_, x_total, y_total = layer_activations.shape
            x = x_total // 2
            y = y_total // 2
        else:
            raise NotImplementedError(
                f"Not implemented for coord of length/type {len(self.coord)}/{type(self.coord)}"
            )
        return -layer_activations[:, chn, x, y]


def get_neuron_rf(coord, acts) -> Tuple[int]:
    if len(coord) == 3:
        chn, x, y = coord
    # take center neuron if only channel
    elif len(coord) == 1:
        chn = coord
        *_, x_total, y_total = acts.shape
        x = x_total // 2
        y = y_total // 2
    else:
        raise NotImplementedError(
            f"Not implemented for coord of length/type {len(self.coord)}/{type(self.coord)}"
        )
    return chn, x, y


class SVDNeuronObjective(FeatureObjective):
    def __call__(self, activations: torch.Tensor):
        uprojy = self.transform_activations(activations, self.layer)
        chn, x, y = get_neuron_rf(self.coord, uprojy)
        return -uprojy[:, chn, x, y]


class SVDNeuronMultiLayerCoord(FeatureObjective):
    def __call__(self, activations: torch.Tensor):
        sum_svd_obj = 0.0
        if isinstance(self.layer, list) and isinstance(self.coord, list):
            assert len(self.layer) == len(
                self.coord
            ), f"Length of layers ({len(self.layer)}) and coordinates ({len(self.coord)}) not equal"

            for idx, lyr in enumerate(self.layer):
                uprojy = self.transform_activations(activations, lyr)
                chn, x, y = get_neuron_rf(self.coord[idx], uprojy)
                sum_svd_obj += -uprojy[:, chn, x, y].mean()

        elif isinstance(self.layer, list) and isinstance(self.coord, int):
            for _, lyr in enumerate(self.layer):
                uprojy = self.transform_activations(activations, lyr)
                chn, x, y = get_neuron_rf((self.coord,), uprojy)
                sum_svd_obj += -uprojy[:, chn, x, y].mean()

        else:
            raise TypeError(
                f"Expect layer and/or coord to be a list, received {type(self.layer)} and {type(self.coord)} respectively"
            )
        return sum_svd_obj


class NeuronMultiLayerCoord(FeatureObjective):
    def __call__(self, activations: torch.Tensor):
        sum_neuron_obj = 0.0
        if isinstance(self.layer, list) and isinstance(self.coord, list):
            assert len(self.layer) == len(
                self.coord
            ), f"Length of layers ({len(self.layer)}) and coordinates ({len(self.coord)}) not equal"

            for idx, lyr in enumerate(self.layer):
                layer_acts = activations[lyr]
                chn, x, y = get_neuron_rf(self.coord[idx], layer_acts)
                sum_neuron_obj += -layer_acts[:, chn, x, y].mean()

        elif isinstance(self.layer, list) and isinstance(self.coord, int):
            for _, lyr in enumerate(self.layer):
                layer_acts = activations[lyr]
                chn, x, y = get_neuron_rf((self.coord,), layer_acts)
                sum_neuron_obj += -layer_acts[:, chn, x, y].mean()

        else:
            raise TypeError(
                f"Expect layer and/or coord to be a list, received {type(self.layer)} and {type(self.coord)} respectively"
            )
        return sum_neuron_obj


class NeuronBasis(Enum):
    ACTIVATION = auto()
    SVD = auto()


def neurons_dictionary_objective(
    layer_neuron_weights: Dict[str, List[Tuple[int]]],
    neuron_type: NeuronBasis,
    transform_activations: Optional[Callable] = None,
) -> FeatureObjective:
    """Reads a dictionary of {layer : [neurons...]} and returns a single objective to optimize. Requires
    a transform_activations function if neuron_type is NeuronBasis.SVD"""
    layers = []
    coords = []
    for layer, neuron_coordinates in layer_neuron_weights.items():
        for n_coord in neuron_coordinates:
            layers.append(layer)
            coords.append(n_coord)

    if neuron_type == NeuronBasis.ACTIVATION:
        neuron_multi_objective = NeuronMultiLayerCoord(layer=layers, coord=coords,)
        return neuron_multi_objective
    elif neuron_type == NeuronBasis.SVD:
        if transform_activations:
            svd_neuron_multi_objective = SVDNeuronMultiLayerCoord(
                layer=layers, coord=coords, transform_activations=transform_activations,
            )
            return svd_neuron_multi_objective
        else:
            raise RuntimeError(
                f"The SVD neuron basis requires a transform_activations function"
            )
    else:
        raise TypeError(f"Unexpected neuron type {neuron_type}")

def dictionary_objective(
    layer_neuron_weights: Dict[str, List[Tuple[int]]],
    neuron_type: NeuronBasis,
    transform_activations: Optional[Callable] = None,
) -> FeatureObjective:
    """Reads a dictionary of {layer : [neurons...]} and returns a single objective to optimize. Requires
    a transform_activations function if neuron_type is NeuronBasis.SVD"""
    layers = []
    coords = []
    for layer, neuron_coordinates in layer_neuron_weights.items():
        for n_coord in neuron_coordinates:
            layers.append(layer)
            coords.append(n_coord)

    if neuron_type == NeuronBasis.ACTIVATION:
        neuron_multi_objective = ChannelMultiLayerCoord(layer=layers, coord=coords,)
        return neuron_multi_objective
    elif neuron_type == NeuronBasis.SVD:
        if transform_activations:
            svd_neuron_multi_objective = SVDMultiLayerCoord(
                layer=layers, coord=coords, transform_activations=transform_activations,
            )
            return svd_neuron_multi_objective
        else:
            raise RuntimeError(
                f"The SVD neuron basis requires a transform_activations function"
            )
    else:
        raise TypeError(f"Unexpected neuron type {neuron_type}")
