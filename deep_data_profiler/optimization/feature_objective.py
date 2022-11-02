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
        weights: Optional[List[float]] = None,
    ):
        self.layer = layer
        self.coord = coord
        self.weights = weights

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
    """Objective for feature visualization of a single channel, or multiple channels"""

    def __call__(self, activations: torch.Tensor) -> torch.Tensor:
        return -activations[self.layer][:, self.coord].mean()


class Diversity(FeatureObjective):
    """Originally from https://distill.pub/2017/feature-visualization/.
    Uses a gram matrix to define style (see https://ieeexplore.ieee.org/document/7780634)."""

    def __call__(self, activations: torch.Tensor) -> torch.Tensor:
        layer_activations = activations[layer]
        batch, c, h, w = layer_activations.shape
        flattened = layer_activations.view(batch, c, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        return (
            -sum(
                [
                    sum([(grams[i] * grams[j]).sum() for j in range(batch) if j != i])
                    for i in range(batch)
                ]
            )
            / batch
        )


class SVDMean(FeatureObjective):
    """Objective for feature visualization for a single SVD signal, or multiple SVD signals"""

    def __call__(self, activations: torch.Tensor):
        uprojy = self.transform_activations(activations, self.layer)
        return -uprojy[:, self.coord].mean()


class ChannelMultiLayerCoord(FeatureObjective):
    """Objective for feature visualization for channel(s) across multiple layers"""

    def __call__(self, activations: torch.Tensor):
        sum_channels_obj = 0.0

        # if weights not passed, assign equal weight to all.
        # Not the most efficient, but DRY.
        if not self.weights:
            self.weights = [1.0] * len(self.coord)
        # check the type of the layer coordinate passed
        if isinstance(self.layer, list) and isinstance(self.coord, list):
            # assert that the layer, coordinate, and weight lists have equal length
            assert len(self.layer) == len(
                self.coord
            ), f"Length of layers ({len(self.layer)}) and coordinates ({len(self.coord)}) not equal"
            assert len(self.layer) == len(
                self.weights
            ), f"Length of layers ({len(self.layer)}) and weights ({len(self.weights)}) not equal"
            for idx, lyr in enumerate(self.layer):
                sum_channels_obj -= (
                    self.weights[idx] * activations[lyr][:, self.coord[idx]].mean()
                )
        elif isinstance(self.layer, list) and (
            isinstance(self.coord, int) or isinstance(self.coord, slice)
        ):
            for idx, lyr in enumerate(self.layer):
                sum_channels_obj -= (
                    self.weights[idx] * activations[lyr][:, self.coord].mean()
                )
        else:
            raise TypeError(
                f"Expect layer and/or coord to be a list, received {type(self.layer)} and {type(self.coord)} respectively"
            )
        return sum_channels_obj


class SVDMultiLayerCoord(FeatureObjective):
    """Objective for feature visualization for SVD signal(s) across multiple layers"""

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
    """Objective for feature visualization for a single neuron, or multiple neurons"""

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


def get_neuron_rf(coord: Tuple[int], acts: torch.Tensor) -> Tuple[int]:
    """Get receptive field of a neuron.
    Parameters
    ----------
    coord : tuple
        Coordinate of neuron.
    acts : torch.Tensor
        Activations of layer.
    Returns
    -------
    tuple
        Receptive field of neuron."""
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
    """Objective for feature visualization for a single SVD spatial, or multiple SVD spatials"""

    def __call__(self, activations: torch.Tensor):
        uprojy = self.transform_activations(activations, self.layer)
        chn, x, y = get_neuron_rf(self.coord, uprojy)
        return -uprojy[:, chn, x, y]


class SVDNeuronMultiLayerCoord(FeatureObjective):
    """Objective for feature visualization for SVD spatial(s) across multiple layers"""

    def __call__(self, activations: torch.Tensor):
        sum_svd_obj = 0.0

        # if weights not passed, assign equal weight to all.
        # Not the most efficient, but DRY.
        if not self.weights:
            self.weights = [1.0] * len(self.coord)

        # check the type of the layer coordinate passed
        if isinstance(self.layer, list) and isinstance(self.coord, list):
            # assert that the layer, coordinate, and weight lists have equal length
            assert len(self.layer) == len(
                self.coord
            ), f"Length of layers ({len(self.layer)}) and coordinates ({len(self.coord)}) not equal"
            assert len(self.layer) == len(
                self.weights
            ), f"Length of layers ({len(self.layer)}) and weights ({len(self.weights)}) not equal"

            for idx, lyr in enumerate(self.layer):
                uprojy = self.transform_activations(activations, lyr)
                chn, x, y = get_neuron_rf(self.coord[idx], uprojy)
                sum_svd_obj -= self.weights[idx] * uprojy[:, chn, x, y].mean()

        elif isinstance(self.layer, list) and isinstance(self.coord, int):
            for idx, lyr in enumerate(self.layer):
                uprojy = self.transform_activations(activations, lyr)
                chn, x, y = get_neuron_rf((self.coord,), uprojy)
                sum_svd_obj -= self.weights[idx] * uprojy[:, chn, x, y].mean()

        else:
            raise TypeError(
                f"Expect layer and/or coord to be a list, received {type(self.layer)} and {type(self.coord)} respectively"
            )
        return sum_svd_obj


class NeuronMultiLayerCoord(FeatureObjective):
    """Objective for feature visualization for neuron(s) across multiple layers"""

    def __call__(self, activations: torch.Tensor):
        sum_neuron_obj = 0.0
        # if weights not passed, assign equal weight to all.
        # Not the most efficient, but DRY.
        if not self.weights:
            self.weights = [1.0] * len(self.coord)

        if isinstance(self.layer, list) and isinstance(self.coord, list):
            # assert that the layer, coordinate, and weight lists have equal length
            assert len(self.layer) == len(
                self.coord
            ), f"Length of layers ({len(self.layer)}) and coordinates ({len(self.coord)}) not equal"
            assert len(self.layer) == len(
                self.weights
            ), f"Length of layers ({len(self.layer)}) and weights ({len(self.weights)}) not equal"

            for idx, lyr in enumerate(self.layer):
                layer_acts = activations[lyr]
                chn, x, y = get_neuron_rf(self.coord[idx], layer_acts)
                sum_neuron_obj -= self.weights[idx] * layer_acts[:, chn, x, y].mean()

        elif isinstance(self.layer, list) and isinstance(self.coord, int):
            for idx, lyr in enumerate(self.layer):
                layer_acts = activations[lyr]
                chn, x, y = get_neuron_rf((self.coord,), layer_acts)
                sum_neuron_obj -= self.weights[idx] * layer_acts[:, chn, x, y].mean()

        else:
            raise TypeError(
                f"Expect layer and/or coord to be a list, received {type(self.layer)} and {type(self.coord)} respectively"
            )
        return sum_neuron_obj
