import torch
import torch.nn.functional as F
from enum import Enum, auto
from typing import Optional, Union, List, Callable, Dict, Tuple

from .feature_objective import (
    FeatureObjective,
    NeuronMultiLayerCoord,
    SVDNeuronMultiLayerCoord,
    SVDNeuronObjective,
    SVDMultiLayerCoord,
    ChannelMultiLayerCoord,
)


class NeuronBasis(Enum):
    '''Enum for the different neuron basis types.'''
    ACTIVATION = auto()
    SVD = auto()


def neurons_dictionary_objective(
    layer_neuron_weights: Dict[str, List[Tuple[int]]],
    neuron_type: NeuronBasis = NeuronBasis.ACTIVATION,
    transform_activations: Optional[Callable] = None,
) -> FeatureObjective:
    """Reads a dictionary of {layer : [neurons...]} and returns a single objective to optimize. Requires
    a transform_activations function if neuron_type is NeuronBasis.SVD
    
    Parameters
    ----------
        layer_neuron_weights (Dict[str, List[Tuple[int]]]): A dictionary of {layer : [neurons...]}
        neuron_type (NeuronBasis, optional): The neuron basis type. Defaults to NeuronBasis.ACTIVATION.
        transform_activations (Optional[Callable], optional): A function to transform the activations. Defaults to None.
    Returns
    -------
        FeatureObjective: The objective to optimize.
    """
    layers = []
    coords = []
    weights = []

    # iterate through dict, checking if it's of the form {layer : [neurons...]}
    # or {layer : [(neuron, weight), ...]} at each layer
    for layer, neuron_coordinates in layer_neuron_weights.items():
        if isinstance(neuron_coordinates[0][0], tuple):
            nc, ws = list(zip(*neuron_coordinates))
            layers.extend([layer] * len(neuron_coordinates))
            coords.extend(nc)
            weights.extend(ws)

        elif isinstance(neuron_coordinates[0][0], int):
            layers.extend([layer] * len(neuron_coordinates))
            coords.extend(list(neuron_coordinates))
        else:
            raise TypeError(
                f"Unexpected neuron or weight combination, {neuron_coordinates[0]}"
            )

    if neuron_type == NeuronBasis.ACTIVATION:
        neuron_multi_objective = NeuronMultiLayerCoord(
            layer=layers, coord=coords, weights=weights
        )
        return neuron_multi_objective
    elif neuron_type == NeuronBasis.SVD:
        if transform_activations:
            svd_neuron_multi_objective = SVDNeuronMultiLayerCoord(
                layer=layers,
                coord=coords,
                transform_activations=transform_activations,
                weights=weights,
            )
            return svd_neuron_multi_objective
        else:
            raise RuntimeError(
                f"The SVD neuron basis requires a transform_activations function"
            )
    else:
        raise TypeError(f"Unexpected neuron type {neuron_type}")


def dictionary_objective(
    layer_neuron_weights: Union[Dict[str, List[Tuple[int]]], Dict[str, List[int]]],
    neuron_type: NeuronBasis = NeuronBasis.ACTIVATION,
    transform_activations: Optional[Callable] = None,
) -> FeatureObjective:
    """Reads a dictionary of either {layer : [neurons...]} or {layer : [(neuron, weight), ...]} 
    and returns a single objective to optimize. Requires a transform_activations function if neuron_type 
    is NeuronBasis.SVD

    Parameters
    ----------
        layer_neuron_weights (Union[Dict[str, List[Tuple[int]]], Dict[str, List[int]]]): A dictionary of either
            {layer : [neurons...]} or {layer : [(neuron, weight), ...]}
        neuron_type (NeuronBasis, optional): The neuron basis type. Defaults to NeuronBasis.ACTIVATION.
        transform_activations (Optional[Callable], optional): A function to transform the activations. Defaults to None.
    Returns
    -------
        FeatureObjective: The objective to optimize.
    """

    layers = []
    coords = []
    weights = []
    # iterate through dict, checking if it's of the form {layer : [neurons...]}
    # or {layer : [(neuron, weight), ...]} at each layer
    for layer, neuron_coordinates in layer_neuron_weights.items():
        if isinstance(neuron_coordinates[0], tuple):
            nc, ws = list(zip(*neuron_coordinates))
            layers.extend([layer] * len(neuron_coordinates))
            coords.extend(nc)
            weights.extend(ws)

        elif isinstance(neuron_coordinates[0], int):
            layers.extend([layer] * len(neuron_coordinates))
            coords.extend(list(neuron_coordinates))
        else:
            raise TypeError(
                f"Unexpected neuron or weight combination, {neuron_coordinates[0]}"
            )

    if neuron_type == NeuronBasis.ACTIVATION:
        neuron_multi_objective = ChannelMultiLayerCoord(
            layer=layers, coord=coords, weights=weights
        )
        return neuron_multi_objective
    elif neuron_type == NeuronBasis.SVD:
        if transform_activations:
            svd_neuron_multi_objective = SVDMultiLayerCoord(
                layer=layers,
                coord=coords,
                transform_activations=transform_activations,
                weights=weights,
            )
            return svd_neuron_multi_objective
        else:
            raise RuntimeError(
                f"The SVD neuron basis requires a transform_activations function"
            )
    else:
        raise TypeError(f"Unexpected neuron type {neuron_type}")
