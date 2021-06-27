import numpy as np
import scipy.sparse as sp
from typing import List
from deep_data_profiler.classes import Profile


def jaccard_simple(mat1: sp.spmatrix, mat2: sp.spmatrix) -> float:
    """
    Computes the jaccard similarity of two sets = size of their
    intersection / size of their union

    Parameters
    ----------
    mat1 : scipy.sparse matrix
    mat2 : scipy.sparse matrix

    Returns
    -------
     : float
    """

    intersection = mat1.multiply(mat2)
    union = mat1 + mat2
    return intersection.count_nonzero() / union.count_nonzero()


def instance_jaccard(
    profile1: Profile, profile2: Profile, neuron: bool = True
) -> float:
    """
    Computes the proportion of synapses(or neurons/neurons) of profile1 that
    belongs to profile2 synapses(or neurons/neurons)

    Parameters
    ----------
    profile1 : Profile
        Typically a single image profile
    profile2 : Profile
        Typically an aggregated profile of many images
    neuron : bool
        Set to True if wish to compute proportions in terms of neurons instead
         of synapses

    Returns
    -------
     : float
        The proportion of profile1 in profile2.
    """
    if profile1.num_inputs == 0 or profile2.num_inputs == 0:
        return 0
    if neuron:
        intersection = [
            profile1.neuron_counts[layer]
            .multiply(profile2.neuron_counts[layer])
            .count_nonzero()
            if layer in profile2.neuron_counts
            else 0
            for layer in profile1.neuron_counts
        ]
        aprofile_size = profile1.size
    else:
        intersection = [
            profile1.synapse_counts[layer]
            .multiply(profile2.synapse_counts[layer])
            .count_nonzero()
            if layer in profile2.synapse_counts
            else 0
            for layer in profile1.synapse_counts
        ]
        aprofile_size = profile1.num_synapses

    return sum(intersection) / aprofile_size


def avg_jaccard(
    profile1: Profile, profile2: Profile, neuron: bool = True, layers: List[int] = None
) -> float:
    """
    Computes the jaccard similarity at each layer using synapse sets (or
    neuron sets) then averages the values.

    Parameters
    ----------
    profile1 : Profile
    profile2 : Profile
    neuron : bool, optional, default=False
        Set to true if wish to compute the iou on the neuron sets instead
        of the synapse sets
    layers: list, optional, deafult=None
        Specify a list of layers to calculate similarity over, defaults
        to all layers of the profile

    Returns
    -------
     : float
        Mean Intersection-over-Union (IOU) across layers of synapse (neuron) sets
        in Profile object.

    See also
    --------
    jaccard_simple
    """
    if profile1.num_inputs == 0 or profile2.num_inputs == 0:
        return 0

    if neuron:
        layers = layers or profile1.neuron_counts.keys()
        aprofile = profile1.neuron_counts
        bprofile = profile2.neuron_counts

    else:
        layers = layers or profile1.synapse_counts.keys()
        aprofile = profile1.synapse_counts
        bprofile = profile2.synapse_counts

    iou = [
        jaccard_simple(aprofile[layer], bprofile[layer]) if layer in bprofile else 0
        for layer in layers
    ]

    return np.mean(iou)


def jaccard(
    profile1: Profile, profile2: Profile, neuron: bool = True, layers: List[int] = None
) -> float:
    """
    Computes the jaccard similarity metric between two profiles using
    the aggregation of all synapse sets (or neuron set across all layers

    Parameters
    ----------
    profile1 : Profile
    profile2 : Profile
    neuron : bool, optional, default=False
        Set to true if wish to compute the jaccard on the neuron sets instead
        of the synapse sets
    layers: list, optional, default=None
        Specify a list of layers to calculate similarity over, defaults
        to all layers of the profile

    Returns
    -------
     : float

    See also
    --------
    jaccard_simple
    """
    if profile1.num_inputs == 0 or profile2.num_inputs == 0:
        return 0
    if neuron:
        layers = layers or (
            profile1.neuron_counts.keys() | profile2.neuron_counts.keys()
        )
        aprofile = sp.block_diag(
            [
                profile1.neuron_counts[layer]
                if layer in profile1.neuron_counts
                else sp.coo_matrix(profile2.neuron_counts[layer].shape)
                for layer in layers
            ]
        )
        bprofile = sp.block_diag(
            [
                profile2.neuron_counts[layer]
                if layer in profile2.neuron_counts
                else sp.coo_matrix(profile1.neuron_counts[layer].shape)
                for layer in layers
            ]
        )
    else:
        layers = layers or (
            profile1.synapse_counts.keys() | profile2.synapse_counts.keys()
        )
        aprofile = sp.block_diag(
            [
                profile1.synapse_counts[layer]
                if layer in profile1.synapse_counts
                else sp.coo_matrix(profile2.synapse_counts[layer].shape)
                for layer in layers
            ]
        )
        bprofile = sp.block_diag(
            [
                profile2.synapse_counts[layer]
                if layer in profile2.synapse_counts
                else sp.coo_matrix(profile1.synapse_counts[layer].shape)
                for layer in layers
            ]
        )

    return jaccard_simple(aprofile, bprofile)
