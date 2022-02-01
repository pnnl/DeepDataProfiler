from .input_feature import InputFeature

import torch
import torch.nn.functional as F

import numpy as np
from lucent.misc.io import show
from lucent.optvis.render import tensor_to_img_array
import lucent.optvis.transform as transform
from typing import Callable, Tuple, Dict, Union, List, Optional
from tqdm import tqdm

from deep_data_profiler.utils import TorchHook
from deep_data_profiler import ChannelProfiler
from .svd_utils import project_svd
from .feature_helpers import (
    NeuronBasis,
    neurons_dictionary_objective,
    dictionary_objective,
)
from .input_feature import FeatureVizType


def dictionary_optimization(
    model: torch.nn.Module,
    layer_neuron_weights: Union[Dict[str, List[Tuple[int]]], Dict[str, List[int]]],
    device: Optional[torch.device] = None,
    threshold: int = 512,
    neuron_type: NeuronBasis = NeuronBasis.SVD,
    neuron: bool = True,
) -> torch.Tensor:
    """
    A convenience function for the optimization. Accepts a model
    (to be wrapped with TorchHook and Profiler), a dictionary of layer weights,
    the basis (namely, Euclidean or SVD), and whether to consider a single
    receptive field or the entire channel/signal.

    Parameters
    ----------
    model: torch.nn.Module
        PyTorch model.
    layer_neuron_weights: Union[Dict[str, List[Tuple[int]]], Dict[str, List[int]]]
        a dictionary of either {layer : [neurons...]} or
        {layer : [(neuron, weight), ...]}
    threshold: int
       Number of iterations in the optimization.
    Returns
    -------
    fv_object : torch.Tensor
        The optimized feature.
    """
    profiler = ChannelProfiler(model)
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # hook model
    model = TorchHook(model, device=device)
    # layers passed in dictionary
    layers = list(layer_neuron_weights.keys())
    model.add_hooks(layers)

    if NeuronBasis.SVD:
        svd_projection_model = project_svd(profiler)
    else:
        svd_projection_model = lambda x: x

    if neuron:
        objective = neurons_dictionary_objective(
            layer_neuron_weights,
            neuron_type,
            transform_activations=svd_projection_model,
        )
    else:
        objective = dictionary_objective(
            layer_neuron_weights,
            neuron_type,
            transform_activations=svd_projection_model,
        )

    fv_object = InputFeature(FeatureVizType.FFT_IMAGE, dims=(1, 3, 224, 224))

    optimizer_fv = torch.optim.Adam([fv_object.fv_tensor], lr=5e-2)

    fvtransforms = transform.standard_transforms
    fvtransforms = fvtransforms.copy()

    fvtransforms.append(transform.normalize())

    transform_f = transform.compose(fvtransforms)

    for i in tqdm(
        range(1, threshold + 1),
    ):

        # zero gradients
        optimizer_fv.zero_grad()

        # forward pass
        img_in = transform_f(fv_object())
        _, activations = model.forward(img_in)
        loss1 = objective(activations)

        loss1.backward()

        optimizer_fv.step()

    image = tensor_to_img_array(fv_object())
    show(image)
    return fv_object()


def optimization_fv(
    fv_object: InputFeature,
    model: TorchHook,
    objective: Callable[[torch.Tensor, str, Tuple[int]], torch.Tensor],
    threshold: int = 512,
) -> torch.Tensor:
    """
    A flexible function for the optimization. Accepts a model, an objective to optimize,
    and a threshold for the number of iterations. Returns the optimized feature.
    Parameters
    ----------
    fv_object: InputFeature
        Initial InputFeature to be optimized.
    model: TorchHook
        PyTorch model with hooks.
    objective: Callable
       An objective, most likely a FeatureObjective, that returns
       a scalar value (that we use to optime).
    threshold: int
       Number of iterations in the optimization.
    Returns
    -------
    fv_object : torch.Tensor
        The optimized feature.
    """
    optimizer_fv = torch.optim.Adam([fv_object.fv_tensor], lr=5e-2)

    fvtransforms = transform.standard_transforms
    fvtransforms = fvtransforms.copy()

    fvtransforms.append(transform.normalize())

    transform_f = transform.compose(fvtransforms)

    for i in tqdm(
        range(1, threshold + 1),
    ):

        # zero gradients
        optimizer_fv.zero_grad()

        # forward pass
        img_in = transform_f(fv_object())
        _, activations = model.forward(img_in)
        loss1 = objective(activations)

        loss1.backward()

        optimizer_fv.step()

    image = tensor_to_img_array(fv_object())
    show(image)
    return fv_object()


def optimization_fv_diversity(
    fv_object: InputFeature,
    model: TorchHook,
    objective: Callable[[torch.Tensor, str, Tuple[int]], torch.Tensor],
    layer: str,
    threshold: int = 512,
    diversity_term_weight: float = 1e2,
    steering_grad_weight: float = 1e-3,
) -> torch.Tensor:
    """
    Feature visualization with diversity. Expects the fv_object to be a batch of features.
    For the original formulation, see https://distill.pub/2017/feature-visualization/.
    For the re-weighting with diversity, see [our upcoming paper? / for now PP slides].

    Parameters
    ----------
    fv_object: InputFeature
        Initial InputFeature to be optimized.
    model: TorchHook
        PyTorch model with hooks.
    objective: Callable
       An objective, most likely a FeatureObjective, that returns
       a scalar value (that we use to optime).
    threshold: int
       Number of iterations in the optimization.
    diversity_term_weight: float
       Amount to weight the diversity objective in the optimization.
    steering_grad_weight: float
       Absolute amount to add to the gradient used to weight the
       diversity activations.
    Returns
    -------
    fv_object : torch.Tensor
        The optimized feature.
    """
    optimizer_fv = torch.optim.Adam([fv_object.fv_tensor], lr=5e-2)

    fvtransforms = transform.standard_transforms
    fvtransforms = fvtransforms.copy()

    fvtransforms.append(transform.normalize())

    transform_f = transform.compose(fvtransforms)

    for i in tqdm(
        range(1, threshold + 1),
    ):

        # zero gradients
        optimizer_fv.zero_grad()

        # forward pass
        img_in = transform_f(fv_object())
        _, activations = model.forward(img_in)
        image_activations = activations[layer]
        image_activations.retain_grad()
        loss1 = objective(activations)

        loss1.backward(retain_graph=True)

        ######### diversity term
        # grad used to weight diversity in direction of the objective
        steering_grad = torch.autograd.grad(loss1, image_activations)[0]

        _, activations = model.forward(img_in)
        image_activations = activations[layer]

        # re-weight the activations used in the diversity by the gradient
        layer_t = image_activations * (steering_grad + steering_grad_weight)

        # actual diversity objective: compute gram matrices, use negative
        # pairwise cosine similarity
        batch, channels, _, _ = layer_t.shape
        flattened = layer_t.view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        weighted_diversity = (
            diversity_term_weight
            * sum(
                [
                    sum([(grams[i] * grams[j]).sum() for j in range(batch) if j != i])
                    for i in range(batch)
                ]
            )
            / batch
        )
        weighted_diversity.backward()
        ######### diversity term

        optimizer_fv.step()

    image = tensor_to_img_array(fv_object())
    show(image)
    return fv_object()
