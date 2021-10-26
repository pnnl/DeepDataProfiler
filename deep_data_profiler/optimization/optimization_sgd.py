from .input_feature import InputFeature

import torch
import numpy as np

from lucent.misc.io import show
from lucent.optvis.render import tensor_to_img_array
import lucent.optvis.transform as transform

from typing import Callable, Tuple
from tqdm import tqdm

from deep_data_profiler.utils import TorchHook


def optimization_fv(
    fv_object: InputFeature,
    model: TorchHook,
    objective: Callable[[torch.Tensor, str, Tuple[int]], torch.Tensor],
    threshold: int = 512,
) -> torch.Tensor:
    optimizer_fv = torch.optim.Adam([fv_object.fv_tensor], lr=5e-2)

    fvtransforms = transform.standard_transforms
    fvtransforms = fvtransforms.copy()

    fvtransforms.append(transform.normalize())

    transform_f = transform.compose(fvtransforms)

    for i in tqdm(range(1, threshold + 1),):

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
