from typing import Tuple, Union, Optional
from torch import Tensor
import numpy as np


def marchpast_layer_fit(eigs: np.array, aspect_ratio: float) -> Tuple[np.array]:
    """
    Plots the Marchenko–Pastur distribution fit
    for an empirical spectral distribution.
    See, e.g. https://arxiv.org/abs/1506.04922

    Parameters
    ----------
    eigs : np.array
        Binned eigenvalues
    aspect_ratio : float
        Aspect ratio of NxM matrix

    Returns
    -------
    x : np.array
        intervals on the x-axis for the (discrete) plot
    mp : np.array
        Marchenko–Pastur distribution fit
    """
    x_min, x_max = 0, np.max(eigs)

    # calculate sigma
    lambda_max = np.max(eigs)
    inver_aspect = 1.0 / np.sqrt(aspect_ratio)
    sigma = np.sqrt(lambda_max / np.square(1 + inver_aspect))

    x, mp = marchenko_pastur_pdf(x_min, x_max, aspect_ratio, sigma)
    return x, mp


def aspect_ratio(W: Union[np.array, Tensor]) -> float:
    """
    Grabs the aspect ratio N/M of a matrix, enforcing N > M.

    Parameters ----------
    W : Union[np.array, Tensor]
        A matrix

    Returns
    -------
    aspect_ratio : float
        Aspect ratio of NxM matrix
    """
    if W.shape[1] > W.shape[0]:
        M, N = W.shape
    else:
        N, M = W.shape
    Q = N / M
    return Q, N


def marchenko_pastur_pdf(
    x_min: float, x_max: float, aspect_ratio: float, sigma: Optional[float] = 1
) -> Tuple[np.array]:
    r"""
    Computes the MP PDF given the range, aspect ratio, and sigma.
    Parameters
    ----------
    x_min : float
    x_max : float
    aspect_ratio : float
    sigma : Optional[float]

    Returns
    -------
    x : np.array
        intervals on the x-axis for the (discrete) plot
    mp : np.array
        Marchenko–Pastur distribution fit
    """
    y = 1 / aspect_ratio
    x = np.linspace(x_min, x_max, 1000)

    # max eigenvalue
    b = np.power(sigma * (1 + np.sqrt(1 / aspect_ratio)), 2)
    # min eigenvalue
    a = np.power(sigma * (1 - np.sqrt(1 / aspect_ratio)), 2)
    return x, (1 / (2 * np.pi * sigma * sigma * x * y)) * np.sqrt((b - x) * (x - a))
