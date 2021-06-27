import torch
import powerlaw
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from typing import Dict, Tuple, List, Optional
from deep_data_profiler.utils import aspect_ratio, get_children


class SpectralAnalysis:

    """
    Spectral Analysis is based on methods originating from Random Matrix theory,
    brought to deep neural networks by Martin and Mahoney.  `Traditional and Heavy-Tailed Self Regularization in Neural Network Models <https://arxiv.org/abs/1901.08276/>`_ by Martin and Mahoney
    `Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data <https://arxiv.org/abs/2002.06716/>`_ by Martin, Peng,and Mahoney

    These methods act only on the weights of the Fully Connected and
    Convolutional layers a deep neural network. Despite this, they have
    proven effective in predicting
    1. Test accuracies with no access to the data distribution on which it was trained OR tested
    2. Relative performance between models of similar architecture classes
    3. Model and architecture improvements while training



    The major improvement we make over the above work is our handling of
    convolutional layers: our methods are more principled, and over an
    order of magnitude faster than the code released by the authors in
    https://github.com/CalculatedContent/WeightWatcher.

    Attributes
    ----------
    implemented_classes : set
        List of classes in PyTorch we can examine,
        i.e. have implemented spectral distributions
    model : torch.nn.Module()
        model to be spectral-analyzed
    """

    def __init__(self, model: torch.nn.Module) -> None:

        super().__init__()

        self.implemented_classes = {
            torch.nn.Linear,
            torch.nn.Conv2d,
        }

        if isinstance(model, Iterable):
            self.model = model
        else:
            self.model = torch.nn.Sequential(model)

    def __repr__(self) -> str:
        if self.model.__class__.__name__:
            repr_name = f"SpectralAnalysis for a {self.model.__class__.__name__}"
        else:
            repr_name = f"SpectralAnalysis for a {self.model}"
        return repr_name

    def spectral_analysis(
        self, plot: bool = False
    ) -> Dict[int, Tuple[np.array, float]]:
        """
        Returns a dictionary keyed by the order of
        the linear and convolutional layers, with the
        eigenvalues of :math:`X = W W^T`.
        Optional plot of the spectrum.

        Parameters
        ----------
        plot: bool
            Plot per-layer empirical spectral distribution.

        Returns
        -------
        eigdict: Dict[int, Tuple[float, float]]
            Dictionary with keys of the nth layer proviled,
            values of :attr:`(eigenvalues, Q)`, where :attr:`eigenvalues`
            are those of the weight matrix for the layer, and :attr:`Q`
            is the aspect ratio of the matrix.
        """
        eigdict = {}
        all_ops = get_children(self.model)
        operation_list = [op for op in all_ops if type(op) in self.implemented_classes]

        for idx, layer in enumerate(operation_list, 1):
            if type(layer) == torch.nn.modules.linear.Linear:
                X = layer._parameters["weight"].detach()
                X_linear = layer._parameters["weight"].detach()
                # compute aspect ratio
                Q, N = aspect_ratio(X_linear)
                # calculate the singular values with jax
                sigma = (
                    torch.svd(
                        X_linear,
                        compute_uv=False,
                    )
                    .S.cpu()
                    .numpy()
                )
                # square to get eigenvalues of W = X^TX
                eigenvalues = np.asarray(sigma * sigma) / len(X_linear)

            elif type(layer) == torch.nn.modules.conv.Conv2d:
                X = layer._parameters["weight"].detach()
                reshape_tens = torch.flatten(X, start_dim=1, end_dim=-1)
                # compute aspect ratio
                Q, N = aspect_ratio(reshape_tens)
                sigma = (
                    torch.svd(
                        reshape_tens,
                        compute_uv=False,
                    )
                    .S.cpu()
                    .numpy()
                )
                # square to get eigenvalues of W = X^TX
                eigenvalues = np.asarray(sigma * sigma) / len(reshape_tens)

            eigdict[idx] = (eigenvalues, Q)

            if plot:
                plt.hist(eigenvalues, bins="auto", density=True)
                plt.ylabel("ESD")
                plt.xlabel("Eigenvalues of $X$")
                plt.title(f"Layer {idx} spectrum")
                plt.show()

        return eigdict

    def fit_power_law(
        self,
        eig_dict=None,
        plot_alpha: bool = False,
        plot_eig=False,
    ) -> Dict[int, Tuple[float, float]]:
        r"""
        Fits the eigenvalue spectrum distribution of
        the layer weights :math:`X = W W^T` with a power-law distribution.
        Uses the MLE approach from https://arxiv.org/abs/0706.1062.

        Parameters
        ----------
        eigdict: Dict[int, Tuple[np.array, float]]
            Optional, useful if pre-computed with `.spectral_analysisr()`
            Dictionary with keys of the nth layer proviled,
            values of :attr:`(eigenvalues, Q)`, where :attr:`eigenvalues`
            are those of the weight matrix for the layer, and :attr:`Q`
            is the aspect ratio of the matrix.
        plot_alpha: bool
            Plot per-layer power-law fit of the
            eigenvalue spectrum distribution.
        plot_eig: bool
            Plot per-layer eigenvalue spectrum distribution

        Returns
        -------
        alpha_dict: Dict[int, Tuple[float, float]]
            Dictionary with keys of the nth layer proviled,
            values of `(alpha, eig_max)`, where `alpha`
            is the power law fit alpha, i.e:
            :math: \rho(\lambda) \sim \lambda^{-\alpha}.
            `eig_max` is the max eigenvalue.
        """
        if not eig_dict:
            eig_dict = self.spectral_analysis(plot=plot_eig)
        all_layers = list(eig_dict.keys())
        alpha_dict = {}
        for layer in all_layers:
            eigenvalues, Q = eig_dict[layer]
            eig_max = np.max(eigenvalues)
            results = powerlaw.Fit(eigenvalues, verbose=False)
            alpha = results.power_law.alpha
            alpha_dict[layer] = alpha, eig_max
            if plot_alpha:
                results.plot_pdf(color="b")
                results.power_law.plot_pdf(
                    color="r", linewidth=2, linestyle="--"
                )  # noqa
                plt.title(
                    f"Linear layer {layer} power law fit \n alpha = {round(alpha, 3)}"
                )  # noqa
                plt.ylabel("Spectral density (log)")
                plt.xlabel("Eigenvalues of $W_{FC}W_{FC}^T$ (log)")
                plt.show()
        return alpha_dict

    def layer_RMT(
        self,
        alpha_dict: Optional[Dict[int, Tuple[float, float]]] = None,
        verbose: bool = False,
        plot_alpha: bool = False,
        plot_eig: bool = False,
    ) -> Tuple[List[str], Dict]:
        """Prints the random matrix theory phenomenology of
        the layer eigenspectrum distribution from :math:`X = W W^T`.
        From https://arxiv.org/abs/1901.08276

        Parameters
        ----------
        alpha_dict: Dict[int, Tuple[float, float]]
            Optional, useful if pre-computed with `.spectral_analysisr()`
            Dictionary with keys of the nth layer proviled,
            values of `(alpha, eig_max)`
        plot_alpha: bool
            Plot per-layer power-law fit of the
            eigenvalue spectrum distribution.
        plot_eig: bool
            Plot per-layer eigenvalue spectrum distribution
        """

        layer_proclamations = []
        if not alpha_dict:
            alpha_dict = self.fit_power_law(plot_alpha=plot_alpha, plot_eig=plot_eig)
        layers = list(alpha_dict.keys())
        for layer in layers:
            # get PL coeff for layer, and convert to mu
            alpha, _ = alpha_dict[layer]
            mu = (alpha - 1) * 2
            # rough definition of phenomenology
            if verbose:
                print(f"Phenomenology for layer {layer}")
            # if 0 <= mu <= 2:
            if 0 <= mu <= 4:
                text = (
                    f"Layer {layer} prediction: regularized and "
                    + "good performance on a test set."
                )
                layer_proclamations.append(text)
                if verbose:
                    print(text)
                # print("(Very Heavy Tailed ESD)")
            # elif 2 < mu <= 4:
            #     print("Heavy-Tailed. Predict regularized and well-trained")
            elif mu > 4 and mu < 7:
                text = (
                    f"Layer {layer} prediction: somewhat well regularized "
                    + "and likely has good performance on a test set."
                )
                layer_proclamations.append(text)
                if verbose:
                    print(text)
                # print("Weakly Heavy-Tailed.")
            else:
                text = (
                    f"Layer {layer} prediction: very likely not trained "
                    + "and not regularized."
                )
                layer_proclamations.append(text)
                if verbose:
                    print(text)
        return layer_proclamations

    def universal_metric(self, alpha_dict=None) -> float:
        r"""
        Returns the universal capacity metric
        :math:`\widehat{\alpha}=\frac{1}{L} \sum_{l} \alpha_{l} \log \lambda_{\max , l}` from
        https://arxiv.org/abs/2002.06716

        Parameters
        ----------
        alpha_dict: Dict[int, Tuple[float, float]]
            Optional, useful if pre-computed with `.spectral_analysisr()`
            Dictionary with keys of the nth layer proviled,
            values of `(alpha, eig_max)`

        Returns
        -------
        metric: float
            Universal capacity metric. A useful engineering metric
            for average case capacity in DNNs, from
            https://arxiv.org/abs/1901.08278
        """
        if not alpha_dict:
            alpha_dict = self.fit_power_law(plot_alpha=False)
        if len(alpha_dict):
            metric = sum(
                alpha * np.log(eig_max) for alpha, eig_max in alpha_dict.values()
            ) / len(alpha_dict)
        else:
            metric = None

        return metric
