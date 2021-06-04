import pytest
import torch
import torchvision.models.vgg as vgg
import torchvision.models.resnet as resnet
import deep_data_profiler as ddp
from deep_data_profiler.algorithms import SpectralAnalysis
from collections import Counter
import scipy.sparse as sp
from scipy import stats
import numpy as np


class TorchExample:
    def __init__(self):
        torch.manual_seed(0)
        self.input = x = torch.randn(1, 3, 224, 224)
        self.model = vgg.vgg16(pretrained=True).to("cpu").eval()
        self.profiler = ddp.TorchProfiler(self.model)
        self.layerdict = self.profiler.create_layers()
        self.output, self.actives = self.profiler.model.forward(x)


@pytest.fixture
def torch_example():
    return TorchExample()


class ProfileExample:
    def __init__(self):
        self.neuron_counts = {1: sp.coo_matrix([[2], [1], [1], [1]])}
        self.synapse_counts = {1: sp.coo_matrix([[1, 1], [1, 0], [1, 0], [1, 0]])}
        self.synapse_weights = {
            1: sp.coo_matrix([[0.1, 0.2], [0.3, 0], [0.4, 0], [0.5, 0]])
        }
        self.num_inputs = 1


@pytest.fixture
def profile_example():
    return ProfileExample()


class SpectralExample:
    def __init__(self, model):
        self.model = model
        self.spectral_profile = SpectralAnalysis(self.model)
        self.eigdict = self.spectral_profile.spectral_analysis()
        self.alpha_dict = self.spectral_profile.fit_power_law(eig_dict=self.eigdict)
        self.layer_rmt = self.spectral_profile.layer_RMT(alpha_dict=self.alpha_dict)
        self.universal_metric = self.spectral_profile.universal_metric(
            alpha_dict=self.alpha_dict
        )


class MiniSpectralModel(torch.nn.Module):
    r"""A tiny model to test SpectralAnalysis. This model
    is not meant to do anything else: there is no forward method!"""

    def __init__(self) -> None:
        super(MiniSpectralModel, self).__init__()
        # define the linear layer
        linear_layer = torch.nn.Linear(in_features=10, out_features=10, bias=False)
        linear_layer.weight.data = (
            torch.eye(10) * torch.Tensor(list(range(1, 11)))[None, :]
        )
        self.linear_layer = linear_layer

        # define the convolutional layer
        conv_layer = torch.nn.Conv2d(
            in_channels=10, out_channels=10, kernel_size=1, bias=False
        )
        new_weights = torch.eye(10) * torch.Tensor(list(range(1, 11)))[None, :]
        shape = conv_layer.weight.data.shape
        conv_layer.weight.data = new_weights.reshape(shape)
        self.conv_layer = conv_layer


@pytest.fixture
def spectral_resnet_example():
    model = resnet.resnet18(pretrained=True).to("cpu").eval()
    return SpectralExample(model)


@pytest.fixture
def spectral_mini_example():
    model = MiniSpectralModel()
    return SpectralExample(model)


class AlphaFitExample:
    def __init__(self, a: float = 2.0) -> None:
        # creates a powerlaw pdf to test our fit on
        x = np.linspace(stats.powerlaw.ppf(0.01, a), stats.powerlaw.ppf(0.99, a), 300)
        self.alpha_dict = {1: (stats.powerlaw.pdf(x, a), 1)}


@pytest.fixture
def alpha_example_reasonable():
    return AlphaFitExample(2)


@pytest.fixture
def alpha_example_unreasonable():
    return AlphaFitExample(10)
