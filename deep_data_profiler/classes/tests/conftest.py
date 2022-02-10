import deep_data_profiler as ddp
from deep_data_profiler.algorithms import SpectralAnalysis
import numpy as np
import pytest
from scipy import stats
import scipy.sparse as sp
import torch
import torchvision.models.resnet as resnet
import torchvision.models.vgg as vgg


class HookExample:
    def __init__(self, model):
        torch.manual_seed(0)
        self.input = torch.randn(1, 3, 224, 224)
        self.model = model
        self.hooks = ddp.TorchHook(model)
        self.hooks.add_hooks(self.hooks.module_dict.keys())
        self.output, self.actives = self.hooks.forward(self.input)
        self.actives_dims = {layer: act.shape for layer, act in self.actives.items()}


@pytest.fixture
def vgg16_hook():
    model = vgg.vgg16(pretrained=True)
    return HookExample(model)


@pytest.fixture
def resnet18_hook():
    model = resnet.resnet18(pretrained=True)
    return HookExample(model)


class TorchExample:
    def __init__(self, modelhooks, proftype):
        torch.manual_seed(0)
        self.input = modelhooks.input
        self.model = modelhooks.model
        if proftype == "element":
            self.profiler = ddp.ElementProfiler(self.model)
        elif proftype == "channel":
            self.profiler = ddp.ChannelProfiler(self.model)
        elif proftype == "spatial":
            self.profiler = ddp.SpatialProfiler(self.model)
        elif proftype == "svd":
            self.profiler = ddp.SVDProfiler(self.model)
        self.layerdict = self.profiler.layerdict
        self.output = modelhooks.output
        self.actives = modelhooks.actives


@pytest.fixture
def element_example(vgg16_hook):
    return TorchExample(vgg16_hook, "element")


@pytest.fixture
def channel_example(vgg16_hook):
    return TorchExample(vgg16_hook, "channel")


@pytest.fixture
def spatial_example(vgg16_hook):
    return TorchExample(vgg16_hook, "spatial")


@pytest.fixture
def svd_example(vgg16_hook):
    return TorchExample(vgg16_hook, "svd")


class ProfileToyExample:
    def __init__(self):
        self.neuron_counts = {1: sp.coo_matrix([[2], [1], [1], [1]])}
        self.synapse_counts = {1: sp.coo_matrix([[1, 1], [1, 0], [1, 0], [1, 0]])}
        self.synapse_weights = {
            1: sp.coo_matrix([[0.1, 0.2], [0.3, 0], [0.4, 0], [0.5, 0]])
        }
        self.num_inputs = 1


@pytest.fixture
def profile_toy_example():
    return ProfileToyExample()


class ProfileExample:
    def __init__(self, profiler_example):
        self.input = profiler_example.input
        self.profiler = profiler_example.profiler
        self.profile = self.profiler.create_profile(self.input)


@pytest.fixture
def element_profile(element_example):
    return ProfileExample(element_example)


@pytest.fixture
def channel_profile(channel_example):
    return ProfileExample(channel_example)


@pytest.fixture
def spatial_profile(spatial_example):
    return ProfileExample(spatial_example)


class LayerExample:
    def __init__(self, module, in_shape, x_ldx, y_ldx, aloc=(20, 5), bloc=(50, 9)):
        a_block = torch.Tensor(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [2, 4, 6, 8, 10],
                [1, 3, 5, 7, 9],
                [3, 5, 2, 4, 1],
            ]
        )
        b_block = torch.Tensor(
            [
                [5, 4, 3, 2, 1],
                [12, 8, 4, 0, 2],
                [3, 3, 3, 3, 3],
                [4, 5, 6, 4, 5],
                [2, 3, 4, 8, 9],
            ]
        )

        self.input = torch.zeros(in_shape)
        self.input[:, aloc[0], : aloc[1], : aloc[1]] = a_block
        self.input[:, bloc[0], bloc[1] :, bloc[1] :] = b_block
        self.x_in = {x_ldx: self.input}
        self.ldx = y_ldx
        self.module = module


@pytest.fixture
def vgg16_linear(vgg16_hook):
    model = vgg16_hook
    return LayerExample(
        model.hooks.module_dict["classifier.0"],
        model.actives_dims["avgpool"],
        19,
        20,
        bloc=(50, 2),
    )


@pytest.fixture
def vgg16_adaptive_avg_pool2d(vgg16_hook):
    model = vgg16_hook
    dims = model.actives_dims["avgpool"]
    return LayerExample(
        model.hooks.module_dict["avgpool"],
        dims[:2] + (dims[-1] * 2,) * 2,
        18,
        19,
    )


@pytest.fixture
def vgg16_max2d(vgg16_hook):
    model = vgg16_hook
    return LayerExample(
        model.hooks.module_dict["features.30"],
        model.actives_dims["features.29"],
        17,
        18,
    )


@pytest.fixture
def vgg16_conv2d(vgg16_hook):
    model = vgg16_hook
    return LayerExample(
        model.hooks.module_dict["features.28"],
        model.actives_dims["features.27"],
        16,
        17,
    )


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
