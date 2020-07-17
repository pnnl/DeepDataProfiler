import pytest
import pickle
import torch
import torchvision.models.vgg as vgg
import itertools as it
import deep_data_profiler as ddp
from collections import Counter


class TorchExample():
    def __init__(self, nlayers):
        torch.manual_seed(0)
        self.input = x = torch.randn(1, 3, 224, 224)
        self.model = vgg.vgg16(pretrained=True).to('cpu').eval()
        self.profiler = ddp.TorchProfiler(self.model)
        self.layerdict = self.profiler.create_layers(nlayers)
        self.output, self.actives = self.profiler.model.forward(x)


@pytest.fixture
def torch_example():
    return TorchExample(6)


@pytest.fixture
def torch_mini_example():
    return TorchExample(2)


class ProfileExample():
    def __init__(self):
        self.neuron_counts = {0: Counter({1: 2, 2: 1, 3: 1, 4: 1})}
        self.synapse_counts = {0: Counter({(1, 0): 2, (2, 0): 1, (3, 0): 1, (4, 0): 1})}
        self.synapse_weights = {0: {(1, 0, 0.1), (1, 0, 0.2), (2, 0, 0.3), (3, 0, 0.4), (4, 0, 0.5)}}
        self.num_inputs = 1


class ProfileExample2():
    def __init__(self):
        ex1 = [1, 2, 2, 3, 5]
        ex2 = [2, 3, 4, 5, 6]
        ex3 = [2, 2, 2, 4, 4]
        nc1 = {0: Counter([(0,)]), 1: Counter(ex1)}
        nc2 = {0: Counter([(0,)]), 1: Counter(ex2)}
        nc3 = {0: Counter([(1,)]), 1: Counter(ex3)}
        sc1 = {0: Counter([(0, 0)]), 1: Counter([(x, 0) for x in ex1])}
        sc2 = {0: Counter([(0, 0)]), 1: Counter([(x, 0) for x in ex2])}
        sc3 = {0: Counter([(1, 1), (0, 0)]), 1: Counter([(x, 1) for x in ex3] + [(x, 0) for x in ex3])}
        sw1 = {0: {(0, 0, 1)}, 1: {(x[0], x[1], .1 * idx + .1) for idx, x in enumerate(sc1[1])}}
        sw2 = {0: {(0, 0, 1)}, 1: {(x[0], x[1], .2 * idx + .1) for idx, x in enumerate(sc2[1])}}
        sw3 = {0: {(1, 1, 1)}, 1: {(x[0], x[1], .2 * idx + .1) for idx, x in enumerate(sc3[1])}}
        self.p1 = ddp.Profile(neuron_counts=nc1, synapse_counts=sc1, synapse_weights=sw1, num_inputs=1)
        self.p2 = ddp.Profile(neuron_counts=nc2, synapse_counts=sc2, synapse_weights=sw2, num_inputs=1)
        self.p3 = ddp.Profile(neuron_counts=nc3, synapse_counts=sc3, synapse_weights=sw3, num_inputs=1)


@pytest.fixture
def profile_example():
    return ProfileExample()


@pytest.fixture
def profile_example2():
    return ProfileExample2()
