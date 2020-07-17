import numpy as np
import pytest


def test_profiler_model(torch_example):
    tp = torch_example
    x = tp.input
    y = tp.model.forward(x)
    yp, actives = tp.profiler.model.forward(x)
    assert np.allclose(yp.detach().numpy(), y.detach().numpy(), rtol=1e-04, atol=1e-4), f'failure: {v - y_out[0,ch,i,j]}'


def test_create_layers(torch_example):
    tp = torch_example
    m = tp.profiler.model.available_modules()
    assert len(m) == 39
    layerdict = tp.profiler.create_layers()
    assert len(layerdict) == 23
    layerdict = tp.profiler.create_layers(3)
    assert len(layerdict) == 4


def test_contrib_linear(torch_example):
    tp = torch_example
    ydx = (425,)
    x_in = tp.actives['avgpool']
    y_out = tp.actives['classifier.1']
    layers = tp.layerdict[3][0]
    nc, sc, sw = tp.profiler._contrib_linear(x_in, y_out, ydx, layers, threshold=0.1)
    assert nc[(479, 0, 6)] == 1
    assert sc[((479, 0, 6), (425,))] == 1
    assert sorted(list(sw))[0][-1] - 0.013844 < 10e-5
    assert len(sw) == 5


def test_contrib_adaptive_avg_pool2d(torch_example):
    tp = torch_example
    ydx = (479, 0, 6)
    x_in = tp.actives['features.30']
    y_out = tp.actives['avgpool']
    layers = tp.layerdict[4][0]
    nc, sc, sw = tp.profiler._contrib_adaptive_avg_pool2d(x_in, y_out, ydx, layers, threshold=0.1)
    assert nc[(479, 0, 6)] == 1
    assert sc[((479, 0, 6), (479, 0, 6))] == 1
    assert list(sw)[0][-1] == 1


def test_contrib_max2d(torch_example):
    tp = torch_example
    ydx = (479, 0, 6)
    x_in = tp.actives['features.29']
    y_out = tp.actives['features.30']
    layers = tp.layerdict[5][0]
    nc, sc, sw = tp.profiler._contrib_max2d(x_in, y_out, ydx, layers, threshold=0.1)
    assert nc[(479, 1, 13)] == 1
    assert sc[((479, 1, 13), (479, 0, 6))] == 1
    assert list(sw)[0][2] == 1


def test_contrib_conv2d(torch_example):
    tp = torch_example
    ydx = (479, 10, 13)
    x_in = tp.actives['features.27']
    y_out = tp.actives['features.29']
    layers = tp.layerdict[6][0]
    nc, sc, sw = tp.profiler._contrib_conv2d(x_in, y_out, ydx, layers, threshold=0.1)
    assert nc[(246, 11, 13)] == 1
    assert sc[((246, 11, 13), (479, 10, 13), (246, 2, 1))] == 1
    assert list(sw)[0][2] - 0.063957 < 10e-5


def test_create_profiles(torch_mini_example):
    tm = torch_mini_example
    profile = tm.profiler.create_profile(tm.input, tm.layerdict, threshold=0.1)
    assert len(profile.neuron_counts.keys()) == 3
    assert len(profile.synapse_counts[1]) == 7
    assert sorted(list(profile.synapse_weights[1]))[0][-1] - 0.020209 < 10e-5


def test_create_profiles_with_parallel(torch_mini_example):
    tm = torch_mini_example
    profile = tm.profiler.create_profile(tm.input, tm.layerdict, parallel=True, threshold=0.1)
    assert len(profile.neuron_counts.keys()) == 3
    assert len(profile.synapse_counts[1]) == 7
    assert sorted(list(profile.synapse_weights[1]))[0][-1] - 0.020209 < 10e-5
