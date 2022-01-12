import numpy as np
import pytest
import torch

def test_profiler_model(spatial_example):
    tp = spatial_example
    x = tp.input
    y = tp.model.forward(x)
    yp, actives = tp.profiler.model.forward(x)
    assert torch.allclose(
        yp, y, rtol=1e-04, atol=1e-4
    ), f"failure: {v - y_out[0,ch,i,j]}"


def test_create_layers(spatial_example):
    tp = spatial_example
    m = tp.profiler.model.available_modules()
    assert len(m) == 39
    assert len(tp.layerdict) == 23

def test_contrib_linear(spatial_example, vgg16_linear):
    tp = spatial_example
    linear = vgg16_linear

    y_out = {linear.ldx: linear.module(linear.input.view(-1)).unsqueeze(0)}

    infl_neurons = torch.LongTensor([1104, 2560])

    layers = tp.layerdict[linear.ldx][0]

    nc, sc, sw = tp.profiler.contrib_linear(
        linear.x_in, y_out, infl_neurons, layers, threshold=0.1
    )

    assert np.allclose(nc.col, [17, 23])
    assert np.allclose(nc.data, [2, 2])
    assert np.allclose(sc.row, sw.row, [1104, 2560, 1104, 2560])
    assert np.allclose(sc.col, sw.col, [17, 17, 23, 23])
    assert np.allclose(sw.data.max(), 0.5381661, atol=10e-5)


def test_contrib_adaptive_avg_pool2d(spatial_example, vgg16_adaptive_avg_pool2d):
    tp = spatial_example
    avgpool = vgg16_adaptive_avg_pool2d

    y_out = {avgpool.ldx: avgpool.module(avgpool.input)}

    infl_neurons = torch.LongTensor([48, 8])

    layers = tp.layerdict[avgpool.ldx][0]

    nc, sc, sw = tp.profiler.contrib_adaptive_avg_pool2d(
        avgpool.x_in, y_out, infl_neurons, layers, threshold=0.1
    )

    assert np.allclose(nc.col, [30,31,44,45,180,181,194,195])
    assert np.allclose(sc.row, sw.row, [48,48,48,48,8,8,8,8])
    assert np.allclose(sc.col, sw.col, [180,181,194,195,30,31,44,45])
    assert np.allclose(sw.max(),0.3461538, atol=10e-5)

def test_contrib_max2d(spatial_example, vgg16_max2d):
    tp = spatial_example
    maxpool = vgg16_max2d

    y_out = {maxpool.ldx: maxpool.module(maxpool.input)}

    infl_neurons = torch.LongTensor([39, 2])

    layers = tp.layerdict[maxpool.ldx][0]

    nc, sc, sw = tp.profiler.contrib_max2d(
        maxpool.x_in, y_out, infl_neurons, layers, threshold=0.1
    )

    assert np.allclose(nc.col, [4,5,18,19,148,149,162,163])
    assert np.allclose(sc.row, sw.row, [39,39,39,39,2,2,2,2])
    assert np.allclose(sc.col, sw.col, [148,149,162,163,4,5,18,19])
    assert np.allclose(sw.data, 0.25)

def test_contrib_conv2d(spatial_example, vgg16_conv2d):
    tp = spatial_example
    conv = vgg16_conv2d

    y_out = {conv.ldx: conv.module(conv.input)}

    infl_neurons = torch.LongTensor([31, 17])

    layers = tp.layerdict[conv.ldx][0]

    nc, sc, sw = tp.profiler.contrib_conv2d(
        conv.x_in, y_out, infl_neurons, layers, threshold=0.1
    )

    assert np.allclose(nc.col, [18, 32])
    assert np.allclose(sc.row, sw.row, [31, 17])
    assert np.allclose(sc.col, sw.col, [32, 18])
    assert np.allclose(sw.max(), 0.1703450, atol=10e-5)
