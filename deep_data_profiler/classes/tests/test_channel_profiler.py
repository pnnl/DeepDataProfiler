import numpy as np
import pytest
import torch


def test_profiler_model(channel_example):
    tp = channel_example
    x = tp.input
    y = tp.model.forward(x)
    yp, actives = tp.profiler.model.forward(x)
    assert torch.allclose(
        yp, y, rtol=1e-04, atol=1e-4
    ), f"failure: {v - y_out[0,ch,i,j]}"


def test_create_layers(channel_example):
    tp = channel_example
    m = tp.profiler.model.available_modules()
    assert len(m) == 39
    assert len(tp.layerdict) == 23


def test_contrib_linear(channel_example, vgg16_linear):
    tp = channel_example
    linear = vgg16_linear

    y_out = {linear.ldx: linear.module(linear.input.view(-1)).unsqueeze(0)}

    infl_neurons = torch.LongTensor([1104, 2560])

    layers = tp.layerdict[linear.ldx][0]

    nc, sc, sw = tp.profiler.contrib_linear(
        linear.x_in, y_out, infl_neurons, layers, threshold=0.1
    )
    assert np.allclose(nc.col, [20, 50])
    assert np.allclose(nc.data, [2, 2])
    assert np.allclose(sc.row, sw.row, [1104, 1104, 2560, 2560])
    assert np.allclose(sc.col, sw.col, [20, 50, 20, 50])
    assert np.allclose(sw.data.max(), 0.5381661, atol=10e-5)


def test_contrib_adaptive_avg_pool2d(channel_example, vgg16_adaptive_avg_pool2d):
    tp = channel_example
    avgpool = vgg16_adaptive_avg_pool2d

    y_out = {avgpool.ldx: avgpool.module(avgpool.input)}

    infl_neurons = torch.LongTensor([20, 50])

    layers = tp.layerdict[avgpool.ldx][0]

    nc, sc, sw = tp.profiler.contrib_adaptive_avg_pool2d(
        avgpool.x_in, y_out, infl_neurons, layers, threshold=0.1
    )

    assert np.allclose(nc.col, [20, 50])
    assert np.allclose(sc.row, sc.col, [20, 50])
    assert np.allclose(sw.row, sw.col, [20, 50])


def test_contrib_max2d(channel_example, vgg16_max2d):
    tp = channel_example
    maxpool = vgg16_max2d

    y_out = {maxpool.ldx: maxpool.module(maxpool.input)}

    infl_neurons = torch.LongTensor([20, 50])

    layers = tp.layerdict[maxpool.ldx][0]

    nc, sc, sw = tp.profiler.contrib_max2d(
        maxpool.x_in, y_out, infl_neurons, layers, threshold=0.1
    )
    assert np.allclose(nc.col, [20, 50])
    assert np.allclose(sc.row, sc.col, [20, 50])
    assert np.allclose(sw.row, sw.col, [20, 50])


def test_contrib_conv2d(channel_example, vgg16_conv2d):
    tp = channel_example
    conv = vgg16_conv2d

    y_out = {conv.ldx: conv.module(conv.input)}

    infl_neurons = torch.LongTensor([120, 428])

    layers = tp.layerdict[conv.ldx][0]

    nc, sc, sw = tp.profiler.contrib_conv2d(
        conv.x_in, y_out, infl_neurons, layers, threshold=0.1
    )
    assert np.allclose(nc.col, [20])
    assert np.allclose(nc.data, [2])
    assert np.allclose(sc.row, sw.row, [120, 428])
    assert np.allclose(sc.col, sw.col, [20, 20])
