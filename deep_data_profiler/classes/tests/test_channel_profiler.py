import numpy as np
import pytest
import torch
import torch.nn.functional as F


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

def test_profiler_model(channel_example):
    tp = channel_example
    x = tp.input
    y = tp.model.forward(x)
    yp, actives = tp.profiler.model.forward(x)
    assert np.allclose(
        yp.detach().numpy(), y.detach().numpy(), rtol=1e-04, atol=1e-4
    ), f"failure: {v - y_out[0,ch,i,j]}"


def test_create_layers(channel_example):
    tp = channel_example
    m = tp.profiler.model.available_modules()
    assert len(m) == 39
    assert len(tp.layerdict) == 23


def test_contrib_linear(channel_example):
    tp = channel_example
    in_shape = tp.actives["avgpool"].shape
    out_shape = tp.actives["classifier.0"].shape

    x_in = torch.zeros(in_shape)
    y_out = torch.zeros(out_shape)

    x_in[:, 20, :5, :5] = a_block
    x_in[:, 50, 2:, 2:] = b_block

    linear_module = tp.profiler.model.available_modules()["classifier.0"]
    W = linear_module._parameters["weight"]
    B = linear_module._parameters["bias"]

    y_out = F.linear(x_in.view(-1), W, B).unsqueeze(0)

    x_in = {19: x_in}
    y_out = {20: y_out}

    infl_neurons = torch.Tensor([1104, 2560]).long()

    layers = tp.layerdict[20][0]
    nc, sc, sw = tp.profiler.contrib_linear(
        x_in, y_out, infl_neurons, layers, threshold=0.1
    )
    assert np.allclose(nc.col, [20, 50])
    assert np.allclose(nc.data, [2, 2])
    assert np.allclose(sc.row, sw.row, [1104, 1104, 2560, 2560])
    assert np.allclose(sc.col, sw.col, [20, 50, 20, 50])
    assert sw.data.max() - 0.5381661 < 10e-5


def test_contrib_adaptive_avg_pool2d(channel_example):
    tp = channel_example
    in_shape = (1, 512, 14, 14)
    out_shape = (1, 512, 7, 7)

    x_in = torch.zeros(in_shape)
    y_out = torch.zeros(out_shape)

    x_in[:, 20, :5, :5] = a_block
    x_in[:, 50, 9:, 9:] = b_block

    y_out = F.adaptive_avg_pool2d(x_in, out_shape[2:])

    x_in = {18: x_in}
    y_out = {19: y_out}

    infl_neurons = torch.Tensor([[20, 1, 1], [50, 6, 6]]).long()
    infl_channels = infl_neurons[:, 0]

    layers = tp.layerdict[19][0]

    nc, sc, sw = tp.profiler.contrib_adaptive_avg_pool2d(
        x_in, y_out, infl_channels, layers, threshold=0.1
    )

    assert np.allclose(nc.col, [20, 50])
    assert np.allclose(sc.row, sc.col, [20, 50])
    assert np.allclose(sw.row, sw.col, [20, 50])


def test_contrib_max2d(channel_example):
    tp = channel_example
    in_shape = tp.actives["features.29"].shape
    out_shape = tp.actives["features.30"].shape

    x_in = torch.zeros(in_shape)
    y_out = torch.zeros(out_shape)

    x_in[:, 20, :5, :5] = a_block
    x_in[:, 50, 9:, 9:] = b_block

    maxpool_module = tp.profiler.model.available_modules()["features.30"]
    kernel_size = maxpool_module.kernel_size

    y_out = F.max_pool2d(x_in, kernel_size)

    x_in = {17: x_in}
    y_out = {18: y_out}
    infl_neurons = torch.Tensor([[20, 1, 2], [50, 5, 4]]).long()
    infl_channels = infl_neurons[:, 0]

    layers = tp.layerdict[18][0]
    nc, sc, sw = tp.profiler.contrib_max2d(
        x_in, y_out, infl_channels, layers, threshold=0.1
    )
    assert np.allclose(nc.col, [20, 50])
    assert np.allclose(sc.row, sc.col, [20, 50])
    assert np.allclose(sw.row, sw.col, [20, 50])


def test_contrib_conv2d(channel_example):
    tp = channel_example
    in_shape = tp.actives["features.27"].shape
    out_shape = tp.actives["features.28"].shape

    x_in = torch.zeros(in_shape)
    y_out = torch.zeros(out_shape)

    x_in[:, 20, :5, :5] = a_block
    x_in[:, 50, 9:, 9:] = b_block

    conv_module = tp.profiler.model.available_modules()["features.28"]
    stride = conv_module.stride
    padding = conv_module.padding
    W = conv_module._parameters["weight"]
    B = conv_module._parameters["bias"]

    x_in = {16: x_in}
    y_out = {17: y_out}
    infl_neurons = torch.Tensor([[120, 2, 3], [428, 2, 3]]).long()
    infl_channels = infl_neurons[:, 0]

    layers = tp.layerdict[17][0]
    nc, sc, sw = tp.profiler.contrib_conv2d(
        x_in, y_out, infl_channels, layers, threshold=0.1
    )
    assert np.allclose(nc.col, [20])
    assert np.allclose(nc.data, [2])
    assert np.allclose(sc.row, sw.row, [120, 428])
    assert np.allclose(sc.col, sw.col, [20, 20])
