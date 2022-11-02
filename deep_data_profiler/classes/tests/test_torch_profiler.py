import numpy as np
import torch


def test_profiler_model(torch_example):
    tp = torch_example
    x = tp.input
    y = tp.model.forward(x)
    yp, actives = tp.profiler.model.forward(x)
    assert torch.allclose(
        yp, y, rtol=1e-04, atol=1e-4
    ), f"failure: {v - y_out[0,ch,i,j]}"


def test_create_layers(element_example):
    tp = element_example
    m = tp.profiler.model.available_modules()
    assert len(m) == 39
    layerdict = tp.profiler.create_layers()
    assert len(layerdict) == 23



def test_contrib_linear(element_example, vgg16_linear):
    tp = element_example
    linear = vgg16_linear

    y_out = {linear.ldx: linear.module(linear.input.view(-1)).unsqueeze(0)}

    infl_neurons = torch.LongTensor([1104, 2560])

    layers = tp.layerdict[linear.ldx][0]

    nc, sc, sw = tp.profiler.contrib_linear(
        linear.x_in, y_out, infl_neurons, layers, threshold=0.1
    )
    assert np.allclose(nc.row, [20, 50])
    assert np.allclose(nc.col, [17, 23])
    assert np.allclose(sc.row, sw.row, [1104, 1104, 2560, 2560])
    assert np.allclose(sc.col, sw.col, [997, 2473, 997, 2473])
    assert np.allclose(sw.data.max(), 0.5381661, atol=10e-5)


def test_contrib_adaptive_avg_pool2d(element_example, vgg16_adaptive_avg_pool2d):
    tp = element_example
    avgpool = vgg16_adaptive_avg_pool2d

    y_out = {avgpool.ldx: avgpool.module(avgpool.input)}

    infl_neurons = torch.LongTensor([[20, 1, 1], [50, 6, 6]])

    layers = tp.layerdict[avgpool.ldx][0]

    nc, sc, sw = tp.profiler.contrib_adaptive_avg_pool2d(
        avgpool.x_in, y_out, infl_neurons, layers, threshold=0.1
    )

    assert np.allclose(nc.row, [20, 50])
    assert np.allclose(nc.col, [31, 195])
    assert np.allclose(sc.row, sw.row, [988, 2498])
    assert np.allclose(sc.col, sw.col, [3951, 9995])


def test_contrib_max2d(element_example, vgg16_max2d):
    tp = element_example
    maxpool = vgg16_max2d

    y_out = {maxpool.ldx: maxpool.module(maxpool.input)}

    infl_neurons = torch.LongTensor([[20, 1, 2], [50, 5, 4]])

    layers = tp.layerdict[maxpool.ldx][0]

    nc, sc, sw = tp.profiler.contrib_max2d(
        maxpool.x_in, y_out, infl_neurons, layers, threshold=0.1
    )

    assert np.allclose(nc.row, [20, 50])
    assert np.allclose(nc.col, [32, 149])
    assert np.allclose(sc.row, sw.row, [989, 2489])
    assert np.allclose(sc.col, sw.col, [3952, 9949])


def test_contrib_conv2d(element_example, vgg16_conv2d):
    tp = element_example
    conv = vgg16_conv2d

    y_out = {conv.ldx: conv.module(conv.input)}

    infl_neurons = torch.LongTensor([[120, 2, 3], [428, 2, 3]])

    layers = tp.layerdict[conv.ldx][0]

    nc, sc, sw = tp.profiler.contrib_conv2d(
        conv.x_in, y_out, infl_neurons, layers, threshold=0.1
    )
    assert np.allclose(nc.row, [20, 20])
    assert np.allclose(nc.col, [16, 18])
    assert np.allclose(sc.row, sw.row, [23551, 83919])
    assert np.allclose(sc.col, sw.col, [3936, 3938])
