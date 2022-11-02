import numpy as np
import pytest
import torch


def test_profiler_model(svd_example):
    tp = svd_example
    profiled = tp.profiler.create_influential(torch.zeros(1, 3, 224, 224))
    nc_1 = profiled.neuron_counts[1].todense().argmax()
    nc_2 = profiled.neuron_counts[11].todense().sum()
    nw_1 = profiled.neuron_weights[22].todense().sum()
    assert nc_1 == 14, f"svd counts failure"
    assert nc_2 == 8, f"svd counts failure"
    assert np.isclose(nw_1, 1.4163294), f"svd weights failure"


def test_create_layers(svd_example):
    tp = svd_example
    m = tp.profiler.model.available_modules()
    assert len(m) == 39
    assert len(tp.layerdict) == 23
