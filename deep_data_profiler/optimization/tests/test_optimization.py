import pytest
import torch
from deep_data_profiler.optimization import dictionary_optimization


def test_optimization_dict(resnet18_example):
    model = resnet18_example.model
    # define signals dictionary
    signals_receptive_fields_weights = {
        "conv1": [((0, 1, 1)), ((20, 1, 1))],
        "layer1.0.conv2": [((50,)), ((13, 1, 1))],
    }

    # take two gradient steps with the dictionary optimization helper module
    output = dictionary_optimization(model, signals_receptive_fields_weights, threshold=2)
    assert output.shape == torch.Size([1, 3, 224, 224])