import pytest
import torch
from deep_data_profiler.optimization import (
    dictionary_optimization,
    dictionary_objective,
    neurons_dictionary_objective,
    optimization_fv,
    NeuronBasis,
    project_svd,
    FeatureVizType,
    InputFeature,
    ChannelObjective,
    SVDMean,
)


def test_optimization_dict(resnet18_example):
    """Test for neuron optimization with a dictionary of signals, in the default basis"""
    model = resnet18_example.model
    # define signals dictionary
    signals_receptive_fields_weights = {
        "conv1": [((0, 1, 1)), ((20, 1, 1))],
        "layer1.0.conv2": [((50,)), ((13, 1, 1))],
    }

    # take two gradient steps with the dictionary optimization helper module
    output = dictionary_optimization(
        model, signals_receptive_fields_weights, threshold=2
    )

    assert output.shape == torch.Size([1, 3, 224, 224])


def test_optimization_dict_activation(resnet18_example):
    """Test for neuron optimization with a dictionary of signals, in the activation/euclidean hidden layer basis"""
    model = resnet18_example.model
    # define signals dictionary
    signals_receptive_fields_weights = {
        "conv1": [((0, 1, 1)), ((20, 1, 1))],
        "layer1.0.conv2": [((50,)), ((13, 1, 1))],
    }

    # take two gradient steps with the dictionary optimization helper module
    output = dictionary_optimization(
        model,
        signals_receptive_fields_weights,
        threshold=2,
        neuron_type=NeuronBasis.ACTIVATION,
    )

    assert output.shape == torch.Size([1, 3, 224, 224])


def test_optimization_dict_channel(resnet18_example):
    """Test for channel optimization with a dictionary of signals, in the default basis"""
    model = resnet18_example.model
    # define signals dictionary
    signals_receptive_fields_weights = {"conv1": [0, 20], "layer1.0.conv2": [50, 13]}
    # take two gradient steps with the dictionary optimization helper module
    # set neuron to false to use channel objective
    output = dictionary_optimization(
        model, signals_receptive_fields_weights, threshold=2, neuron=False
    )

    assert output.shape == torch.Size([1, 3, 224, 224])


def test_optimization_dict_channel(resnet18_example):
    """Test for channel optimization with a dictionary of signals, in the activation/euclidean hidden layer basis"""
    model = resnet18_example.model
    # define signals dictionary
    signals_receptive_fields_weights = {"conv1": [0, 20], "layer1.0.conv2": [50, 13]}
    # take two gradient steps with the dictionary optimization helper module
    # set neuron to false to use channel objective
    output = dictionary_optimization(
        model,
        signals_receptive_fields_weights,
        threshold=2,
        neuron_type=NeuronBasis.ACTIVATION,
        neuron=False,
    )

    assert output.shape == torch.Size([1, 3, 224, 224])


def test_optimization_simple_fft_channel(resnet18_example):
    """Test for channel optimization with the image in the Fourier basis, in the activation/euclidean hidden layer basis"""
    hooked_model = resnet18_example.hooked_model
    hooked_model.add_hooks(["conv1", "layer1.0.conv2"])
    objective1 = ChannelObjective("layer1.0.conv2", coord=5)
    feature_image = InputFeature(FeatureVizType.FFT_IMAGE, dims=(1, 3, 224, 224))
    # take two gradient ascent steps
    output1 = optimization_fv(feature_image, hooked_model, objective1, threshold=2)

    objective2 = ChannelObjective("conv1", coord=5)
    feature_image = InputFeature(FeatureVizType.FFT_IMAGE, dims=(1, 3, 224, 224))
    # take two gradient ascent steps
    output2 = optimization_fv(feature_image, hooked_model, objective2, threshold=2)

    assert output1.shape == torch.Size([1, 3, 224, 224])
    assert output2.shape == torch.Size([1, 3, 224, 224])


def test_optimization_simple_rgb_channel(resnet18_example):
    """Test for channel optimization with the image in the RGB / pixel basis, in the activation/euclidean hidden layer basis"""
    hooked_model = resnet18_example.hooked_model
    hooked_model.add_hooks(["conv1", "layer1.0.conv2"])
    objective1 = ChannelObjective("layer1.0.conv2", coord=5)
    feature_image = InputFeature(FeatureVizType.RGB_IMAGE, dims=(1, 3, 224, 224))
    # take two gradient ascent steps
    output1 = optimization_fv(feature_image, hooked_model, objective1, threshold=2)

    objective2 = ChannelObjective("conv1", coord=5)
    feature_image = InputFeature(FeatureVizType.RGB_IMAGE, dims=(1, 3, 224, 224))
    # take two gradient ascent steps
    output2 = optimization_fv(feature_image, hooked_model, objective2, threshold=2)

    assert output1.shape == torch.Size([1, 3, 224, 224])
    assert output2.shape == torch.Size([1, 3, 224, 224])


def test_optimization_svd_mean_fft(resnet18_example):
    """Test for channel optimization with the image in the Fourier basis, in the SVD hidden layer basis"""
    hooked_model = resnet18_example.hooked_model
    svd_projection = resnet18_example.svd_projection
    hooked_model.add_hooks(["conv1", "layer1.0.conv2"])

    svd_simple_objective1 = SVDMean(
        layer="layer1.0.conv2", coord=45, transform_activations=svd_projection
    )

    feature_image = InputFeature(FeatureVizType.FFT_IMAGE, dims=(1, 3, 224, 224))
    # take two gradient ascent steps
    output1 = optimization_fv(
        feature_image, hooked_model, svd_simple_objective1, threshold=2
    )

    svd_simple_objective2 = SVDMean(
        layer="conv1", coord=5, transform_activations=svd_projection
    )
    feature_image = InputFeature(FeatureVizType.FFT_IMAGE, dims=(1, 3, 224, 224))
    output2 = optimization_fv(
        feature_image, hooked_model, svd_simple_objective2, threshold=2
    )

    assert output1.shape == torch.Size([1, 3, 224, 224])
    assert output2.shape == torch.Size([1, 3, 224, 224])

def test_optimization_svd_mean_fft(resnet18_example):
    """Test for channel optimization with the image in the Fourier basis, in the SVD hidden layer basis"""
    hooked_model = resnet18_example.hooked_model
    svd_projection = resnet18_example.svd_projection
    hooked_model.add_hooks(["conv1", "layer1.0.conv2"])

    svd_simple_objective1 = SVDMean(
        layer="layer1.0.conv2", coord=45, transform_activations=svd_projection
    )

    feature_image = InputFeature(FeatureVizType.FFT_IMAGE, dims=(1, 3, 224, 224))
    # take two gradient ascent steps
    output1 = optimization_fv(
        feature_image, hooked_model, svd_simple_objective1, threshold=2
    )

    svd_simple_objective2 = SVDMean(
        layer="conv1", coord=5, transform_activations=svd_projection
    )
    feature_image = InputFeature(FeatureVizType.FFT_IMAGE, dims=(1, 3, 224, 224))
    output2 = optimization_fv(
        feature_image, hooked_model, svd_simple_objective2, threshold=2
    )

    assert output1.shape == torch.Size([1, 3, 224, 224])
    assert output2.shape == torch.Size([1, 3, 224, 224])


def test_optimization_svd_mean_RGb(resnet18_example):
    """Test for channel optimization with the image in the RGB/pixel basis, in the SVD hidden layer basis"""
    hooked_model = resnet18_example.hooked_model
    svd_projection = resnet18_example.svd_projection
    hooked_model.add_hooks(["conv1", "layer1.0.conv2"])

    svd_simple_objective1 = SVDMean(
        layer="layer1.0.conv2", coord=45, transform_activations=svd_projection
    )

    feature_image = InputFeature(FeatureVizType.RGB_IMAGE, dims=(1, 3, 224, 224))
    # take two gradient ascent steps
    output1 = optimization_fv(
        feature_image, hooked_model, svd_simple_objective1, threshold=2
    )

    svd_simple_objective2 = SVDMean(
        layer="conv1", coord=5, transform_activations=svd_projection
    )
    feature_image = InputFeature(FeatureVizType.RGB_IMAGE, dims=(1, 3, 224, 224))
    output2 = optimization_fv(
        feature_image, hooked_model, svd_simple_objective2, threshold=2
    )

    assert output1.shape == torch.Size([1, 3, 224, 224])
    assert output2.shape == torch.Size([1, 3, 224, 224])
