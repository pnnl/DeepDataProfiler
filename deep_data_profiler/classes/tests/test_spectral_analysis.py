import numpy as np
from deep_data_profiler.algorithms import SpectralAnalysis
import powerlaw
import pytest  # noqa


def test_mini_metric(spectral_mini_example):
    spectral_analysis = spectral_mini_example
    metric = spectral_analysis.universal_metric
    analytic_metric = 2.840775052245874 * np.log(10)
    assert np.isclose(
        metric, analytic_metric, rtol=1e-04, atol=1e-4
    ), f"failure from analytic metric: {metric - analytic_metric}"


def test_mini_perlayer(spectral_mini_example):
    spectral_analysis = spectral_mini_example

    eigenvalue_dict = spectral_analysis.eigdict
    conv_evals = eigenvalue_dict[1][0]
    linear_evals = eigenvalue_dict[2][0]
    assert np.allclose(conv_evals, linear_evals)

    alpha_dict = spectral_analysis.alpha_dict
    conv_alpha, _ = alpha_dict[1]
    linear_alpha, _ = alpha_dict[2]
    assert np.allclose(conv_evals, linear_evals)

    assert len(spectral_analysis.layer_rmt) == 2


def test_resnet_metric(spectral_resnet_example):
    # checks if metric calculated provided alpha dict is
    # the same as calculated fresh
    spectral_analysis = spectral_resnet_example
    resnet_total_analysis = SpectralAnalysis(spectral_resnet_example.model)
    metric1 = spectral_analysis.universal_metric
    metric2 = resnet_total_analysis.universal_metric()

    assert metric1 == metric2
    assert round(metric1, 4) == -13.6798


def test_resnet_layers(spectral_resnet_example):
    spectral_analysis = spectral_resnet_example
    assert len(spectral_analysis.layer_rmt) == 21

    eigdict = spectral_analysis.eigdict
    alpha_dict = spectral_analysis.alpha_dict
    assert len(eigdict) == len(alpha_dict)

    calc_alpha = powerlaw.Fit(spectral_analysis.eigdict[1][0]).alpha
    pre_calc_alpha = spectral_analysis.alpha_dict[1][0]
    assert calc_alpha == pre_calc_alpha
