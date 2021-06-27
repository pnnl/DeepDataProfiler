import numpy as np
import pytest
import deep_data_profiler as ddp


def test_profile_constructor(profile_example):
    p = ddp.Profile()
    assert len(p.synapse_weights) == 0
    pe = profile_example
    p = ddp.Profile(
        neuron_counts=pe.neuron_counts,
        synapse_counts=pe.synapse_counts,
        synapse_weights=pe.synapse_weights,
        num_inputs=pe.num_inputs,
    )
    assert np.allclose(p.neuron_counts[1].data, [2, 1, 1, 1])
    assert np.allclose(p.synapse_weights[1].data, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_profile_attributes(profile_example):
    pe = profile_example
    p = ddp.Profile(
        neuron_counts=pe.neuron_counts,
        synapse_counts=pe.synapse_counts,
        synapse_weights=pe.synapse_weights,
        num_inputs=3,
    )
    assert p.neuron_counts == pe.neuron_counts
    assert p.synapse_counts == pe.synapse_counts
    assert p.synapse_weights == pe.synapse_weights
    assert p.num_inputs == 3
    assert p.total == 5
    assert p.size == 4
    q = ddp.Profile(
        neuron_counts=pe.neuron_counts,
        synapse_counts=pe.synapse_counts,
        synapse_weights=pe.synapse_weights,
        num_inputs=3,
    )
    assert p == q


def test_add(profile_example):
    pe = profile_example
    p = ddp.Profile(
        neuron_counts=pe.neuron_counts,
        synapse_counts=pe.synapse_counts,
        synapse_weights=pe.synapse_weights,
        num_inputs=3,
    )
    q = ddp.Profile(
        neuron_counts=pe.neuron_counts,
        synapse_counts=pe.synapse_counts,
        synapse_weights=pe.synapse_weights,
        num_inputs=4,
    )
    s = p + q
    assert s.size == 4
    assert s.total == 10
    assert s.num_inputs == 7
    p += p
    assert p.total == 10
    assert p.num_inputs == 6
