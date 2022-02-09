import numpy as np
import pytest
import deep_data_profiler as ddp


def test_profile_constructor(profile_toy_example):
    p = ddp.Profile()
    assert len(p.synapse_weights) == 0
    pe = profile_toy_example
    p = ddp.Profile(
        neuron_counts=pe.neuron_counts,
        synapse_counts=pe.synapse_counts,
        synapse_weights=pe.synapse_weights,
        num_inputs=pe.num_inputs,
    )
    assert np.allclose(p.neuron_counts[1].data, [2, 1, 1, 1])
    assert np.allclose(p.synapse_weights[1].data, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_profile_attributes(profile_toy_example):
    pe = profile_toy_example
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


def test_add(profile_toy_example):
    pe = profile_toy_example
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


def test_element_dict_view(element_profile):
    p = element_profile
    d = p.profile.dict_view()
    assert d.synapse_counts[3] == {
        ((2, (10, 192, 66)), (3, (10, 96, 33))): 1,
        ((2, (36, 53, 124)), (3, (36, 26, 62))): 1,
        ((2, (45, 52, 124)), (3, (45, 26, 62))): 1,
    }
    assert d.neuron_counts[2] == {
        (2, (45, 52, 124)): 2,
        (2, (36, 53, 124)): 2,
        (2, (10, 192, 66)): 2,
    }


def test_channel_dict_view(channel_profile):
    p = channel_profile
    d = p.profile.dict_view()
    assert d.synapse_counts[3] == {
        ((2, (10,)), (3, (10,))): 1,
        ((2, (36,)), (3, (36,))): 1,
        ((2, (45,)), (3, (45,))): 1,
    }
    assert d.neuron_counts[2] == {(2, (10,)): 2, (2, (36,)): 2, (2, (45,)): 2}


def test_spatial_dict_view(spatial_profile):
    p = spatial_profile
    d = p.profile.dict_view()
    assert d.synapse_counts[19] == {
        ((18, (0, 0)), (19, (0, 0))): 1,
        ((18, (0, 5)), (19, (0, 5))): 1,
        ((18, (0, 6)), (19, (0, 6))): 1,
    }
    assert d.neuron_counts[18] == {(18, (0, 0)): 2, (18, (0, 5)): 2, (18, (0, 6)): 2}
