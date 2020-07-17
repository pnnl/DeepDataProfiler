import numpy as np
import pytest
import deep_data_profiler as ddp
from collections import Counter
import warnings


def test_profile_constructor(profile_example):
    p = ddp.Profile()
    assert len(p.synapse_weights) == 0
    pe = profile_example
    p = ddp.Profile(neuron_counts=pe.neuron_counts,
                    synapse_counts=pe.synapse_counts,
                    synapse_weights=pe.synapse_weights,
                    num_inputs=pe.num_inputs)
    assert p.neuron_counts[0][2] == 1


def test_profile_attributes(profile_example):
    pe = profile_example
    p = ddp.Profile(neuron_counts=pe.neuron_counts,
                    synapse_counts=pe.synapse_counts,
                    synapse_weights=pe.synapse_weights,
                    num_inputs=3)
    assert p.neuron_counts == pe.neuron_counts
    assert p.synapse_counts == pe.synapse_counts
    assert p.synapse_weights == pe.synapse_weights
    assert p.num_inputs == 3
    assert p.total == 5
    assert p.size == 4
    q = ddp.Profile(neuron_counts=pe.neuron_counts,
                    synapse_counts=pe.synapse_counts,
                    synapse_weights=pe.synapse_weights,
                    num_inputs=3)
    assert p == q


def test_add(profile_example):
    pe = profile_example
    p = ddp.Profile(neuron_counts=pe.neuron_counts,
                    synapse_counts=pe.synapse_counts,
                    synapse_weights=pe.synapse_weights,
                    num_inputs=3)
    q = ddp.Profile(neuron_counts=pe.neuron_counts,
                    synapse_counts=pe.synapse_counts,
                    synapse_weights=pe.synapse_weights,
                    num_inputs=4)
    s = p + q
    assert s.size == 4
    assert s.total == 10
    assert s.num_inputs == 7
    p += p
    assert p.total == 10
    assert p.num_inputs == 6


def test_jaccard(profile_example2):
    pe = profile_example2
    p1 = pe.p1
    p2 = pe.p2
    assert ddp.jaccard(p1, p2) - 0.5714285 < 10e-5


def test_avg_jaccard(profile_example2):
    pe = profile_example2
    p1 = pe.p1
    p2 = pe.p2
    assert ddp.avg_jaccard(p1, p2) == 0.5


def test_instance_jaccard(profile_example2):
    pe = profile_example2
    p1 = pe.p1
    p2 = pe.p2
    p3 = pe.p3
    assert ddp.instance_jaccard(p1, p1 + p2) == 1
    assert ddp.instance_jaccard(p3, p1 + p2) == 0.5
