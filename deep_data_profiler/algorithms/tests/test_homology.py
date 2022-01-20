import pytest
import deep_data_profiler as ddp
import numpy as np


def test_tuples_from_dict(synapse_example):
    ex = synapse_example
    sw = ex.synapse_weights

    tuples = ddp.tuples_from_dict(sw)
    expected = [
        ((0, (0,)), (1, (0,)), 1),
        ((0, (0,)), (1, (1,)), 1),
        ((0, (0,)), (1, (2,)), 1),
        ((1, (0,)), (2, (0,)), 0.5),
        ((1, (2,)), (2, (0,)), 0.5),
    ]

    tupleset = set(tuples)
    expectedset = set(expected)
    assert len(tuples) == len(tupleset)
    assert tupleset == expectedset


def test_mat_from_graph(digraph_example):
    ex = digraph_example
    graph = digraph_example.graph

    distmat = ddp.mat_from_graph(graph)
    expected = np.asarray(
        [
            [0, 1, 1, 1, 1.5],
            [np.inf, 0, np.inf, np.inf, 0.5],
            [np.inf, np.inf, 0, np.inf, np.inf],
            [np.inf, np.inf, np.inf, 0, 0.5],
            [np.inf, np.inf, np.inf, np.inf, 0],
        ]
    )
    assert np.allclose(distmat, expected)
