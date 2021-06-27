import pytest
import deep_data_profiler as ddp
import scipy.sparse as sp
import networkx as nx


class SynapseExample:
    def __init__(self):
        self.synapse_weights = {
            1: {
                ((0, (0,)), (1, (0,))): 1,
                ((0, (0,)), (1, (1,))): 1,
                ((0, (0,)), (1, (2,))): 1,
            },
            2: {((1, (0,)), (2, (0,))): 0.5, ((1, (2,)), (2, (0,))): 0.5},
        }


@pytest.fixture
def synapse_example():
    return SynapseExample()


class TuplesExample:
    def __init__(self):
        self.tuples = [
            ((0, 0), (1, 0), 1),
            ((0, 0), (1, 1), 1),
            ((0, 0), (1, 2), 1),
            ((1, 0), (2, 0), 0.5),
            ((1, 2), (2, 0), 0.5),
        ]


@pytest.fixture
def tuples_example():
    return TuplesExample()


class DiGraphExample:
    def __init__(self):
        edges = [
            ((0, 0), (1, 0), 1),
            ((0, 0), (1, 1), 1),
            ((0, 0), (1, 2), 1),
            ((1, 0), (2, 0), 0.5),
            ((1, 2), (2, 0), 0.5),
        ]

        G = nx.DiGraph()
        for e in edges:
            G.add_edge(e[0], e[1], weight=e[2])

        self.graph = G


@pytest.fixture
def digraph_example():
    return DiGraphExample()


class ProfileExample:
    def __init__(self):
        nc1 = {
            0: sp.coo_matrix([[4], [0]]),
            1: sp.coo_matrix([[1], [1], [1], [0], [1], [0]]),
        }
        nc2 = {
            0: sp.coo_matrix([[5], [0]]),
            1: sp.coo_matrix([[0], [1], [1], [1], [1], [1]]),
        }
        nc3 = {
            0: sp.coo_matrix([[2], [2]]),
            1: sp.coo_matrix([[0], [1], [0], [1], [0], [0]]),
        }

        sw1 = {
            1: sp.coo_matrix([[0.1, 0], [0.2, 0], [0.3, 0], [0, 0], [0.4, 0], [0, 0]])
        }
        sw2 = {
            1: sp.coo_matrix([[0, 0], [0.1, 0], [0.3, 0], [0.5, 0], [0.7, 0], [0.9, 0]])
        }
        sw3 = {
            1: sp.coo_matrix([[0, 0], [0.5, 0.1], [0, 0], [0.7, 0.3], [0, 0], [0, 0]])
        }

        sc1 = {1: sw1[1].astype(bool).astype(int)}
        sc2 = {1: sw2[1].astype(bool).astype(int)}
        sc3 = {1: sw3[1].astype(bool).astype(int)}

        self.p1 = ddp.Profile(
            neuron_counts=nc1, synapse_counts=sc1, synapse_weights=sw1, num_inputs=1
        )
        self.p2 = ddp.Profile(
            neuron_counts=nc2, synapse_counts=sc2, synapse_weights=sw2, num_inputs=1
        )
        self.p3 = ddp.Profile(
            neuron_counts=nc3, synapse_counts=sc3, synapse_weights=sw3, num_inputs=1
        )


@pytest.fixture
def profile_example():
    return ProfileExample()
