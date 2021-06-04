import networkx as nx
import numpy as np
from typing import Callable, Dict, List, Tuple, Union
from ripser import ripser

Neuron = Tuple[int, Union[Tuple[int], Tuple[int, int]]]
SynapseDict = Dict[Tuple[Neuron, Neuron], float]


def tuples_from_dict(
    d: Dict[int, SynapseDict], layers: List[int] = None
) -> List[Tuple[Neuron, Neuron, float]]:
    """
    Returns a list of tuples representing synapses up to a specified layer

    Parameters
    ----------
    d : dict
        A dictionary of synapse weights, keyed by layer and synapse
    layers : list, optional, default=None
        If None (default), dictionary entries for all layers will be included in the list of tuples,
        otherwise, entries from the layers given in the list will be included

    Returns
    -------
    tuples : list
        A list of tuples of the form ((layer1, neuron1), (layer2, neuron2), weight)

    Note
    ----
    Neurons are renamed to be (layer, neuron) since different layers may have the same neuron indices.
    The parameter nlayers is an inclusive bound.
    """
    tuples = []
    if layers is None:
        layers = d.keys()

    for ldx in layers:
        temp = d[ldx]
        tuples += [
            ((t[0][0], t[0][1][0]), (t[1][0], t[1][1][0]), temp[t]) for t in temp
        ]

    return tuples


def graph_from_tuples(
    tuples: List[Tuple[Neuron, Neuron, float]],
    directed: bool = True,
    weight_func: Callable[[float], float] = (lambda x: x),
) -> nx.Graph:
    """
    Returns a weighted graph constructed from a set of tuples

    Parameters
    ----------
    tuples : iterable
        An iterable of tuples of the form (vertex1, vertex2, weight)
    weight_func : function
        A function that takes the weight from a tuple as its input
        and returns the weight of the edge between vertex1 and vertex2 in the graph,
        default= f(x)=x

    Returns
    -------
    G : networkx.Graph
        A weighted graph with edge weights given by the weight_func of the tuple weights
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for t in tuples:
        wt = max(0, weight_func(t[2]))
        G.add_edge(t[0], t[1], weight=wt)

    return G


def mat_from_graph(
    G: nx.Graph, kernel: Callable[[float], float] = (lambda x: x)
) -> np.ndarray:
    """
    Given a weighted graph create a matrix based on a kernel function of the shortest weighted path distance.

    Parameters
    ----------
    G : networkx.Graph
        A weighted graph
    kernel : function
        A function of the shortest weighted path distance, default= f(x)=x

    Returns
    -------
    matrix : numpy.ndarray

    Note
    ----
    This will only be a distance matrix if the kernel function keeps things as distances.
    """
    Vdict = {v: i for i, v in enumerate(G.nodes)}

    matrix = np.full((G.order(), G.order()), np.inf)
    np.fill_diagonal(matrix, 0)
    for pair in nx.shortest_path_length(G, weight="weight"):
        i = Vdict[pair[0]]
        for v2 in pair[1]:
            matrix[i, Vdict[v2]] = kernel(pair[1][v2])

    return matrix


def ripser_of_distmat(dist_matrix: np.ndarray, maxdim: int = 1) -> dict:
    """
    Given a distance matrix compute the persistent homology using ripser.

    Parameters
    ----------
    dist_matrix : numpy.ndarray
        A distance matrix
    maxdim : int, optional, default=1
        The maximum homology dimension to compute, default=1

    Returns
    -------
    R : dict
        A dictionary holding the results of the computation

    Note
    ----
    The given dist_mat should be a distance matrix, but this is not strictly enforced
    """
    R = ripser(dist_matrix, distance_matrix=True, maxdim=maxdim)

    return R


def ripser_of_graph(
    G: nx.Graph, kernel: Callable[[float], float] = (lambda x: x), maxdim: int = 1
) -> dict:
    """
    Given a weighted graph compute the persistent homology using ripser.

    Parameters
    ----------
    G : networkx.Graph
        A weighted graph
    kernel : function
        A function of the shortest weighted path distance, default= f(x)=x
    maxdim : int
        The maximum homology dimension to compute, default=1

    Returns
    -------
    R : dict
        A dictionary holding the results of the computation

    Note
    ----
    The given kernel function should produce a distance matrix, but this is not strictly enforced
    """
    dist_matrix = mat_from_graph(G, kernel=kernel)
    R = ripser(dist_matrix, distance_matrix=True, maxdim=maxdim)

    return R
