import sys
import os
import json
import pickle
from collections import defaultdict, Counter, OrderedDict
import warnings
import concurrent.futures

import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
import torchvision
from torchvision import transforms
import torchvision.models as tvmodels
from torchvision.transforms.functional import affine, hflip, to_pil_image, to_tensor

import deep_data_profiler as ddp

warnings.filterwarnings("ignore")

__all__ = [
    "model_graph",
    "super_node_graph",
    "available_modules",
    "bottle_graph",
    "seq_graph",
]

# To ensure reproducibility we set the pytorch seed. This only works if the versioning of libraries is held
# constant
torch.manual_seed(0)
# We will also specify the device. The profiling code will run on cpu or cuda.
device = torch.device("cpu")


def model_graph(model, full_seq=True):
    def fs(x):
        return False if type(x) == tvmodels.resnet.ResNet else full_seq

    _, nodes, edges, _, pos = seq_graph(model, full_seq=fs(model))
    return nx.DiGraph(edges), nodes, pos


def available_modules(model):
    """
    Create module reference dictionary linking node names in graph to
    modules in model where possible

    Parameters
    ----------
    model : torch.model

    Returns
    -------
    : dict
        dictionary of available modules
    """

    g, ndlist, _ = model_graph(model)

    avail_modules = OrderedDict()
    for k in range(len(ndlist)):

        ndl = ndlist[k].split(".")
        try:
            temp = model._modules[ndl[0]]
            for idx in range(1, len(ndl)):
                try:
                    temp = temp._modules[ndl[idx]]
                except:
                    try:
                        temp = temp[ndlist[k]]
                    except:
                        temp = ndlist[k]
            if len(list(temp.named_children())) != 0:
                # temp = None  # ndlist[k]
                continue
        except:
            # temp = None  # ndlist[k]
            continue

        avail_modules[ndlist[k]] = temp

    return avail_modules


def module_dict(module, return_class=False):
    """
    Ordered dictionary of all submodules of module

    Parameters
    ----------
    module : pytorch.model or pytorch.module

    Returns
    -------
    d : OrderedDict

    """
    d = OrderedDict()
    for n, m in module.named_children():
        if len(list(m.named_children())) == 0:
            if return_class:
                d[n] = [type(m), m]
            else:
                d[n] = m
        else:
            if return_class:
                d[n] = [type(m), module_dict(m)]
            else:
                d[n] = module_dict(m, return_class=return_class)
    return d


class Gindex:

    """
    Convenience class for position counter for nodes in
    networkx graph. Used for visualizing model's architecture

    Attributes
    ----------
    fixed : bool
    start : int
    value : int

    """

    def __init__(self, start=0, fixed=True):
        self.fixed = fixed
        self.start = start
        self.value = self.start

    def __call__(self):
        return self.value

    def inc(self, i=1):
        if not self.fixed:
            temp = self.value
            self.value += i
            return temp
        else:
            return self.value


def resnet_graph(
    module,
    nd="x_in",
    nodes=None,
    edges=None,
    prefix=None,
    level=1,
    pos=None,
    full_seq=False,
):
    """
    Create networkx directed graph for pytorch.resnet BasicBlock or Bottleneck module with node names given by named children of the module

    Parameters
    ----------
    module : torch.model
        Description
    nd : str, default : 'x_in'
        layer name of basic block graph to build
    nodes : list,  default : None, optional
        existing list of nodes to include in new graph
    edges : list, default : None, optional
        existing list of edges to include in new graph
    prefix : str, default : None, optional
        layer name
    mod_dict : dict, default : None, optional
        dictionary to rename layers if needed
    level : int, default : 1
        corresponds to position level for drawing the graph
    pos : dict, default : None, optional
        coordinate positions for nodes passed to networkx draw method
    full_seq : bool, default : False
        if module is fully sequential then every layer will be incremented in its row position

    Returns
    -------
    prefix, nodes, edges, level + 4, pos

    prefix : str
        Name of last point in subgraph
    nodes : list
        list of string names for nodes
    edges : list
        list of ordered pairs of node names
    : int
        last row level of graph
    pos : dict

    Note
    ----
    The returned nodes are given in the order they should be evaluated in the profile.
    If modifying this code be mindful of the order of the nodes list
    """

    first_node = nd
    nodes = nodes or [nd]
    edges = edges or list()

    pos = pos or dict()
    if prefix is None:
        if type(m) == tvmodels.resnet.Bottleneck:
            prefix = "bottleneck"
        elif type(m) == tvmodels.resnet.BasicBlock:
            prefix = "basicblock"
    prefix_node = f"{prefix}"
    last_node = f"{prefix}.relu"
    add_node = f"{prefix}.resnetadd"
    nodes += [prefix_node]
    edges += [(add_node, last_node), (last_node, prefix_node)]
    #     nodes += [last_node, add_node]
    #     edges += [(add_node, last_node)]

    smods = OrderedDict(
        (k, v) for k, v in module.named_children() if k != "downsample" and k != "relu"
    )
    sdict = {str(k): v for k, v in enumerate(smods)}
    seq_resnet = Sequential(
        *[
            module._modules[k]
            for k in module._modules.keys()
            if k != "downsample" and k != "relu"
        ]
    )
    nd, nodes, edges, _, pos = seq_graph(
        seq_resnet,
        first_node,
        nodes,
        edges,
        prefix,
        sdict,
        level + 1,
        pos,
        full_seq=full_seq,
    )
    edges.append((nd, add_node))

    if module.downsample is not None:
        nd, nodes, edges, _, pos = seq_graph(
            module.downsample,
            first_node,
            nodes,
            edges,
            prefix + ".downsample",
            level=level,
            pos=pos,
            full_seq=full_seq,
        )
        edges.append((nd, add_node))
        dnodes = [nd for nd in nodes if nd.startswith(f"{prefix}.downsample")]
        for ddx, dnd in enumerate(dnodes):
            pos[dnd] = [0.5 * (ddx + 1), level - 0.3 + 0.6 * ddx]
    else:
        edges.append((first_node, add_node))

    pos[add_node] = (-0.5, level + 2)
    #     pos[last_node] = (-0.5, level + 3)
    pos[last_node] = (-0.5, level + 2.5)
    pos[prefix_node] = (-0.5, level + 3)
    nodes += [last_node, add_node]
    return prefix, nodes, edges, level + 4, pos


#     return last_node, nodes, edges, level + 4, pos


def seq_graph(
    module,
    nd="x_in",
    nodes=None,
    edges=None,
    prefix=None,
    mod_dict=None,
    level=0,
    pos=None,
    full_seq=True,
):
    """
    Create networkx directed graph for pytorch Sequential module with node names given by named children of the module

    Parameters
    ----------
    module : torch.model
        Description
    nd : str, default : 'x_in'
        layer name of sequential graph to build
    nodes : list,  default : None, optional
        existing list of nodes to include in new graph
    edges : list, default : None, optional
        existing list of edges to include in new graph
    prefix : str, default : None, optional
        layer name
    mod_dict : dict, default : None, optional
        dictionary to rename layers if needed
    level : int, default : 0
        corresponds to position level for drawing the graph
    pos : dict, default : None, optional
        coordinate positions for nodes passed to networkx draw method
    full_seq : bool, default : True
        if module is fully sequential then every layer will be incremented in its row position

    Returns
    -------
    nd : str
    nodes : list
        list of string names for nodes
    edges : list
        list of ordered pairs of node names
    : int
        last row level of graph
    pos : dict
        coordinate positions keyed by node names for passing to draw method in networkx.

    Note
    ----
    The returned nodes are given in the order they should be evaluated in the profile.
    If modifying this code be mindful of the order of the nodes list
    """
    children = OrderedDict(module.named_children())
    ckeys = list(children.keys())
    offset = len(ckeys) // 2

    first_level = level
    if full_seq:
        rindex = Gindex(level, fixed=False)
        cindex = Gindex(0, fixed=True)
    elif first_level == 0 and not full_seq:
        rindex = Gindex(level, fixed=False)
        cindex = Gindex(-0.5, fixed=True)
    else:
        rindex = Gindex(level, fixed=True)
        cindex = Gindex(-1 * offset, fixed=False)
    pos = pos or {nd: (rindex.inc(), cindex.inc())}  # position of first node
    if first_level == 0 and not full_seq:
        pos["x_in"] = (-0.5, level)

    nodes = nodes or [nd]
    edges = edges or list()

    if mod_dict is None:

        def mod_names(c):
            return c

    else:

        def mod_names(c):
            return mod_dict[c]

    for cdx in range(len(ckeys)):
        m = children[ckeys[cdx]]
        if prefix is not None:
            temp = f"{prefix}.{mod_names(ckeys[cdx])}"
        else:
            temp = mod_names(ckeys[cdx])
        if len(list(m.named_children())) == 0:
            nodes.append(temp)
            pos[temp] = (cindex.inc(), rindex.inc())
            edges.append((nd, temp))
            nd = temp
        elif type(m) == torch.nn.modules.container.Sequential:
            nd, nodes, edges, level, pos = seq_graph(
                m, nd, nodes, edges, temp, level=rindex(), pos=pos, full_seq=full_seq
            )

            rindex = Gindex(level, first_level != 0)
        elif (
            type(m) == tvmodels.resnet.Bottleneck
            or type(m) == tvmodels.resnet.BasicBlock
        ):
            nd, nodes, edges, level, pos = resnet_graph(
                m, nd, nodes, edges, temp, level=rindex(), pos=pos
            )
            rindex = Gindex(level, first_level != 0)
        else:
            print(f"not implemented for {temp}")
            return
    return nd, nodes, edges, rindex.inc(), pos
