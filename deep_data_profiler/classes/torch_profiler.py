from abc import ABC, abstractmethod
from collections import OrderedDict
from deep_data_profiler.classes.profile import Profile
from deep_data_profiler.utils import model_graph, TorchHook
import inspect
import networkx as nx
import torch
from torch.nn.modules import activation, dropout, batchnorm
from typing import Dict, List, Optional, Tuple, Union, Any

class TorchProfiler(ABC):

    """
    A TorchProfiler wraps a PyTorch model into a TorchHook model which can register
    activations as it evaluates data. Subclasses of TorchProfiler profile model inputs
    by tracing influential neurons through the layers of the network during classification.
    Each subclass implementation defines neurons, influence, and contribution differently.

    Attributes
    ----------
    activation_classes : list
        List of activation classes in PyTorch
    dropout_classes : list
        List of dropout classes in PyTorch
    implemented_classes : list
        List of classes in PyTorch which have contrib functions implemented
    contrib_functions : list
        List of contrib functions ordered by their corresponding classes in
        implemented_classes (does not include "contrib_resnetadd")
    model : torch.nn.Module
        Model to be profiled
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:

        super().__init__()
        self.activation_classes = [
            m[1]
            for m in inspect.getmembers(activation, inspect.isclass)
            if m[1].__module__ == "torch.nn.modules.activation"
        ]
        self.dropout_classes = [
            m[1]
            for m in inspect.getmembers(dropout, inspect.isclass)
            if m[1].__module__ == "torch.nn.modules.dropout"
        ]
        self.batchnorm_classes = [
            m[1]
            for m in inspect.getmembers(batchnorm, inspect.isclass)
            if m[1].__module__ == "torch.nn.modules.batchnorm"
        ]

        # at present the index of the implemented class must match the index of the
        # contrib_function used for that class. When we have enough many to one
        # relationships between the two lists we will change this to a dictionary.
        self.implemented_classes = [
            torch.nn.Linear,
            torch.nn.MaxPool2d,
            torch.nn.AdaptiveAvgPool2d,
            torch.nn.Conv2d,
        ]
        self.contrib_functions = [
            "contrib_linear",
            "contrib_max2d",
            "contrib_adaptive_avg_pool2d",
            "contrib_conv2d",
        ]

        self.model = TorchHook(model, device)
        self.device = device
        self.hooks = self.model.available_modules()
        supernodes, SG, pos = self.super_nodes_graph()
        self.supernodes = supernodes
        self.SG = SG
        self.pos = pos
        self.sght = self.sgheadtail()
        self.pred_dict = {
            nd[0]: sorted([preds[0] for preds in self.SG.predecessors(nd)])
            for nd in sorted(self.SG)
        }
        self.layerdict = self.create_layers()

    def super_nodes_graph(self) -> Tuple[OrderedDict, Any, Dict]:
        """
        Returns
        -------
        supernodes : OrderedDict
        SG : nx.DiGraph
        sgpos : dict
        """
        model = self.model.model  # grab back the original model
        G, ndorder, pos = model_graph(model)

        # group nodes into layers
        pmas = (
            self.model.available_modules()
        )  # links the graph nodes to PyTorch modules
        nodes = G.nodes

        # A layer starts with an implemented class. Some of these correspond to pytorch
        # modules, others were custom made for the application, like 'add'.
        def chk_type(n):
            try:
                if n.endswith("add") or type(pmas[n]) in self.implemented_classes:
                    return True
            except:
                return False

        # The implemented nodes are the first function to apply to input into the DDPlayer,
        # the successors which do not change the shape are added to the DDPlayer.
        implemented_nodes = [n for n in nodes if chk_type(n)]
        supernodes = OrderedDict({"x_in": ["x_in"]})

        for n in implemented_nodes:
            temp = [n]
            while True:
                scs = list(G.successors(temp[-1]))
                if (len(scs) == 0 or len(scs) > 1) or (
                    len(scs) == 1 and (chk_type(scs[0]))
                ):
                    break
                else:
                    temp.append(scs[0])
            supernodes[n] = temp

        SGnodes = list(supernodes.keys())
        SG = nx.DiGraph()  # the Digraph of the DDP network
        for nd in SGnodes:
            tail = supernodes[nd][-1]
            try:
                scs = list(G.successors(tail))
                snds = [(nd, snd) for snd in scs if snd in SGnodes]
                SG.add_edges_from(snds)
            except Exception as ex:
                print(ex)
                continue

        # Rename nodes according to sorted node order for SG sequences layers in the graph

        def sortorder(x):
            return ndorder.index(x)

        ordG = {
            v: (k, v) for k, v in enumerate(sorted(list(SG.nodes()), key=sortorder))
        }
        nx.relabel_nodes(SG, ordG, copy=False)

        sgpos = dict()
        for nd in SG.nodes():
            sgpos[nd] = pos[nd[1]]

        return supernodes, SG, sgpos

    def sgheadtail(self) -> Dict[int, Tuple[str, str]]:
        """
        Returns
        -------
        sght : dict
            Dictionary of the names of the head and tail modules at each layer
        """
        namelist = list()
        sght = dict()
        pmas = self.model.available_modules()
        for nd in self.SG.nodes:
            if nd[0] == 0:
                sght[0] = ("x_in", "x_in")
                continue
            arr = self.supernodes[nd[1]]
            n = -1 * len(arr) - 1
            for k in arr:
                if k in pmas:
                    head = k
                    namelist.append(k)
                    break
            else:
                head = None
            for k in arr[-1:n:-1]:
                if k in pmas:
                    tail = k
                    namelist.append(k)
                    break
            else:
                tail = None
            sght[nd[0]] = (head, tail)
            self.model.add_hooks(namelist)
        return sght

    def create_layers(self) -> Dict[int, Tuple[List[str], str]]:
        """
        Create a dictionary of layers to profile.

        Returns
        -------
        layerdict : OrderedDict
            A dictionary of layers keyed by their position according to the order Pytorch
            uses to list the layers of the model.

        Note
        ----
        Because a weighted module is often followed by an activation and/or normalization
        modules those modules are combined into a single layer to be profiled. As a
        consequence the number of layers generated will typically be less than the number
        of available modules in the model.

        """
        # identify layers to profile.
        layerdict = OrderedDict()  # ordered dictionary to feed to profiler

        layerdict[0] = [
            ["x_in"],
            "contrib_identity",
        ]  # this is the tensor input for the image

        SGnodes = sorted(self.SG.nodes, key=lambda nd: nd[0])
        for ndx in range(1, len(SGnodes)):
            nd = SGnodes[ndx]
            if nd[1].endswith("resnetadd"):
                layerdict[nd[0]] = [self.supernodes[nd[1]], "contrib_resnetadd"]
            else:
                this_type = type(self.model.available_modules()[nd[1]])
                layerdict[nd[0]] = [
                    self.supernodes[nd[1]],
                    self.contrib_functions[self.implemented_classes.index(this_type)],
                ]

        return layerdict

    def build_dicts(
        self,
        x: torch.Tensor,
        infl_threshold: float,
        contrib_threshold: float,
        layers_to_profile: Optional[Union[List[int], Tuple[int]]] = None,
        **kwargs,
    ) -> Tuple[dict, dict, dict, dict]:
        """
        Generates neuron/synapse counts/weights for a given input.

        Parameters
        ----------
        x : torch.Tensor
            input to model being profiled
        layers_to_profile : list or tuple
            list of specific layers to profile or tuple with first,last layers
            (exclusive of last) to profile and all layers in between
        infl_threshold : float
            Parameter for influence
        contrib_threshold : float
            Parameter for contribution

        Returns
        -------
        neuron_counts
        neuron_weights
        synapse_counts
        synapse_weights
        activation_shapes

        """

        # Create empty dictionaries for storing neurons and synapses
        neuron_counts = dict()
        neuron_weights = dict()
        synapse_counts = dict()
        synapse_weights = dict()

        ######## Step 1: Identify the Influential Neurons (Nodes) in the Activation Graph ########
        y, activations = self.model.forward(x)
        activations["x_in"] = x

        activation_shapes = {}
        for ldx, modules in self.layerdict.items():
            if "resnetadd" in modules[1]:
                activation_shapes[ldx] = activations[modules[0][1]].shape
            else:
                activation_shapes[ldx] = activations[modules[0][0]].shape

        # Create an x-dependent function to identify the the highest valued neurons at each layer
        influential_neurons = self.influence_generator(activations, **kwargs)

        # Layers to profile:
        n = len(self.layerdict)
        if layers_to_profile == None:
            ltp = range(1, n)
        elif isinstance(layers_to_profile, list):
            ltp = [lyr for lyr in layers_to_profile if lyr >= 1 or lyr <= n - 1]
        else:  # a tuple is expected
            start = max(1, layers_to_profile[0])
            end = min(n, layers_to_profile[1])
            ltp = range(start, end)

        # Fill dictionaries
        for k in ltp:
            neuron_counts[k], neuron_weights[k] = influential_neurons(k, infl_threshold)

        ######## Step 2: Identify Contributing Neurons (Nodes) and Synaptic connections (Edges) ########

        with torch.no_grad():
            y, activations = self.model.forward(x)
            activations["x_in"] = x

            # Assign activations to x_in and y_out for each layer
            # Note that if f = first available module in a layer
            # y_out = f(x_in)
            y_out = dict()
            x_in = dict()
            for k in ltp:
                y_out[k] = {
                    k: activations[self.sght[k][0]]
                }  # this is the output from the first available module in the layer
                nd = (k, self.layerdict[k][0][0])
                pred_layers = list(self.SG.predecessors(nd))
                x_in[k] = {
                    pd[0]: activations[self.sght[pd[0]][1]] for pd in pred_layers
                }  # this is the output from the last available module in the layer

            results = [
                self.single_profile(
                    x_in[ldx],
                    y_out[ldx],
                    neuron_counts[ldx],
                    ldx,
                    contrib_threshold,
                )
                for ldx in ltp
            ]

            for i, f in enumerate(results):
                ldx = ltp[i]
                x_ldx = list(x_in[ldx].keys())
                ncs, scs, sws = f
                synapse_counts[ldx] = scs
                synapse_weights[ldx] = sws
                if len(x_ldx) == 1:
                    ncs = (ncs,)
                for nc, xdx in zip(ncs, x_ldx):
                    if xdx in neuron_counts:
                        neuron_counts[xdx] += nc
                        neuron_counts[xdx] = neuron_counts[xdx].tocoo()
                    else:
                        neuron_counts[xdx] = nc

        return (
            neuron_counts,
            neuron_weights,
            synapse_counts,
            synapse_weights,
            activation_shapes,
        )

    @abstractmethod
    def influence_generator(self):
        pass

    @abstractmethod
    def single_profile(self):
        pass

    @abstractmethod
    def create_profile(self):
        pass
