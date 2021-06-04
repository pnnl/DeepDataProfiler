from collections import OrderedDict
import torch
import inspect
from torch.nn.modules import activation, dropout, batchnorm
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import networkx as nx
from typing import Callable, Dict, List, Tuple, Union
import warnings
from deep_data_profiler.classes.profile import Profile
from deep_data_profiler.utils import (
    TorchHook,
    submatrix_generator,
    get_index,
    model_graph,
    matrix_convert,
)


class TorchProfiler:

    """
    Torch Profiler wraps a PyTorch model into a TorchHook model which can register activations
    as it evaluates data. Using the activations, inputs to the model may be profiled.

    Attributes
    ----------
    activation_classes : list
        List of activation classes in PyTorch
    dropout_classes : list
        List of dropout classes in PyTorch
    implemented_classes : list
        List of classes in PyTorch which have contrib functions implemented
    contrib_functions : list
        List of contrib functions ordered by their corresponding classes in implemented_classes
    model : torch.nn.Sequential
        Model to be profiled
    """

    def __init__(
        self, model: torch.nn.Sequential, device: torch.device = torch.device("cpu")
    ) -> None:

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

        # at present the index of the implemented class must match the index of the contrib_function used for
        # that class. When we have enough many to one relationships between the two lists we will change this
        # to a dictionary.
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

        self.model = TorchHook(model)
        self.hooks = self.model.available_modules()
        supernodes, SG, pos = self.super_nodes_graph()
        self.supernodes = supernodes
        self.SG = SG
        self.pos = pos
        self.sght = self.sgheadtail()

    def super_nodes_graph(self) -> Tuple[OrderedDict, nx.DiGraph, Dict]:
        """
        Returns
        -------
        supernodes : OrderedDict
        SG : nx.DiGraph
        sgpos : dict
        """
        model = self.model.model  # grab back the original model
        G, ndorder, pos = model_graph(model)

        # ## group nodes into layers
        pmas = (
            self.model.available_modules()
        )  # links the graph nodes to PyTorch modules
        nodes = G.nodes

        # ## a layer starts with an implemented class, some of these correspond to pytorch modules, others were custom made
        # ## for the application, like 'add'
        def chk_type(n):
            try:
                if n.endswith("add") or type(pmas[n]) in self.implemented_classes:
                    return True
            except:
                return False

        # ## The implemented nodes are the first function to apply to input into the DDPlayer,
        # ## the successors which do not change the shape are added to the DDPlayer.
        implemented_nodes = []
        for n in nodes:
            if chk_type(n):
                implemented_nodes.append(n)
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

        #  Rename nodes according to sorted node order for SG
        # sequences layers in the graph

        def sortorder(x):
            return ndorder.index(x)

        ordG = {
            v: (k, v) for k, v in enumerate(sorted(list(SG.nodes()), key=sortorder))
        }
        nx.relabel_nodes(SG, ordG, copy=False)

        sgpos = dict()
        for nd in SG.nodes():
            sgpos[nd] = pos[nd[1]]

        # remove dropout modules
        # for nd in SGnodes[1:]:
        # supernodes[nd] = [n for n in supernodes[nd]
        #                   if type(pmas[n]) not in self.dropout_classes]

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

    def influence_generator(
        self,
        activations: Dict[str, torch.Tensor],
        norm: int = None,
        channels: bool = True,
    ) -> Callable[[int, float], Tuple[sp.coo_matrix, sp.coo_matrix]]:
        """
        Parameters
        ----------
        activations : dict of tensors
        norm : int, optional, default=None
            Specify norm=1 or norm=2 to select influential neurons by L1- or L2-norm, respectively.
            Defaults to select influential neurons by max. values
        channels : bool, optional, default=None
            Profile by channels if True,
            otherwise profile by individual elements

        Returns
        -------
        influential_neurons : function
            A function that will pick out the most influential neurons
            in a layer up to some threshold
        """
        sght = self.sght

        def influential_neurons(
            layer_number: int, threshold: float = 0.1
        ) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
            """
            Parameters
            ----------
            layer_number : int
            threshold : float, optional, default=0.1

            Returns
            -------
            influential_neurons : sp.coo_matrix
                Matrix representing the influential neurons within the threshold
            influential_weights : sp.coo_matrix
                Matrix assigning weights to each influential neuron according to its
                contribution to the threshold
            """
            if layer_number == 0:
                return []
            hd, nd = sght[layer_number]  # head (hd) and tail (nd) modules in layer
            if nd:
                with torch.no_grad():
                    # get activations of head and tail modules
                    hd = activations[hd]
                    nd = activations[nd]

                    # only consider a neuron as potentially influential if its activation
                    # value in the head and tail modules have the same sign
                    head_sign = hd > 0
                    tail_sign = nd > 0
                    t = torch.where(
                        torch.eq(head_sign, tail_sign), nd, torch.zeros(nd.shape)
                    )

                    if norm is not None:
                        if len(t.shape) == 4:
                            if channels:
                                # we have a 3-tensor so take the matrix norm
                                m = torch.linalg.norm(t, ord=norm, dim=(2, 3))
                            else:
                                m = torch.linalg.norm(
                                    t.view((t.shape[0], -1)), ord=norm, dim=0
                                ).unsqueeze(0)
                        else:
                            m = torch.linalg.norm(t, ord=norm, dim=0).unsqueeze(0)
                    else:
                        if len(t.shape) == 4:
                            # we have a 3 tensor so look for greatest contributing channel
                            channel_vals = t.view(t.shape[:2] + (-1,))
                            m, m_idx = torch.max(channel_vals, dim=-1)
                        else:
                            m = t
                        # only takes nonnegative elements
                        m = torch.where(m > 0, m, torch.zeros(m.shape))

                    # sort
                    ordsmat_vals, ordsmat_indices = torch.sort(m, descending=True)

                    # take the cumsum and normalize by total contribution per dim
                    cumsum = torch.cumsum(ordsmat_vals, dim=1)
                    totalsum = cumsum[:, -1]

                    # find the indices within the threshold goal, per dim
                    bool_accept = cumsum / totalsum <= threshold
                    accept = torch.sum(bool_accept, dim=1)

                    accept = torch.where(
                        accept < bool_accept.shape[1], accept, accept - 1
                    )

                    # normalize by final accepted cumsum
                    ordsmat_vals /= cumsum[:, accept]

                    # add additional accept, ie accept + 1
                    try:
                        # use range to enumerate over batch size entries of accept
                        bool_accept[range(len(accept)), accept] = True
                    except IndexError:
                        print("taking all values as influential")

                    # find accepted synapses, all other values zero.
                    # note: it is ordered by largest norm value,
                    # but "unordered" by neuron
                    unordered_weights = torch.where(
                        bool_accept, ordsmat_vals, torch.zeros(ordsmat_vals.shape)
                    )
                    # re-order to mantain proper neuron ordering
                    influential_weights = unordered_weights.gather(
                        1, ordsmat_indices.argsort(1)
                    )

                    if len(t.shape) == 4 and not channels:
                        # m_idx (idx of max neuron per channel) are flat, so start with ch x h*w
                        neuron_weights = torch.zeros(
                            t.shape[1], t.shape[2] * t.shape[3]
                        )
                        # assign infl. weight to max neuron in each infl. channel
                        neuron_weights[range(t.shape[1]), m_idx] = influential_weights
                        # reshape weights to ch x h x w
                        influential_weights = neuron_weights.view(t.shape[1:])

                    influential_neurons = influential_weights.bool().int()

                return matrix_convert(influential_neurons), matrix_convert(
                    influential_weights
                )
            else:
                return sp.coo_matrix((0, 0)), sp.coo_matrix((0, 0))

        return influential_neurons

    def create_layers(self) -> Dict[int, list]:
        """
        Create a dictionary of layers to profile.

        Returns
        -------
        layerdict : OrderedDict
            A dictionary of layers keyed by their position from the beginning of the model according to the order
            Pytorch uses to list the layers of the model.

        Note
        ----
        Because a weighted module is often followed by an activation and/or normalization modules
        those modules are combined into a single layer to be profiled. As a consequence
        the number of layers generated will typically be less than the number of available
        modules in the model.

        """
        # identify layers to profile.
        layerdict = OrderedDict()  # ordered dictionary to feed to profiler

        layerdict[0] = [
            ["x_in"],
            "contrib_identity",
        ]  # this is the tensor input for the image

        SGnodes = sorted(self.SG.nodes, key=lambda nd: nd[0])
        for ndx in range(1, len(SGnodes)):  # sorted(SG.nodes,key=lambda nd: nd[0]):
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

    def single_profile(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        neuron_counts: sp.coo_matrix,
        layerdict: Dict[int, list],
        ldx: int,
        threshold: float,
        channels: bool,
    ) -> Tuple[
        Union[sp.coo_matrix, Tuple[sp.coo_matrix, sp.coo_matrix]],
        sp.coo_matrix,
        sp.coo_matrix,
    ]:
        """
        Profiles a single layer

        Parameters
        ----------
        x_in : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        y_out : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        neuron_counts : sp.coo_matrix
            Matrix representing the influential neurons in the layer
        layerdict : OrderedDict[int, list]
            Keyed by consecutive layers to be profiled
        ldx : int
            Layer number of the layer to be profiled
        threshold : float
            Percentage of contribution to track in a profile
        channels : bool
            profile by channels if True,
            otherwise profile by individual elements

        Returns
        -------
        neuron_counts : sp.coo_matrix or tuple of sp.coo_matrix
        synapse_counts : sp.coo_matrix
        synapse_weights : sp.coo_matrix

        """

        func = getattr(self.__class__, layerdict[ldx][1])
        # get list of influential indices
        flat_idx = neuron_counts.nonzero()
        if channels or func is TorchProfiler.contrib_linear:
            infl_idx = flat_idx[1]
        # unravel to 3D for element-wise conv. layer
        else:
            row_idx, col_idx = np.unravel_index(flat_idx[1], y_out[ldx].shape[-2:])
            infl_idx = np.stack((flat_idx[0], row_idx, col_idx)).T
        infl_idx = torch.Tensor(infl_idx).long()
        # return ncs, scs, sws
        return func(
            self,
            x_in,
            y_out,
            infl_idx,
            layerdict[ldx][0],
            threshold=threshold,
            channels=channels,
        )

        # Two steps:
        # 1. Compute influential neurons out of each layers attributions - assign weights
        # 2. Compute contributing neurons into each layer - assign weights
        # 3. Create dangling edges for each layer???
        # 4. Does PageRank change the edge weightings? If not then we could use existing weights and stop.

    def create_profile(
        self,
        x: torch.Tensor,
        layerdict: OrderedDict,
        layers_to_profile: Union[list, Tuple] = None,
        threshold: float = 0.1,
        norm: int = None,
        channels: bool = True,
    ) -> Profile:
        """
        Generate a profile for a single input data x as it passes through the layers in layerdict

        Parameters
        ----------
        x : torch.Tensor
            input to model being profiled
        layerdict : OrderedDict
            keyed by consecutive layers to be profiled
        layers_to_profile : list or tuple
            list of specific layers to profile or tuple with first,last layers
            (exclusive of last) to profile and all layers inbetween
        threshold : float, optional, default=0.1
            Percentage of contribution to track in a profile. ### TODO - allow a dictionary of varied thresholds per layer
        norm : int, optional, default=None
            Specify norm=1 or norm=2 to select influential neurons by L1- or L2-norm, respectively. Defaults
            to select influential neurons by max. values
        channels: bool, optional, default=True
            Neurons can be defined as channels or as individual elements of the activations. Set channels=True
            create a channel-wise profile and channels=False to create an element-wise profile.

        Returns
        -------
        Profile
            profile contains neuron_counts, neuron_weights, synapse_counts, and synapse_weights
            across layers in layerdict. Corresponding number of images = 1

        """

        if not channels:
            neuron_type = "element"
            warnings.warn(
                "Warning: computation of element-wise profiles is space-intensive, use a low threshold "
                + "(ideally 0.1 or lower for VGG16 or comparable architecture) to avoid memory strain"
            )
        else:
            neuron_type = "channel"

        # Create empty dictionaries for storing neurons and synapses
        neuron_counts = dict()
        neuron_weights = (
            dict()
        )  # these are only assigned to influential neurons on first pass
        synapse_counts = dict()
        synapse_weights = dict()

        ######## Step 0: Grab available modules at head and tail of each layer for easy reference ########
        sght = self.sgheadtail()

        with torch.no_grad():
            n = len(layerdict)
            y, activations = self.model.forward(x)
            activations["x_in"] = x

            ######## Step 1: Identify the Influential Neurons (Nodes) in the Activation Graph ########

            # Create an x-dependent function to identify  the the highest valued neurons
            # at each layer
            influential_neurons = self.influence_generator(activations, norm, channels)

            # Layers to profile:
            if layers_to_profile == None:
                ltp = range(1, n)
            elif isinstance(layers_to_profile, list):
                ltp = [lyr for lyr in layers_to_profile if lyr >= 1 or lyr <= n - 1]
            else:  # a tuple is expected
                start = max(1, layers_to_profile[0])
                end = min(n, layers_to_profile[1])
                ltp = range(start, end)

            # Fill dictionary
            for k in ltp:
                neuron_counts[k], neuron_weights[k] = influential_neurons(
                    k, threshold=threshold
                )

            ######## Step 2: Identify Contributing Neurons (Nodes) and Synaptic connections (Edges) ########

            # Assign activations to x_in and y_out for each layer
            # Note that if f = first available module in a layer
            # y_out = f(x_in)
            y_out = dict()
            x_in = dict()
            for k in ltp:
                y_out[k] = {
                    k: activations[sght[k][0]]
                }  # this is the output from the first available module in the layer
                nd = (k, layerdict[k][0][0])
                pred_layers = list(self.SG.predecessors(nd))
                x_in[k] = {
                    pd[0]: activations[sght[pd[0]][1]] for pd in pred_layers
                }  # this is the output from the last available module in the layer

            results = [
                self.single_profile(
                    x_in[ldx],
                    y_out[ldx],
                    neuron_counts[ldx],
                    layerdict,
                    ldx,
                    threshold,
                    channels,
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

        return Profile(
            neuron_counts=neuron_counts,
            neuron_weights=neuron_weights,
            synapse_counts=synapse_counts,
            synapse_weights=synapse_weights,
            num_inputs=1,
            neuron_type=neuron_type,
        )

    def contrib_max2d(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str],
        threshold: float = None,
        channels: bool = True,
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix]:
        """
        Return the contributing synapse for a torch.nn.Max2D layer

        Parameters
        ----------
        x_in : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        y_out : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        infl_neurons : torch.Tensor
            tensor containing indices of influential neurons in y_out
            dimensions: num_influential if channel-wise
                        num_influential,3 if element-wise
        layer : List[str]
            list containing single key in self.model.available_modules() dictionary
        threshold : None or float
            not used, placeholder for uniformity in arguments.
        channels : bool
            profile by channels if True,
            otherwise profile by individual elements

        Returns
        -------
        neuron_counts : sp.coo_matrix
        synapse_counts : sp.coo_matrix
        synapse_weights : sp.coo_matrix
        """

        # grab tensors from dictionaries
        if len(y_out) > 1 or len(x_in) > 1:
            raise NotImplementedError(
                "contrib_max2d cannot handle more than one dict"
                + "key for x_in or y_out"
            )
        else:
            y_ldx = list(y_out.keys())[0]
            y_out = y_out[y_ldx]
            x_ldx = list(x_in.keys())[0]
            x_in = x_in[x_ldx]

        with torch.no_grad():
            xdims = x_in.shape
            ydims = y_out.shape

            if channels:
                # copy of contrib_identity below; another approach is to make
                # a copy @staticmethod version of contrib_identity to torch_profiler
                ch = infl_neurons

                neuron_counts = torch.zeros(xdims[:2], dtype=torch.int)
                synapse_counts = torch.zeros(ydims[1], xdims[1], dtype=torch.int)
                synapse_weights = torch.zeros(ydims[1], xdims[1])

                neuron_counts[:, ch] = 1
                synapse_counts[ch, ch] = 1
                synapse_weights[ch, ch] = 1
            else:
                maxpool = self.model.available_modules()[layer[0]]

                # Grab dimensions of maxpool from parameters
                stride = maxpool.stride
                kernel_size = maxpool.kernel_size
                padding = maxpool.padding

                # get number of influential neurons (batchsize)
                num_infl = infl_neurons.shape[0]
                # get channel, i, j indices (unravel if flat)
                if len(infl_neurons.shape) == 1:
                    ch, i, j = get_index(infl_neurons, kernel_size)
                else:
                    ch, i, j = infl_neurons.unbind(dim=1)

                # find submatrix/receptive field for each influential neuron
                # xmat dimensions: num_infl x kernel_size x kernel_size
                xmat = submatrix_generator(x_in, stride, kernel_size, padding)(i, j)[
                    range(num_infl), ch
                ]
                # find max. val. from each submatrix/receptive field
                maxval, maxidx = torch.max(xmat.view(num_infl, kernel_size ** 2), dim=1)

                assert torch.allclose(
                    maxval, y_out[0, ch, i, j], rtol=1e-04, atol=1e-4
                ), f"maxpool failure: {maxval - y_out[0,ch,i,j]}"

                # convert indices of max. vals. from submatrix to full index in x_in
                maxi = (maxidx // kernel_size + stride * i) - padding
                maxj = (maxidx % kernel_size + stride * j) - padding

                # construct neuron counts
                neuron_counts = torch.zeros(xdims[1:], dtype=torch.int)
                neuron_counts[ch, maxi, maxj] += 1

                # construct sparse tensor synapse counts and weights
                # sparse index represents synapse as [ch_out,i_out,j_out,ch_in,i_in_j_in] (ch_out == ch_in for maxpool)
                indices = torch.stack((ch, i, j, ch, maxi, maxj))
                values = torch.ones(num_infl)
                size = ydims[1:] + xdims[1:]
                synapse_counts = torch.sparse_coo_tensor(
                    indices, values, size, dtype=torch.int
                ).coalesce()
                synapse_weights = torch.sparse_coo_tensor(
                    indices, values, size
                ).coalesce()

        return (
            matrix_convert(neuron_counts),
            matrix_convert(synapse_counts),
            matrix_convert(synapse_weights),
        )

    def contrib_adaptive_avg_pool2d(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str],
        threshold: float = 0.1,
        channels: bool = True,
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix]:
        """
        Return the contributing synapses for a torch.nn.AdaptiveAveragePool layer

        Parameters
        ----------
        x_in : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        y_out : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        infl_neurons : torch.Tensor
            tensor containing indices of influential neurons in y_out
            dimensions: num_influential if channel-wise
                        num_influential,3 if element-wise
        layer : List[str]
            list containing single key in self.model.available_modules() dictionary
        threshold : float
        channels : bool
            profile by channels if True,
            otherwise profile by individual elements

        Returns
        -------
        neuron_counts : sp.coo_matrix
        synapse_counts : sp.coo_matrix
        synapse_weights : sp.coo_matrix
        """
        # grab tensors from dictionaries
        if len(y_out) > 1 or len(x_in) > 1:
            raise NotImplementedError(
                "contrib_adaptive_avg_pool2d cannot "
                + "handle more than one dict "
                + "key for x_in or y_out"
            )
        else:
            y_ldx = list(y_out.keys())[0]
            y_out = y_out[y_ldx]
            x_ldx = list(x_in.keys())[0]
            x_in = x_in[x_ldx]

        with torch.no_grad():
            xdims = x_in.shape
            ydims = y_out.shape

            if channels:
                # copy of contrib_identity below; another approach is to make
                # a copy @staticmethod version of contrib_identity to torch_profiler
                ch = infl_neurons

                neuron_counts = torch.zeros(xdims[:2], dtype=torch.int)
                synapse_counts = torch.zeros(ydims[1], xdims[1], dtype=torch.int)
                synapse_weights = torch.zeros(ydims[1], xdims[1])

                neuron_counts[:, ch] = 1
                synapse_counts[ch, ch] = 1
                synapse_weights[ch, ch] = 1
            else:
                avgpool = self.model.available_modules()[layer[0]]

                # Grab dimensions of avgpool from parameters
                output_size = avgpool.output_size[0]
                input_size = xdims[-1]
                stride = input_size // output_size
                kernel_size = input_size - (output_size - 1) * stride

                # get number of influential neurons (batchsize)
                num_infl = infl_neurons.shape[0]
                # get channel, i, j indices (unravel if flat)
                if len(infl_neurons.shape) == 1:
                    ch, i, j = get_index(infl_neurons, kernel_size)
                else:
                    ch, i, j = infl_neurons.unbind(dim=1)

                # set multiplier for computing average
                scalar = 1 / (kernel_size ** 2)
                # set goal at thresholded percentage of influential neuron activations
                goal = threshold * y_out[0, ch, i, j]

                # find submatrix/receptive field for each influential neuron
                # xmat dimensions: num_infl x kernel_size x kernel_size
                xmat = submatrix_generator(x_in, stride, kernel_size)(i, j)[
                    range(num_infl), ch
                ]
                # only consider values > 0
                xmat = torch.where(xmat > 0, xmat, torch.zeros(xmat.shape))

                # order neurons in submatrix/receptive field by greatest value/contribution (times scalar multiplier)
                ordsmat_vals, ordsmat_indices = torch.sort(
                    xmat.view(num_infl, kernel_size ** 2) * scalar,
                    dim=1,
                    descending=True,
                )
                # take cumulative sum of ordered values
                cumsum = torch.cumsum(ordsmat_vals, dim=1)

                assert torch.allclose(
                    cumsum[:, -1], y_out[0, ch, i, j], rtol=1e-04, atol=1e-4
                ), f"avgpool failure: {cumsum[-1] - y_out[0,ch,i,j]}"

                # find the indices within the threshold goal, per dim
                bool_accept = cumsum <= goal.unsqueeze(-1)
                accept = torch.sum(bool_accept, dim=1)
                # if accept == kernel_size**2, all values taken as contributors
                # subtract 1 in this case to avoid IndexError when adding additional accept
                accept = torch.where(accept < kernel_size ** 2, accept, accept - 1)

                # add additional accept, ie accept + 1
                bool_accept[range(len(accept)), accept] = True
                # normalize values to get weights
                ordsmat_vals /= cumsum[range(len(accept)), accept].unsqueeze(-1)

                # convert accepted contributor indices in submatrix to full index in x_in
                ordsi = ordsmat_indices // kernel_size + stride * i.unsqueeze(-1)
                ordsj = ordsmat_indices % kernel_size + stride * j.unsqueeze(-1)

                # repeat each ch, i, j index for each contributor to that influential neuron
                ch = ch.repeat_interleave(accept + 1)
                i = i.repeat_interleave(accept + 1)
                j = j.repeat_interleave(accept + 1)

                # only take accepted indices
                ordsi = ordsi[bool_accept]
                ordsj = ordsj[bool_accept]

                # construct neuron counts
                neuron_counts = torch.zeros(xdims[1:], dtype=torch.int)
                neuron_counts[ch, ordsi, ordsj] += 1

                # construct sparse tensor synapse counts and weights
                # sparse index represents synapse as [ch_out,i_out,j_out,ch_in,i_in_j_in] (ch_out == ch_in for avgpool)
                indices = torch.stack((ch, i, j, ch, ordsi, ordsj))
                # only take accepted values
                values = ordsmat_vals[bool_accept]
                size = ydims[1:] + xdims[1:]
                synapse_counts = torch.sparse_coo_tensor(
                    indices, torch.ones(values.shape, dtype=torch.int), size
                ).coalesce()
                synapse_weights = torch.sparse_coo_tensor(
                    indices, values, size
                ).coalesce()

        return (
            matrix_convert(neuron_counts),
            matrix_convert(synapse_counts),
            matrix_convert(synapse_weights),
        )

    def contrib_conv2d(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str],
        threshold: float = 0.1,
        channels: bool = True,
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix]:
        """
        Profile a single output neuron from a 2d conv layer

        Parameters
        ----------
        x_in : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        y_out : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        infl_neurons : torch.Tensor
            tensor containing indices of influential neurons in y_out
            dimensions: num_influential if channel-wise
                        num_influential,3 if element-wise
        layer : List[str]
            list containing keys in self.model.available_modules() dictionary
            for conv2d these will refer to a convolutional module and an
            activation module
        threshold : float
        channels : bool
            profile by channels if True,
            otherwise profile by individual elements

        Returns
        -------
        neuron_counts : sp.coo_matrix
        neuron_weights : sp.coo_matrix
        synapse_counts : sp.coo_matrix

        Note
        ----
        Only implemented for convolution using filters with same height and width
        and strides equal in both dimensions and padding equal in all dimensions

        Synapse profiles for conv2d are indexed by 3 sets of tuples one for each neuron
        and on one for the index of the filter used.
        """
        if len(y_out) > 1 or len(x_in) > 1:
            raise NotImplementedError(
                "contrib_conv2d cannot handle more than one dict"
                + "key for x_in or y_out"
            )
        else:
            y_ldx = list(y_out.keys())[0]
            y_out = y_out[y_ldx]
            x_ldx = list(x_in.keys())[0]
            x_in = x_in[x_ldx]

        with torch.no_grad():
            conv = layer[0]
            conv = self.model.available_modules()[conv]

            # assumption is that kernel size, stride are equal in both dimensions
            # and padding preserves input size
            kernel_size = conv.kernel_size[0]
            stride = conv.stride[0]
            padding = conv.padding[0]
            dilation = conv.dilation[0]
            W = conv._parameters["weight"]
            B = (
                conv._parameters["bias"]
                if conv._parameters["bias"] is not None
                else None
            )

            # define dimensions, in channels and out channels
            xdims = x_in.shape
            ydims = y_out.shape
            in_channels, h_in, w_in = xdims[1:]
            out_channels, h_out, w_out = ydims[1:]

            # get number of influential neurons
            num_infl = infl_neurons.shape[0]

            if channels:
                ch = infl_neurons

                # this is not [:,ch] because W dims are [#filters=outdim, #channels = indim, ker_size,ker_size] - don't erase this
                W = W[ch]
                B = B[ch] if B is not None else torch.Tensor()

                y_true = y_out[0, ch, :, :].detach()

                # reshape to batch convolution for num_influential neurons
                W2 = W.view(
                    num_infl * in_channels,  ## changed to in_channels
                    kernel_size,
                    kernel_size,
                )
                x_in_stacked = x_in.repeat(num_infl, 1, 1, 1)
                # take the depthwise convolution .unsqueeze(-1)
                # reshape x_in_stacked so that we can batch the convolution

                z = F.conv2d(
                    x_in_stacked.view(1, in_channels * num_infl, h_in, w_in),
                    W2.unsqueeze(1),
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=in_channels * num_infl,
                )

                # reshape to be expected [num_influential, in_channels, h, w]
                z = z.view(num_infl, in_channels, h_out, w_out)

                # ignore negative values
                z_nonzero = torch.where(z > 0, z, torch.zeros(z.shape))

                # find the max value per channel
                maxvals = F.max_pool2d(z_nonzero, kernel_size=z.shape[-1]).view(
                    z.shape[0], -1
                )

                # find the top values/channels out of the threshold
                ordsmat_vals, ordsmat_indices = torch.sort(maxvals, descending=True)
                cumsum = torch.cumsum(ordsmat_vals, dim=1)
                totalsum = cumsum[:, -1].unsqueeze(-1)
                goals = cumsum / totalsum  # normalize

                # find the indices within the threshold goal, per dim
                bool_accept = goals <= threshold
                accept = torch.sum(bool_accept, dim=1)
                accept = torch.where(accept < cumsum.shape[1], accept, accept - 1)

                # add additional accept, ie accept + 1
                ordsmat_vals /= cumsum[range(len(accept)), accept].unsqueeze(-1)
                bool_accept[range(len(accept)), accept] = True

                # find accepted synapses, all other values zero.
                unordered_synapses = torch.where(
                    bool_accept, ordsmat_vals, torch.zeros(ordsmat_vals.shape)
                )

                # re-order to mantain proper neuron ordering
                sws_compact = unordered_synapses.gather(1, ordsmat_indices.argsort(1))

                # fullify synapse weights to outdims x indims
                synapse_weights = torch.zeros(out_channels, in_channels)
                synapse_weights[ch] = sws_compact

                synapse_counts = synapse_weights.bool().int()

                # construct neuron counts
                neuron_counts = torch.sum(synapse_counts, dim=0).unsqueeze(0).int()

            else:
                # get channel, i, j indices (unravel if flat)
                if len(infl_neurons.shape) == 1:
                    ch, i, j = get_index(infl_neurons, kernel_size)
                else:
                    ch, i, j = infl_neurons.unbind(dim=1)

                # this is not [:,ch] because W dims are [#filters=outdim, #channels = indim, ker_size,ker_size] - don't erase this
                W = W[ch]
                B = B[ch] if B is not None else torch.Tensor()

                y_true = y_out[0, ch, i, j]

                # find receptive fields for each influential neuron
                # xmat dimensions: num_infl x in_channels x kernel_size x kernel_size
                xmat = submatrix_generator(x_in, stride, kernel_size, padding)(i, j)

                # convolve weights and receptive fields
                z = W * xmat
                # order neurons in receptive field by greatest value/contribution
                # ordsmat dimensions: num_infl x in_channels*kernel_size^2
                ordsmat_vals, ordsmat_indices = torch.sort(
                    z.view(num_infl, in_channels * kernel_size ** 2),
                    dim=1,
                    descending=True,
                )
                # take cumulative sum of ordered values
                cumsum = torch.cumsum(ordsmat_vals, dim=1)
                # apply ReLU or other activation module for comparison
                # values = actf(cumsum)

                # assert torch.allclose(
                #     values[:,-1] + B,
                #     y_true,
                #     rtol=1e-04,
                #     atol=1e-4
                # ), f'conv2d failure: {values[:,-1] - y_true}'

                # normalize for comparison to the threshold
                totalsum = cumsum[:, -1].unsqueeze(-1)
                goals = cumsum / totalsum

                # find the indices within the threshold goal, per dim
                bool_accept = goals <= threshold
                accept = torch.sum(bool_accept, dim=1)
                # if accept == kernel_size**2, all values taken as contributors
                # subtract 1 in this case to avoid IndexError when adding additional accept
                accept = torch.where(accept < kernel_size ** 2, accept, accept - 1)

                # normalize
                ordsmat_vals /= cumsum[range(len(accept)), accept].unsqueeze(-1)
                # add additional accept, ie accept + 1
                bool_accept[range(len(accept)), accept] = True

                # convert flattened contributor indices to index in receptor field
                ords_ch, ords_row, ords_col = get_index(ordsmat_indices, kernel_size)
                # convert receptor field index to full index in x_in
                ordsi = ords_row + stride * i.unsqueeze(-1) - padding
                ordsj = ords_col + stride * j.unsqueeze(-1) - padding

                # repeat each ch, i, j index for each contributor to that influential neuron
                ch = ch.repeat_interleave(accept + 1)
                i = i.repeat_interleave(accept + 1)
                j = j.repeat_interleave(accept + 1)
                # only take accepted indices
                ords_ch = ords_ch[bool_accept]
                ordsi = ordsi[bool_accept]
                ordsj = ordsj[bool_accept]

                # indices represents synapses as [ch_out,i_out,j_out,ch_in,i_in_j_in]
                indices = torch.stack((ch, i, j, ords_ch, ordsi, ordsj))
                # only take accepted values
                values = ordsmat_vals[bool_accept]
                size = ydims[1:] + xdims[1:]

                # identify and remove contributor indices that come from the padded margin
                # padding indices have either row or col index outside of the range [0, #row/col)
                # addition gives logical OR between boolean tensors
                padding_idx = (
                    (indices[4] < 0)
                    + (indices[4] >= size[4])
                    + (indices[5] < 0)
                    + (indices[5] >= size[5])
                )
                standard_idx = torch.logical_not(padding_idx)

                # remove padding indices and values
                indices = indices[:, standard_idx]
                values = values[standard_idx]

                # take [ch_in,i_in,j_in] rows from synapse indices
                ch_in = indices[3]
                i_in = indices[4]
                j_in = indices[5]
                # construct neuron counts
                neuron_counts = torch.zeros(xdims[1:], dtype=torch.int)
                neuron_counts[ch_in, i_in, j_in] += 1

                # construct sparse tensor counts and weights
                synapse_counts = torch.sparse_coo_tensor(
                    indices, torch.ones(values.shape, dtype=torch.int), size
                ).coalesce()
                synapse_weights = torch.sparse_coo_tensor(
                    indices, values, size
                ).coalesce()

        return (
            matrix_convert(neuron_counts),
            matrix_convert(synapse_counts),
            matrix_convert(synapse_weights),
        )

    def contrib_linear(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str],
        threshold: float = 0.1,
        channels: bool = True,
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix]:
        """
        Profile output neurons from a linear layer

        Parameters
        ----------
        x_in : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        y_out : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        infl_neurons : torch.Tensor
            tensor containing indices of influential neurons in y_out
            dimensions: num_influential
        layer : List[str]
            list containing single key in self.model.available_modules() dictionary
        threshold : float
        channels : bool
            profile by channels if True,
            otherwise profile by individual elements

        Returns
        -------
        neuron_counts : sp.coo_matrix
        synapse_counts : sp.coo_matrix
        synapse_weights : sp.coo_matrix
        """
        # use the same profiling for channels or neurons for linear
        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_linear cannot handle more than one dict"
                + "key for x_in or y_out"
            )
        else:
            y_ldx = list(y_out.keys())[0]
            y_out = y_out[y_ldx]
            x_ldx = list(x_in.keys())[0]
            x_in = x_in[x_ldx]

        j = infl_neurons

        with torch.no_grad():
            linear = layer[0]
            linear = self.model.available_modules()[linear]

            W = linear._parameters["weight"]
            B = linear._parameters["bias"]

            # subtract bias here, and then ignore for cumsum/totalsum ops
            goal = threshold * y_out[0, j]  # 1d tensor with batch size elements # noqa
            if torch.all(goal <= 0):
                warnings.warn("an output neuron is less than 0")
            ydims = y_out[0].shape
            xdims = x_in[0].shape
            # change if statement when we batch multiple images
            if len(xdims) > 1:
                holdx = torch.Tensor(x_in)
                x_in = x_in[0].flatten().unsqueeze(0)
            z = x_in[0] * W[j]

            # testing can also be batched, i.e.
            assert torch.isclose(torch.sum(z, dim=1) + B[j], y_out[0, j]).all()

            # ignore negative values
            z = torch.where(z > 0, z, torch.zeros(z.shape))

            ordsmat_vals, ordsmat_indices = torch.sort(z, descending=True)

            # take the cumsum and normalize by total contribution per dim
            cumsum = torch.cumsum(ordsmat_vals, dim=1).detach()

            # find the indices within the threshold goal, per dim
            bool_accept = cumsum + B[j].unsqueeze(-1) <= goal.unsqueeze(-1)
            accept = torch.sum(bool_accept, dim=1)

            # add additional accept, ie accept + 1
            try:
                ordsmat_vals /= cumsum[range(len(accept)), accept].unsqueeze(-1)
                bool_accept[range(len(accept)), accept] = True
            except IndexError:
                # print("taking all values as influential")
                pass

            # find accepted synapses, all other values zero.
            unordered_synapses = torch.where(
                bool_accept, ordsmat_vals, torch.zeros(ordsmat_vals.shape)
            )
            # re-order to mantain proper neuron ordering
            # note: ordered by largest norm value, but "unordered" by neuron
            sws_compact = unordered_synapses.gather(1, ordsmat_indices.argsort(1))

            # sum contribution per channel if x_in is a conv layer
            if len(xdims) > 1 and channels:
                sws_compact = sws_compact.view(len(j), xdims[0], -1)
                sws_compact = torch.sum(sws_compact, dim=-1)

            # fullify synapse weights to outdims x indims
            if len(xdims) == 1 or channels:
                synapse_weights = torch.zeros(ydims[0], xdims[0])
            else:
                synapse_weights = torch.zeros(ydims[0], xdims[0] * xdims[1] * xdims[2])
            synapse_weights[j] = sws_compact

            synapse_counts = synapse_weights.bool().int()

            # construct neuron counts
            neuron_counts = torch.sum(synapse_counts, dim=0).unsqueeze(0).int()
            if len(xdims) > 1 and not channels:
                synapse_counts = synapse_counts.view(
                    ydims[0], xdims[0], xdims[1], xdims[2]
                )
                synapse_weights = synapse_weights.view(
                    ydims[0], xdims[0], xdims[1], xdims[2]
                )
                neuron_counts = neuron_counts.view(xdims)

        return (
            matrix_convert(neuron_counts),
            matrix_convert(synapse_counts),
            matrix_convert(synapse_weights),
        )

    def contrib_identity(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str] = None,
        threshold: float = None,
        channels: bool = True,
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix]:
        """
        Pass through to keep influential neurons from one layer fixed into the
        next. Used for normalization layers

        Parameters
        ----------
        x_in : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        y_out : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        infl_neurons : torch.Tensor
            tensor containing indices of influential neurons in y_out
            dimensions: num_influential if channel-wise
                        num_influential,3 if element-wise
        layer : None or List[str]
            not used, placeholder for uniformity in arguments
        threshold : None or float
            not used, placeholder for uniformity in arguments
        channels : bool
            profile by channels if True,
            otherwise profile by individual elements


        Returns
        -------
        neuron_counts
        synapse_counts
        synapse_weights
        """
        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_identity requires x_in and y_out to have length 1"
            )
        else:
            y_ldx = list(y_out.keys())[0]
            y_out = y_out[y_ldx]
            x_ldx = list(x_in.keys())[0]
            x_in = x_in[x_ldx]

        xdims = x_in.shape
        ydims = y_out.shape

        if channels:
            ch = infl_neurons
            neuron_counts = torch.zeros(xdims[:2], dtype=torch.int)
            synapse_counts = torch.zeros(ydims[1], xdims[1], dtype=torch.int)
            synapse_weights = torch.zeros(ydims[1], xdims[1])

            neuron_counts[:, ch] += 1
            synapse_counts[ch, ch] += 1
            synapse_weights[ch, ch] = 1
        else:
            ch, i, j = infl_neurons.unbind(dim=1)
            neuron_counts = torch.zeros(xdims[1:], dtype=torch.int)
            neuron_counts[ch, i, j] += 1

            indices = torch.stack((ch, i, j, ch, i, j))
            values = torch.ones((len(infl_neurons)))

            size = ydims[1:] + xdims[1:]

            synapse_counts = torch.sparse_coo_tensor(
                indices, values, size, dtype=torch.int
            )
            synapse_weights = torch.sparse_coo_tensor(indices, values, size)
        return (
            matrix_convert(neuron_counts),
            matrix_convert(synapse_counts),
            matrix_convert(synapse_weights),
        )

    def contrib_resnetadd(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str] = None,
        threshold: float = None,
        channels: bool = True,
    ) -> Tuple[Tuple[sp.coo_matrix, sp.coo_matrix], sp.coo_matrix, sp.coo_matrix]:
        """
        Parameters
        ----------
        x_in : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        y_out : Dict[int, torch.Tensor]
            dict where key is layer, value is tensor
            dimensions: batchsize,channels,height,width
        infl_neurons : torch.Tensor
            tensor containing indices of influential neurons in y_out
            dimensions: num_influential if channel-wise
                        num_influential,3 if element-wise
        layer : None or List[str]
            not used, placeholder for uniformity in arguments
        threshold : None or float
            not used, placeholder for uniformity in arguments
        channels : bool
            profile by channels if True,
            otherwise profile by individual elements

        Returns
        -------
        neuron_counts
        synapse_counts
        synapse_weights

        Raises
        ------
        NotImplementedError
            Raises error if len(x_in) != 2 or len(y_out) != 1
        """

        if len(y_out) != 1 or len(x_in) != 2:
            raise NotImplementedError(
                "contrib_resnetadd requires exactly one item "
                + "in y_out and two in x_in "
            )
        else:
            y_ldx = list(y_out.keys())[0]
            y_out = y_out[y_ldx]
            dims = y_out.shape

            x1_ldx, x2_ldx = sorted(x_in.keys())
            x1_in = x_in[x1_ldx]
            x2_in = x_in[x2_ldx]

            neuron_counts = dict()
            synapse_counts = dict()
            synapse_weights = dict()
            with torch.no_grad():
                if channels:
                    maxvals = torch.zeros(2, dims[1])
                    for i, x in enumerate((x1_in, x2_in)):
                        channel_vals = x.view(x.shape[:2] + (-1,))
                        maxvals[i] = torch.max(channel_vals, dim=-1)[0]
                    maxvals = maxvals[:, infl_neurons]

                    ch_x1 = infl_neurons[maxvals[0] > maxvals[1]]
                    ch_x2 = infl_neurons[maxvals[0] <= maxvals[1]]

                    for i, ch in enumerate((ch_x1, ch_x2)):
                        neuron_counts[i] = torch.zeros(dims[:2], dtype=torch.int)
                        synapse_counts[i] = torch.zeros(
                            dims[1], dims[1], dtype=torch.int
                        )
                        synapse_weights[i] = torch.zeros(dims[1], dims[1])

                        neuron_counts[i][:, ch] = 1
                        synapse_counts[i][ch, ch] = 1
                        synapse_weights[i][ch, ch] = 1

                else:
                    ch, i, j = infl_neurons.unbind(dim=1)
                    vals = torch.stack((x1_in[0, ch, i, j], x2_in[0, ch, i, j]))

                    infl_x1 = infl_neurons[vals[0] > vals[1]]
                    infl_x2 = infl_neurons[vals[0] <= vals[1]]

                    for i, infl in enumerate((infl_x1, infl_x2)):
                        neuron_counts[i] = torch.zeros(dims[1:], dtype=torch.int)
                        neuron_counts[i][infl[:, 0], infl[:, 1], infl[:, 2]] = 1

                        indices = torch.cat((infl, infl), dim=1).T
                        values = torch.ones(len(infl), dtype=torch.int)
                        size = dims[1:] + dims[1:]

                        synapse_counts[i] = torch.sparse_coo_tensor(
                            indices, values, size
                        ).coalesce()
                        synapse_weights[i] = torch.sparse_coo_tensor(
                            indices, values, size
                        ).coalesce()

            neuron_counts = tuple(matrix_convert(neuron_counts[i]) for i in range(2))
            synapse_counts = sp.block_diag(
                [matrix_convert(synapse_counts[i]) for i in range(2)]
            )
            synapse_weights = sp.block_diag(
                [matrix_convert(synapse_weights[i]) for i in range(2)]
            )

            return neuron_counts, synapse_counts, synapse_weights

    def dict_view(
        self,
        profile: Profile,
        layerdict: OrderedDict = None,
    ) -> Profile:
        """
        Creates a dictionary view of a sparse matrix formatted Profile
        Parameters
        ----------
        profile : Profile
            Profile to be reformatted
        layerdict : OrderedDict, optional, default=None
            keyed by consecutive layers to be profiled

        Returns
        -------
         : Profile
            Profile with counts and weights formatted as dicts
        """
        nc_dict = dict()
        nw_dict = dict()
        sc_dict = dict()
        sw_dict = dict()

        layerdict = layerdict or self.create_layers()

        for ldx in profile.neuron_counts:

            if ldx in profile.neuron_weights:
                if profile.neuron_weights[ldx].shape[0] == 1:
                    channels = profile.neuron_weights[ldx].col
                    weights = profile.neuron_weights[ldx].data
                    nw_dict[ldx] = {
                        (ldx, (ch,)): wt for ch, wt in zip(channels, weights)
                    }
                else:
                    channels = profile.neuron_weights[ldx].row
                    elements = profile.neuron_weights[ldx].col
                    weights = profile.neuron_weights[ldx].data
                    nw_dict[ldx] = {
                        (ldx, (ch, el)): wt
                        for ch, el, wt in zip(channels, elements, weights)
                    }

            if profile.neuron_counts[ldx].shape[0] == 1:
                channels = profile.neuron_counts[ldx].col
                counts = profile.neuron_counts[ldx].data
                nc_dict[ldx] = {(ldx, (ch,)): ct for ch, ct in zip(channels, counts)}
            else:
                channels = profile.neuron_counts[ldx].row
                elements = profile.neuron_counts[ldx].col
                counts = profile.neuron_counts[ldx].data
                nc_dict[ldx] = {
                    (ldx, (ch, el)): ct
                    for ch, el, ct in zip(channels, elements, counts)
                }

        for ldx in profile.synapse_counts:
            nd = (ldx, layerdict[ldx][0][0])
            pred_ldx = sorted([pd[0] for pd in self.SG.predecessors(nd)])
            if ldx in profile.synapse_weights:
                if len(pred_ldx) == 2:
                    split = int(profile.synapse_weights[ldx].shape[0] / 2)
                    synapse_weights = (
                        profile.synapse_weights[ldx].tocsr()[:split, :split].tocoo(),
                        profile.synapse_weights[ldx].tocsr()[split:, split:].tocoo(),
                    )
                else:
                    synapse_weights = (profile.synapse_weights[ldx],)

                sw_dict[ldx] = dict()
                for pd, sw in zip(pred_ldx, synapse_weights):
                    out_neurons = sw.row
                    in_neurons = sw.col
                    weights = sw.data
                    sw_dict[ldx].update(
                        {
                            ((pd, (i,)), (ldx, (o,))): wt
                            for i, o, wt in zip(in_neurons, out_neurons, weights)
                        }
                    )

            if len(pred_ldx) == 2:
                split = int(profile.synapse_counts[ldx].shape[0] / 2)
                synapse_counts = (
                    profile.synapse_counts[ldx].tocsr()[:split, :split].tocoo(),
                    profile.synapse_counts[ldx].tocsr()[split:, split:].tocoo(),
                )
            else:
                synapse_counts = (profile.synapse_counts[ldx],)

            sc_dict[ldx] = dict()
            for pd, sc in zip(pred_ldx, synapse_counts):
                out_neurons = sc.row
                in_neurons = sc.col
                counts = sc.data
                sc_dict[ldx].update(
                    {
                        ((pd, (i,)), (ldx, (o,))): wt
                        for i, o, wt in zip(in_neurons, out_neurons, counts)
                    }
                )

        return Profile(
            neuron_counts=nc_dict,
            neuron_weights=nw_dict,
            synapse_counts=sc_dict,
            synapse_weights=sw_dict,
            num_inputs=profile.num_inputs,
            neuron_type=f"{profile.neuron_type} (dictview)",
        )
