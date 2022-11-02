from collections import OrderedDict
import torch
import inspect
from torch.nn.modules import activation, dropout, batchnorm
from typing import Dict, List, Optional, Tuple, Union


class TorchProfiler:

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
        List of contrib functions ordered by their corresponding classes in implemented_classes
    model : torch.nn.Sequential
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
        infl_threshold: float,
        contrib_threshold: float,
        layers_to_profile: Optional[Union[List[int], Tuple[int]]] = None,
        **kwargs,
    ) -> Tuple[dict, dict, dict, dict]:
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

        activation_shapes = {}
        for ldx, modules in self.layerdict.items():
            if "resnetadd" in modules[1]:
                activation_shapes[ldx] = activations[modules[0][1]].shape
            else:
                activation_shapes[ldx] = activations[modules[0][0]].shape

        # Create an x-dependent function to identify the the highest valued neurons at each layer
        influential_neurons = self.influence_generator(activations, **kwargs)

            ######## Step 1: Identify the Influential Neurons (Nodes) in the Activation Graph ########

        # Fill dictionaries
        for k in ltp:
            neuron_counts[k], neuron_weights[k] = influential_neurons(k, infl_threshold)

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
                    k: activations[self.sght[k][0]]
                }  # this is the output from the first available module in the layer
                nd = (k, layerdict[k][0][0])
                pred_layers = list(self.SG.predecessors(nd))
                x_in[k] = {
                    pd[0]: activations[self.sght[pd[0]][1]] for pd in pred_layers
                }  # this is the output from the last available module in the layer

            results = [
                self.single_profile(
                    x_in[ldx],
                    y_out[ldx],
                    neuron_counts[ldx],
                    layerdict,
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
