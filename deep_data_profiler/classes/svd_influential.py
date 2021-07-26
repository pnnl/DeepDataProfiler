from collections import defaultdict, OrderedDict
from typing import Optional, Tuple, List, Dict
import torch
import inspect
from torch.nn.modules import dropout
import networkx as nx
from deep_data_profiler.classes.profile import Profile
from deep_data_profiler.utils import (
    TorchHook,
    model_graph,
)


class SVDProfiler:

    """
    Torch Profiler wraps a PyTorch model into a TorchHook model which can
    register activations as it evaluates data. Using the activations,
    inputs to the model may be profiled.

    Attributes
    ----------
    dropout_classes : list
        List of dropout classes in PyTorch
    implemented_classes : list
        Set of classes in PyTorch which we can find influential for
    model : torch.nn.Sequential()
        Model to be profiled
    """

    def __init__(
        self,
        model: torch.nn.Sequential,
        device: torch.device = torch.device("cpu"),
        compute_svd: bool = True,
    ):

        super().__init__()
        self.dropout_classes = [
            m[1]
            for m in inspect.getmembers(dropout, inspect.isclass)
            if m[1].__module__ == "torch.nn.modules.dropout"
        ]
        self.implemented_classes = {
            torch.nn.Linear,
            torch.nn.Conv2d,
        }

        self.model = TorchHook(model)
        self.hooks = self.model.available_modules()
        supernodes, SG, pos = self.super_nodes_graph()
        self.supernodes = supernodes
        self.SG = SG
        self.pos = pos
        self.sght = self.sgheadtail()

        if compute_svd:
            self.svd_dict = self.create_svd()
        else:
            self.svd_dict = None

    def super_nodes_graph(self):
        model = self.model.model  # grab back the original model
        G, ndorder, pos = model_graph(model)

        # links the graph nodes to PyTorch modules
        pmas = self.model.available_modules()
        nodes = G.nodes

        # a layer starts with an implemented class, some of these correspond to
        # PyTorch modules, others were custom made
        # for the application, like 'add'
        def chk_type(n):
            try:
                if type(pmas[n]) in self.implemented_classes:
                    return True
            except Exception:
                return False

        # The implemented nodes are the first function to apply to input
        # into the DDPlayer,the successors which do not change the shape
        # are added to the DDPlayer.
        implemented_nodes = []
        for n in nodes:
            if chk_type(n):
                implemented_nodes.append(n)
        supernodes = OrderedDict({"x_in": ["x_in"]})

        for n in implemented_nodes:
            temp = [n]
            while True:
                scs = list(G.successors(temp[-1]))
                scs = [item for item in scs if "resnetadd" not in item]
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
            v: (k, v)
            for k, v in enumerate(sorted(list(SG.nodes()), key=sortorder))
        }
        nx.relabel_nodes(SG, ordG, copy=False)

        sgpos = dict()
        for nd in SG.nodes():
            sgpos[nd] = pos[nd[1]]

        # filter dropout modules
        for nd in SGnodes[1:]:
            supernodes[nd] = [
                n
                for n in supernodes[nd]
                if type(pmas[n]) not in self.dropout_classes
            ]

        return supernodes, SG, sgpos

    def sgheadtail(self):
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

    def create_svd(
        self, layers_to_find: Optional[List[int]] = None
    ) -> Dict[int, Tuple[str, torch.svd]]:
        """
        Create a dictionary of the Singular Value Decomposition
        of a layer's weights.
        Args:
            layers_to_find (list, optional): Optional list of layers
                to find influential SVD neurons for.

        Returns
        -------
        svd_dict : dict
            A dictionary of SVD of weights, keyed by their position from
            the beginning of the model, according to the order the layers
            of the model are listed.
        """
        # ordered dictionary to feed to influential profiler
        svd_dict = OrderedDict()

        SGnodes = sorted(self.SG.nodes, key=lambda nd: nd[0])
        if layers_to_find is None:
            ltf = range(1, len(SGnodes))
        elif isinstance(layers_to_find, list):
            ltf = [
                lyr
                for lyr in layers_to_find
                if lyr >= 1 or lyr <= len(SGnodes) - 1
            ]
        else:  # a tuple is expected
            start = max(1, layers_to_find[0])
            end = min(len(SGnodes), layers_to_find[1])
            ltf = range(start, end)

        for ndx in ltf:
            nd = SGnodes[ndx]
            layertype = type(self.model.available_modules()[nd[1]])

            if layertype in self.implemented_classes:
                # grab the weights for the layer
                X = self.hooks[nd[1]]._parameters["weight"].detach()
                # if a Cond2d layer, 'unfold'
                if layertype is torch.nn.Conv2d:
                    X = torch.flatten(X, start_dim=1, end_dim=-1)
                # take SVD and put into dict
                svd = torch.svd(X, compute_uv=True)
                svd_dict[nd[0]] = (self.supernodes[nd[1]], svd)
        return svd_dict

    def create_influential(
        self,
        x: torch.Tensor,
        layers_to_find: Optional[List[int]] = None,
        threshold: float = 0.1,
    ) -> Profile:
        """
        Generate an influential profile for a single input data x.

        Parameters
        ----------
        x : torch.Tensor
            input to model being profiled
        layers_to_find : list, optional
            Optional list of layers to find influential SVD neurons for.
        threshold : float, optional, default=0.1
            Percentage of contribution to track in a profile.

        Returns
        -------
        profile.Profile
            profile contains neuron_counts, neuron_weights across layers.
            Corresponding number of images = 1
        """
        # Create empty dictionaries for storing neurons
        neuron_counts = defaultdict(torch.Tensor)
        neuron_weights = defaultdict(torch.Tensor)

        with torch.no_grad():
            y, activations = self.model.forward(x)

            # dictionary of SVDs of the weights per layer,
            # if not already pre-computed when SVDInfluential was defined
            if not self.svd_dict:
                self.svd_dict = self.create_svd(layers_to_find=layers_to_find)

        for k, (layer_name, svd) in self.svd_dict.items():
            layer_name = layer_name[0]
            layer_activations = activations[layer_name].squeeze(
                0
            )  # noqa remove batch dimension
            layer_reshape = layer_activations.view(
                layer_activations.shape[0], -1
            )

            # get bias term, check it's not None
            bias = self.hooks[layer_name]._parameters["bias"]
            if bias is not None:
                layer_reshape = layer_reshape - bias.unsqueeze(1)

            # take SVD projection
            uprojy = torch.matmul(svd.U.T, layer_reshape)
            # average over the spatial dimensions
            agg = torch.sum(uprojy, axis=1) / uprojy.shape[1]  # noqa torch.max(uprojy, axis=1).values
            # calculate influential neurons
            (
                neuron_counts[k],
                neuron_weights[k],
            ) = SVDProfiler.influential_svd_neurons(agg, threshold=threshold)
        return Profile(
            neuron_counts=neuron_counts,
            neuron_weights=neuron_weights,
            num_inputs=1,
        )
 

    def create_projections(
        self,
        x: torch.Tensor,
        layers_to_find: Optional[List[int]] = None,
    ):
        """
        Generate SVD projections for a single input data x.

        Parameters
        ----------
        x : torch.Tensor
            input to model being profiled
        layers_to_find : list, optional
            Optional list of layers to find influential SVD neurons for.

        Returns
        -------
        projections: dict
            SVD projections keyed by layer
        """
        # Create empty dictionaries for storing neurons
        projections = defaultdict(torch.Tensor)

        with torch.no_grad():
            y, activations = self.model.forward(x)

            # dictionary of SVDs of the weights per layer,
            # if not already pre-computed when SVDInfluential was defined
            if not self.svd_dict:
                self.svd_dict = self.create_svd(layers_to_find=layers_to_find)

        for k, (layer_name, svd) in self.svd_dict.items():
            layer_name = layer_name[0]
            layer_activations = activations[layer_name].squeeze(
                0
            )  # noqa remove batch dimension
            layer_reshape = layer_activations.view(
                layer_activations.shape[0], -1
            )

            # get bias term, check it's not None
            bias = self.hooks[layer_name]._parameters["bias"]
            if bias is not None:
                layer_reshape = layer_reshape - bias.unsqueeze(1)

            # take SVD projection
            uprojy = torch.matmul(svd.U.T, layer_reshape)
            projections[k] = uprojy
            
        return projections

    @staticmethod
    def influential_svd_neurons(
        agg: torch.Tensor, threshold: float = 0.1, norm=1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a dictionary of relative contributions keyed by influential
        SVD neurons for layer up to some threshold
        Parameters
        ----------
        agg : torch.Tensor
            The SVD projections tensor, with some aggregation
            applied. Expected to be 1-D.

        ord (int, float, +/-inf, 'fro', 'nuc', optional)
            order of norm. See
            https://pytorch.org/docs/stable/linalg.html#torch.linalg.norm

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of influential_neurons, influential_weights.
            influential_neurons is a tensor of the same shape as the
            SVD of the weights (neurons).

        """
        m = torch.linalg.norm(
            agg.view((agg.shape[0], -1)), ord=norm, dim=1
        ).unsqueeze(0)

        # sort
        ordsmat_vals, ordsmat_indices = torch.sort(m, descending=True)

        # take the cumsum and normalize by total contribution per dim
        cumsum = torch.cumsum(ordsmat_vals, dim=1)
        totalsum = cumsum[:, -1].detach()

        # find the indices within the threshold goal, per dim
        bool_accept = (cumsum / totalsum.unsqueeze(-1)) <= threshold
        accept = torch.sum(bool_accept, dim=1)

        # normalize by final accepted cumsum
        ordsmat_vals /= cumsum[:, accept - 1]

        # add additional accept, ie accept + 1
        try:
            # use range to enumerate over batch size entries of accept
            bool_accept[range(len(accept)), accept] = True
        except IndexError:
            print("taking all values as influential")

        # find accepted synapses, all other values zero.
        # note: it is ordered by largest norm value
        unordered_weights = torch.where(
            bool_accept, ordsmat_vals, torch.zeros(ordsmat_vals.shape)
        )
        # re-order to mantain proper neuron ordering
        influential_weights = unordered_weights.gather(
            1, ordsmat_indices.argsort(1)
        )

        influential_neurons = influential_weights.bool().int()

        return influential_neurons, influential_weights
