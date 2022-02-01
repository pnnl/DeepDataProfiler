from collections import defaultdict, OrderedDict
import torch
from deep_data_profiler.classes.torch_profiler import TorchProfiler
from deep_data_profiler.utils import matrix_convert
from deep_data_profiler.classes.profile import Profile
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple


class SVDProfiler(TorchProfiler):

    """
    Torch Profiler wraps a PyTorch model into a TorchHook model which can
    register activations as it evaluates data. Using the activations,
    inputs to the model may be profiled.

    The function call to generate an influenial SVD profile is slightly different
    than that for SpatialProfiler/ChannelProfiler. Here is how to create a profile:
    .. highlight:: python
    .. code-block:: python
        import deep_data_profiler as ddp
        # define the profiler
        influential_profiler = ddp.SVDProfiler(model)
        # profile a tensor x
        profile = influential_profiler.create_influential(x)
        # view neuron weights dictionary
        print(profile.neuron_weights)
        # view the neuron weights for a specific layer
        print(profile.neuron_weights[22].todense())
        ...
    Attributes
    ----------
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

        super().__init__(model)
        self.device = device
        self.implemented_classes = [
            torch.nn.Linear,
            torch.nn.Conv2d,
        ]

        if compute_svd:
            self.svd_dict = self.create_svd()
        else:
            self.svd_dict = None

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
            ltf = [lyr for lyr in layers_to_find if lyr >= 1 or lyr <= len(SGnodes) - 1]
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
        activations=None,
        aggregation="sum",
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
            if not activations:
                y, activations = self.model.forward(x)
            else:
                activations = activations

            # dictionary of SVDs of the weights per layer,
            # if not already pre-computed when SVDInfluential was defined
            if not self.svd_dict:
                self.svd_dict = self.create_svd(layers_to_find=layers_to_find)

            for k, (layer_name, svd) in self.svd_dict.items():
                layer_name = layer_name[0]
                layer_activations = activations[layer_name].squeeze(
                    0
                )  # noqa remove batch dimension
                layer_reshape = layer_activations.view(layer_activations.shape[0], -1)

                # get bias term, check it's not None
                bias = self.hooks[layer_name]._parameters["bias"]
                if bias is not None:
                    layer_reshape = layer_reshape - bias.unsqueeze(1)

                # take SVD projection
                uprojy = torch.matmul(
                    svd.U.T.to(self.device), layer_reshape.to(self.device)
                )
                # average over the spatial dimensions
                if aggregation == "sum":
                    agg = (
                        torch.sum(uprojy, axis=1) / uprojy.shape[1]
                    )  # noqa torch.max(uprojy, axis=1).values
                    # calculate influential neurons
                    (
                        neuron_counts[k],
                        neuron_weights[k],
                    ) = SVDProfiler.influential_svd_neurons(
                        agg, threshold=threshold, device=self.device
                    )
                elif aggregation == "max":
                    agg = torch.max(uprojy, dim=1).values
                    (neuron_counts[k], neuron_weights[k],) = (
                        matrix_convert(torch.ones(agg.shape)),
                        matrix_convert(agg),
                    )
                elif aggregation == "min":
                    agg = torch.min(uprojy, dim=1).values
                    (neuron_counts[k], neuron_weights[k],) = (
                        matrix_convert(torch.ones(agg.shape)),
                        matrix_convert(agg),
                    )
                else:
                    raise NotImplementedError(f"Do not recognize aggregation {agg}")
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
            layer_reshape = layer_activations.view(layer_activations.shape[0], -1)

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
        agg: torch.Tensor, threshold: float = 0.1, norm=1, device=torch.device("cpu")
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
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
        influential_neurons : sp.coo_matrix
            Matrix representing the influential neurons within the threshold
        influential_weights : sp.coo_matrix
            Matrix assigning weights to each influential neuron according to its
            contribution to the threshold
        """
        m = torch.linalg.norm(agg.view((agg.shape[0], -1)), ord=norm, dim=1).unsqueeze(
            0
        )

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
            bool_accept, ordsmat_vals, torch.zeros(ordsmat_vals.shape, device=device)
        )
        # re-order to mantain proper neuron ordering
        influential_weights = unordered_weights.gather(1, ordsmat_indices.argsort(1))

        influential_neurons = influential_weights.bool().int()

        return matrix_convert(influential_neurons), matrix_convert(influential_weights)

    # final three methods are defined so the method plays nicely
    # with the newest ddp version
    def influence_generator(self):
        pass

    def single_profile(self):
        pass

    def create_profile(self):
        pass
