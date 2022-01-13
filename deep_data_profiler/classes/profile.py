import scipy.sparse as sp
import copy
import torch
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np

SpMatData = Dict[int, sp.spmatrix]
DictData = Dict[int, Dict[Tuple, float]]


class Profile:

    """Summary"""

    def __init__(
        self,
        neuron_counts: Optional[Union[SpMatData, DictData]] = None,
        neuron_weights: Optional[Union[SpMatData, DictData]] = None,
        synapse_counts: Optional[Union[SpMatData, DictData]] = None,
        synapse_weights: Optional[Union[SpMatData, DictData]] = None,
        activation_shapes: Optional[Union[SpMatData, DictData]] = None,
        pred_dict: Optional[Dict[int, List[int]]] = None,
        num_inputs: Optional[int] = 0,
        neuron_type: Optional[str] = None,
    ) -> None:
        """Summary

        Parameters
        ----------
        neuron_counts : dict, optional
            Dictionary representing profile neurons and their counts,
            i.e. how many synapses they were influential or contributing to
        neuron_weights : dict, optional
            Dictionary representing influential profile neurons and their weights
        synapse_counts : dict, optional
            Dictionary representing profile synapses and their counts
        synapse_weights : dict, optional
            Dictionary representing profile synapses and their weights
        activation_shapes : dict, optional
            Dictionary of the activation tensors shapes, keyed by layer
        pred_dict : dict, optional
            Dictionary giving the layer predecessor hierarchy by layer index
        num_inputs : int, optional
            Number of inputs represented by the profile
        neuron_type: str, optional
            The type of neurons used in the profile, i.e. 'element' or 'channel'

        Note
        ----
        The format for the inputs is very strict so that it can be used to store
        the results of a profiling process but there is no type checking. If the
        input is not in the correct format, the metrics could fail or return inaccurate
        values.
        """

        self._neuron_counts = neuron_counts or dict()
        self._neuron_weights = neuron_weights or dict()
        self._synapse_counts = synapse_counts or dict()
        self._synapse_weights = synapse_weights or dict()
        self._activation_shapes = activation_shapes
        self._pred_dict = pred_dict
        self._num_inputs = num_inputs
        self._neuron_type = neuron_type

    @property
    def neuron_counts(self) -> Union[SpMatData, DictData]:
        """
        Returns
        -------
        neuron_counts : Dict of scipy.sparse matrices or Dict of dicts
            Dictionary representing profile neurons and their counts,
            i.e. how many synapses they were influential or contributing to
        """
        return self._neuron_counts

    @property
    def neuron_weights(self) -> Union[SpMatData, DictData]:
        """
        Returns
        -------
        neuron_weights : Dict of scipy.sparse matrices or Dict of dicts
            Dictionary representing influential profile neurons and their weights
        """
        return self._neuron_weights

    @property
    def synapse_counts(self) -> Union[SpMatData, DictData]:
        """
        Returns
        -------
        synapse_counts : Dict of scipy.sparse matrices or Dict of dicts
            Dictionary representing profile synapses and their counts

        Note
        ----
        For a single image profile (num_inputs=1) all synapses should have a count of 1
        """
        return self._synapse_counts

    @property
    def synapse_weights(self) -> Union[SpMatData, DictData]:
        """
        Returns
        -------
        synapse_weights : Dict of scipy.sparse matrices or Dict of dicts
            Dictionary representing profile synapses and their weights
        """
        return self._synapse_weights

    @property
    def activation_shapes(self) -> Dict[int, torch.Size]:
        """
        Returns
        -------
        activation_shapes: Dict of torch.Sizes
            Dictionary of the activation tensors shapes, keyed by layer
        """
        return self._activation_shapes

    @property
    def pred_dict(self) -> Dict[int, List[int]]:
        """
        Returns
        -------
        pred_dict: Dict of list of ints
            Dictionary giving the layer predecessor hierarchy by layer index
        """
        return self._pred_dict

    @property
    def num_inputs(self) -> int:
        """
        Returns
        -------
        num_inputs : int
            The number of input images represented by the profile

        Note
        ----
        Class profiles and other aggregate profiles will have num_inputs > 1
        """
        return self._num_inputs

    @property
    def neuron_type(self) -> str:
        """
        Returns
        -------
        neuron_type : str
            The type of neurons used in the profile,
            i.e. 'element', 'channel', or 'mixed' (aggregate of profiles with mismatched types)
        """
        return self._neuron_type

    @property
    def total(self) -> int:
        """
        Returns
        -------
        int
            Total sum of neuron counts across all layers
        """
        return sum([self._neuron_counts[layer].sum() for layer in self._neuron_counts])

    @property
    def size(self) -> int:
        """
        Returns
        -------
        int
            Total number of neurons identified as influential or contributing
            (neurons with nonzero neuron counts)
        """
        return sum(
            [self.neuron_counts[layer].count_nonzero() for layer in self.neuron_counts]
        )

    @property
    def num_synapses(self) -> int:
        """
        Returns
        -------
        int
            Total number of synapses across all layers
        """
        return sum(
            [
                self.synapse_counts[layer].count_nonzero()
                for layer in self.synapse_counts
            ]
        )

    def __eq__(self, other: "Profile") -> bool:
        """
        Parameters
        ----------
        other : Profile

        Returns
        -------
        bool
            True if the profile data held by self is equal to the profile data held by other,
            otherwise False

        Note
        ----
        If neuron type is specified by one profile but not the other, the two can still be equal
        if all other data is equal

        """
        return bool(
            self._neuron_counts == other.neuron_counts
            and self._neuron_weights == other.neuron_weights
            and self._synapse_counts == other.synapse_counts
            and self._synapse_weights == other.synapse_weights
            and self._num_inputs == other.num_inputs
            and (
                self._neuron_type == other.neuron_type
                or ((self._neuron_type is None) ^ (other.neuron_type is None))
            )
        )

    def __iter__(self) -> Iterable:
        """
        Returns
        -------
        Iterable
            An iterable over the layer keys of the neuron counts
        """
        return self.neuron_counts.keys()

    def __add__(self, other: "Profile") -> "Profile":
        """
        Adds the neuron and synapse counts and weights of other and self.

        Parameters
        ----------
        other : Profile

        Returns
        -------
        new_profile : Profile
            The aggregate profile of self and other

        Note
        ----
        Not supported for dictionary-formatted profiles

        """
        with torch.no_grad():
            if self.num_inputs == 0:
                new_profile = Profile(profile=other)
            else:
                new_profile = Profile(profile=self)
                if other.num_inputs > 0:
                    new_profile += other

        return new_profile

    def __iadd__(self, other: "Profile") -> "Profile":
        """
        Adds in place the neuron and synapse counts and weights of other to self.

        Parameters
        ----------
        other : Profile

        Returns
        -------
        self : Profile

        Note
        ----
        Not supported for dictionary-formatted profiles

        """
        with torch.no_grad():
            if self._num_inputs == 0:
                self._neuron_counts = copy.deepcopy(other.neuron_counts)
                self._neuron_weights = copy.deepcopy(other.neuron_weights)
                self._synapse_counts = copy.deepcopy(other.synapse_counts)
                self._synapse_weights = copy.deepcopy(other.synapse_weights)
                self._num_inputs = other.num_inputs
                self._neuron_type = other.neuron_type
            elif other.num_inputs > 0:
                self._num_inputs += other.num_inputs
                if (
                    self.neuron_type is not None
                    and other.neuron_type is not None
                    and self.neuron_type != other.neuron_type
                ):
                    self._neuron_type = "mixed"
                    warnings.warn(
                        "Incompatible profiles: mismatched neuron types - output neuron type will be mixed"
                    )

                if (
                    self.neuron_counts.keys() != other.neuron_counts.keys()
                    or self.neuron_weights.keys() != other.neuron_weights.keys()
                    or self.synapse_counts.keys() != other.synapse_counts.keys()
                    or self.synapse_weights.keys() != other.synapse_weights.keys()
                ):
                    warnings.warn(
                        "Warning: These profiles have mismatched layer key sets, aggregation may give strange results"
                    )

                # Combine neuron counts and weights
                for layer in other.neuron_counts:
                    if layer in self.neuron_counts:
                        if (
                            layer in self.neuron_weights
                            and layer in other.neuron_weights
                        ):
                            # use compressed sparse row (csr) matrix for indexing and fast arithmetic
                            # sum of neuron weights (weighted by counts)
                            neuron_weights = (
                                self.neuron_weights[layer].multiply(
                                    self.neuron_counts[layer]
                                )
                                + other.neuron_weights[layer].multiply(
                                    other.neuron_counts[layer]
                                )
                            ).tocsr()
                            # aggregate total neuron counts
                            self._neuron_counts[layer] += other.neuron_counts[layer]
                            # dictionary of keys (dok) matrix lets us index by nonzero values (avoid 0/0)
                            self._neuron_weights[layer] = sp.dok_matrix(
                                neuron_weights.shape
                            )
                            # normalize by total neuron counts
                            self._neuron_weights[layer][neuron_weights.nonzero()] = (
                                neuron_weights[neuron_weights.nonzero()]
                                / self.neuron_counts[layer].tocsr()[
                                    neuron_weights.nonzero()
                                ]
                            )
                            # convert back to COOrdinate matrix
                            self._neuron_weights[layer] = self.neuron_weights[
                                layer
                            ].tocoo()
                        else:
                            # copy neuron weights if keyset is mismatched
                            if layer in other.neuron_weights:
                                self._neuron_weights[layer] = copy.deepcopy(
                                    other.neuron_weights[layer]
                                )
                            # aggregate total neuron counts
                            self._neuron_counts[layer] += other.neuron_counts[layer]

                    else:
                        # mismatched keysets, copy counts and weights if necessary
                        if layer in other.neuron_weights:
                            self._neuron_weights[layer] = copy.deepcopy(
                                other.neuron_weights[layer]
                            )
                        self._neuron_counts[layer] = copy.deepcopy(
                            other.neuron_counts[layer]
                        )

                # Combine synapse counts and weights
                for layer in other.synapse_counts:
                    if layer in self.synapse_counts:
                        if (
                            layer in self.synapse_weights
                            and layer in other.synapse_weights
                        ):
                            synapse_weights = (
                                self.synapse_weights[layer].multiply(
                                    self.synapse_counts[layer]
                                )
                                + other.synapse_weights[layer].multiply(
                                    other.synapse_counts[layer]
                                )
                            ).tocsr()
                            # aggregate total synapse counts
                            self._synapse_counts[layer] += other.synapse_counts[layer]
                            # dictionary of keys (dok) matrix lets us index by nonzero values (avoid 0/0)
                            self._synapse_weights[layer] = sp.dok_matrix(
                                synapse_weights.shape
                            )
                            # normalize by total synapse counts
                            self._synapse_weights[layer][synapse_weights.nonzero()] = (
                                synapse_weights[synapse_weights.nonzero()]
                                / self.synapse_counts[layer].tocsr()[
                                    synapse_weights.nonzero()
                                ]
                            )
                            # convert back to COOrdinate matrix
                            self._synapse_weights[layer] = self.synapse_weights[
                                layer
                            ].tocoo()
                        else:
                            # copy synapse weights if keyset is mismatched
                            if layer in other.synapse_weights:
                                self._synapse_weights[layer] = copy.deepcopy(
                                    other.synapse_weights[layer]
                                )
                            # aggregate total synapse counts
                            self._synapse_counts[layer] += other.synapse_counts
                    else:
                        # mismatched keysets, copy counts and weights if necessary
                        if layer in other.synapse_weights:
                            self._synapse_weights[layer] = copy.deepcopy(
                                other.synapse_weights[layer]
                            )
                        self._synapse_counts[layer] = copy.deepcopy(
                            other.synapse_counts[layer]
                        )

        return self

    def dict_view(self) -> "Profile":
        """
        Returns
        -------
        A copy of the profile with neuron and synapse counts and weights
        reformatted as dicts

        """

        if self._activation_shapes is None or self._pred_dict is None:
            warnings.warn(
                "activation_shapes and pred_dict must be known to generate dictionary view"
            )
            return None

        # construct dicts for neuron counts and weights
        neuron_stats = {"counts": self._neuron_counts, "weights": self._neuron_weights}
        neuron_views = {stat: dict() for stat in neuron_stats}
        for stat in neuron_stats:
            neuron_spmat = neuron_stats[stat]
            neuron_dict = neuron_views[stat]

            for layer, spmat in neuron_spmat.items():
                dims = self._activation_shapes[layer]
                # list neurons and values from sparse matrix
                neurons = list(spmat.todok().items())
                flat_idx, values = zip(*neurons)
                neuron_idx = tuple(idx[1] for idx in flat_idx)

                # convert flat indices to full spatial or element indices if necessary
                if len(dims) == 4 and self._neuron_type != "channel":
                    spatial_idx = np.unravel_index(neuron_idx, dims[2:])
                    if self._neuron_type == "element":
                        channel_idx = np.array([int(idx[0]) for idx in flat_idx])
                        neuron_idx = (channel_idx,) + spatial_idx
                    elif self._neuron_type == "spatial":
                        neuron_idx = spatial_idx
                    neuron_idx = tuple(zip(*neuron_idx))
                else:
                    neuron_idx = tuple(zip(neuron_idx))

                # add each neuron to the dict with its value
                # neuron format: (layer index, (full neuron index,))
                full_idx = tuple((layer, idx) for idx in neuron_idx)
                neuron_dict[layer] = {idx: val for idx, val in zip(full_idx, values)}

        # construct dicts for synapse counts and weights
        synapse_stats = {
            "counts": self._synapse_counts,
            "weights": self._synapse_weights,
        }
        synapse_views = {stat: dict() for stat in synapse_stats}
        for stat in synapse_stats:
            synapse_spmat = synapse_stats[stat]
            synapse_dict = synapse_views[stat]
            for layer in synapse_spmat:
                dims = self._activation_shapes[layer]

                # list predecessors/input layers
                pred_list = self._pred_dict[layer]
                # split sparse matrix if there are two input layers (i.e. resnetadd)
                if len(pred_list) == 2:
                    pred_spmats = (
                        synapse_spmat[layer].tocsr()[: dims[1], : dims[1]],
                        synapse_spmat[layer].tocsr()[dims[1] :, dims[1] :],
                    )
                else:
                    pred_spmats = (synapse_spmat[layer],)

                synapse_dict[layer] = dict()

                # add synapses from each input layer to the dict
                for pred, spmat in zip(pred_list, pred_spmats):
                    pdims = self._activation_shapes[pred]
                    # list input neurons, output neurons, and values from sparse matrix
                    synapses = [(*syn, val) for syn, val in spmat.todok().items()]
                    out_idx, in_idx, values = zip(*synapses)
                    full_idxs = []

                    # convert flat indices of input and output neurons to full spatial
                    # or element indices if necessary
                    for ldx, ldims, neuron_idx in (
                        (pred, pdims, in_idx),
                        (layer, dims, out_idx),
                    ):
                        if len(ldims) == 4 and self._neuron_type != "channel":
                            if self._neuron_type == "element":
                                neuron_idx = np.unravel_index(neuron_idx, ldims[1:])
                            elif self._neuron_type == "spatial":
                                neuron_idx = np.unravel_index(neuron_idx, ldims[2:])
                            neuron_idx = tuple(zip(*neuron_idx))
                        else:
                            neuron_idx = tuple(zip(neuron_idx))

                        full_idxs += [tuple((ldx, idx) for idx in neuron_idx)]
                    in_idx, out_idx = full_idxs

                    # add each synapse to the dict with its value
                    # synapse format: ((input layer index, (input neuron index,)),
                    #                  (output layer index, (output neuron index,)))
                    synapse_dict[layer].update(
                        {(i, o): v for i, o, v in zip(in_idx, out_idx, values)}
                    )

        return Profile(
            neuron_counts=neuron_views["counts"],
            neuron_weights=neuron_views["weights"],
            synapse_counts=synapse_views["counts"],
            synapse_weights=synapse_views["weights"],
            activation_shapes=self._activation_shapes,
            pred_dict=self._pred_dict,
            num_inputs=self._num_inputs,
            neuron_type=f"{self._neuron_type} (dict)",
        )
