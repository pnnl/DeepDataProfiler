from deep_data_profiler.classes.profile import Profile
from deep_data_profiler.classes.torch_profiler import TorchProfiler
from deep_data_profiler.utils import get_index, submatrix_generator
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings


class ChannelProfiler(TorchProfiler):
    def influence_generator(
        self,
        activations: Dict[str, torch.Tensor],
        norm: Optional[int] = None,
    ) -> Callable[[int, float], Tuple[sp.coo_matrix, sp.coo_matrix]]:
        """
        Parameters
        ----------
        activations : dict of tensors
        norm : int, optional
            If given, the order of the norm to be taken to sum the strengths of channel
            activations. Otherwise, the max value of each channel is used

        Returns
        -------
        influential_neurons : function
            A function that will pick out the most influential neurons
            in a layer up to some threshold
        """

        def influential_neurons(
            layer_number: int, threshold: float
        ) -> Tuple[sp.coo_matrix, sp.coo_matrix]:
            """
            Parameters
            ----------
            layer_number : int
            threshold : float

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
            hd, nd = self.sght[layer_number]  # head (hd) and tail (nd) modules in layer
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

                    # check if module is a conv. layer
                    is_conv = len(t.shape) == 4

                    # define dimensions
                    if is_conv:
                        num_channels, h, w = t[0].shape
                    else:
                        num_elements = t[0].shape[0]

                    if norm is not None:
                        # for a conv layer with norm
                        if is_conv:
                            # take the matrix norm to represent each channel
                            m = torch.linalg.norm(t, ord=norm, dim=(2, 3))
                        # for a FC layer with norm
                        else:
                            # take the norm of each element
                            m = torch.linalg.norm(t, ord=norm, dim=0).unsqueeze(0)
                    else:
                        # for a conv layer without norm
                        if is_conv:
                            # take the max value to represent each channel
                            channel_vals = t.view(1, num_channels, h * w)
                            m = torch.max(channel_vals, dim=-1)[0]
                        # for a FC layer without norm
                        else:
                            # take the raw activations
                            m = t
                        # ignore negative elements when not using norm
                        m = torch.where(m > 0, m, torch.zeros(m.shape))

                    # sort by influence
                    ordsmat_vals, ordsmat_indices = torch.sort(m, descending=True)

                    # take the cumsum and normalize by total contribution
                    cumsum = torch.cumsum(ordsmat_vals, dim=1)
                    totalsum = cumsum[:, -1]

                    # find the indices within the threshold goal
                    bool_accept = cumsum / totalsum <= threshold
                    # find the number of accepted neurons
                    accept = bool_accept.sum()

                    # if accept == m.shape[1] (num_channels if conv., num_elements if FC),
                    # all values are taken as influential
                    # subtract 1 in this case to avoid IndexError when adding additional accept
                    if accept == m.shape[1]:
                        accept -= 1

                    # add additional accept, ie accept + 1
                    bool_accept[:, accept] = True

                    # normalize by final accepted cumsum
                    ordsmat_vals /= cumsum[:, accept]

                    # grab accepted neuron values and indices
                    ordsmat_vals = ordsmat_vals[bool_accept]
                    ordsmat_indices = ordsmat_indices[bool_accept]

                    # construct weights and counts sparse matrices
                    influential_weights = sp.coo_matrix(
                        (
                            ordsmat_vals,
                            (torch.zeros(ordsmat_indices.shape), ordsmat_indices),
                        ),
                        shape=m.shape,
                    )
                    influential_weights.eliminate_zeros()

                    influential_neurons = sp.coo_matrix(
                        (
                            np.ones(influential_weights.data.shape),
                            (influential_weights.row, influential_weights.col),
                        ),
                        shape=influential_weights.shape,
                        dtype=int,
                    )

                return influential_neurons, influential_weights
            else:
                return sp.coo_matrix((0, 0)), sp.coo_matrix((0, 0))

        return influential_neurons

    def create_profile(
        self,
        x: torch.Tensor,
        layers_to_profile: Union[list, Tuple] = None,
        threshold: float = 0.1,
        norm: Optional[int] = None,
    ) -> Profile:

        """
        Generate a profile for a single input data x

        Parameters
        ----------
        x : torch.Tensor
            input to model being profiled
        layers_to_profile : list or tuple
            list of specific layers to profile or tuple with first,last layers
            (exclusive of last) to profile and all layers inbetween
        threshold : float, default=0.1
            Percentage of contribution to track in a profile.
        norm : int, optional
            If given, the order of the norm to be taken to sum the strengths of channel
            activations. Otherwise, the max value of each channel is used

        Returns
        -------
        Profile
            profile contains neuron_counts, neuron_weights, synapse_counts, and synapse_weights
            across layers in layers_to_profile. Corresponding number of images = 1

        """

        (
            neuron_counts,
            neuron_weights,
            synapse_counts,
            synapse_weights,
        ) = self.build_dicts(
            x,
            layers_to_profile,
            infl_threshold=threshold,
            contrib_threshold=threshold,
            norm=norm,
        )

        return Profile(
            neuron_counts=neuron_counts,
            neuron_weights=neuron_weights,
            synapse_counts=synapse_counts,
            synapse_weights=synapse_weights,
            num_inputs=1,
            neuron_type="channel",
        )

    def single_profile(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        neuron_counts: sp.coo_matrix,
        ldx: int,
        threshold: float,
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
        ldx : int
            Layer number of the layer to be profiled
        threshold : float
            Percentage of contribution to track in a profile

        Returns
        -------
        neuron_counts : sp.coo_matrix or tuple of sp.coo_matrix
        synapse_counts : sp.coo_matrix
        synapse_weights : sp.coo_matrix

        """
        # get the appropriate contrib function for the module
        func = getattr(self.__class__, self.layerdict[ldx][1])
        # get list of influential indices
        infl_idx = torch.Tensor(neuron_counts.col).long()
        # call contrib function to return neuron counts and synapse counts/weights
        return func(
            self,
            x_in,
            y_out,
            infl_idx,
            self.layerdict[ldx][0],
            threshold=threshold,
        )

    def contrib_linear(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str],
        threshold: float = 0.1,
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
        threshold : float, default=0.1

        Returns
        -------
        neuron_counts : sp.coo_matrix
        synapse_counts : sp.coo_matrix
        synapse_weights : sp.coo_matrix
        """
        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_linear requires x_in and y_out to have exactly one layer key each"
            )

        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x_ldx = list(x_in.keys())[0]
        x_in = x_in[x_ldx]

        with torch.no_grad():
            # grab linear module
            linear = self.model.available_modules()[layer[0]]

            # grab weights and biases
            W = linear._parameters["weight"]
            B = linear._parameters["bias"]

            # check if x_in is conv layer
            conv_input = len(x_in[0].shape) > 1

            # define dimensions
            if conv_input:
                in_channels, h_in, w_in = x_in[0].shape
                # if x_in is conv layer, flatten
                x_in = x_in[0].flatten().unsqueeze(0)
            else:
                in_elements = x_in[0].shape[0]
            out_elements = y_out[0].shape[0]

            # get number of influential neurons
            num_infl = infl_neurons.shape[0]

            # multiply inputs by weights associated with influential neurons
            z = x_in[0] * W[infl_neurons]

            # ignore negative values
            z = torch.where(z > 0, z, torch.zeros(z.shape))

            # sort by contribution
            ordsmat_vals, ordsmat_indices = torch.sort(z, descending=True)

            # take the cumsum
            cumsum = torch.cumsum(ordsmat_vals, dim=1)

            # find the threshold goal
            goal = threshold * y_out[0, infl_neurons]
            # find the indices within the threshold goal
            if B is not None:
                bool_accept = cumsum + B[infl_neurons].unsqueeze(-1) <= goal.unsqueeze(
                    -1
                )
            else:
                bool_accept = cumsum <= goal.unsqueeze(-1)
            accept = torch.sum(bool_accept, dim=1)
            # if accept == x_in.shape[1], all values taken as contributors
            # subtract 1 in this case to avoid IndexError when adding additional accept
            accept = torch.where(accept < x_in.shape[1], accept, accept - 1)
            # add additional accept, ie accept + 1
            bool_accept[range(num_infl), accept] = True

            # normalize by total sum of all accepted contributions
            ordsmat_vals /= cumsum[range(num_infl), accept].unsqueeze(-1)

            # grab accepted contributor values and indices
            ordsmat_vals = ordsmat_vals[bool_accept]
            contrib_idx = ordsmat_indices[bool_accept]
            # repeat each influential neuron once for each of its accepted contributors
            infl_idx = np.repeat(infl_neurons, accept + 1)

            # define shape of synapse counts/weights
            if conv_input:
                # get channel index of each contributor
                contrib_idx = np.unravel_index(contrib_idx, (in_channels, h_in, w_in))[
                    0
                ]
                shape = (out_elements, in_channels)
            else:
                shape = (out_elements, in_elements)

            # construct synapse weights and counts
            synapse_weights = sp.coo_matrix(
                (ordsmat_vals, (infl_idx, contrib_idx)),
                shape=shape,
            )
            synapse_weights.eliminate_zeros()

            # sum contribution weight per channel if x_in is a conv layer
            if conv_input:
                synapse_weights.sum_duplicates()

            synapse_counts = sp.coo_matrix(
                (
                    np.ones(synapse_weights.data.shape),
                    (synapse_weights.row, synapse_weights.col),
                ),
                shape=shape,
                dtype=int,
            )

            # construct neuron counts by summing over columns of synapse counts
            neuron_counts = sp.coo_matrix(
                sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
            )

        return neuron_counts, synapse_counts, synapse_weights

    def contrib_conv2d(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str],
        threshold: float = 0.1,
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix]:
        """
        Profile a single output neuron from a 2D convolutional layer

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
            list containing keys in self.model.available_modules() dictionary,
            corresponding to a convolutional module and an activation module
        threshold : float, default=0.1

        Returns
        -------
        neuron_counts : sp.coo_matrix
        neuron_weights : sp.coo_matrix
        synapse_counts : sp.coo_matrix

        Note
        ----
        Only implemented for convolution using filters with same height and width
        and strides equal in both dimensions and padding equal in all dimensions
        """
        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_conv2d requires x_in and y_out to have exactly one layer key each"
            )

        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x_ldx = list(x_in.keys())[0]
        x_in = x_in[x_ldx]

        with torch.no_grad():
            # grab convolutional module
            conv = self.model.available_modules()[layer[0]]

            # grab weights and biases
            W = conv._parameters["weight"]
            B = conv._parameters["bias"]

            # assumption is that kernel size, stride are equal in both dimensions
            # and padding preserves input size
            kernel_size = conv.kernel_size[0]
            stride = conv.stride[0]
            padding = conv.padding[0]
            dilation = conv.dilation[0]

            # define dimensions
            in_channels, h_in, w_in = x_in[0].shape
            out_channels, h_out, w_out = y_out[0].shape

            # get number of influential neurons
            num_infl = infl_neurons.shape[0]

            # W dims are [#filters=out_channels, #channels=in_channels, kernel_size, kernel_size]
            W = W[infl_neurons]
            B = B[infl_neurons] if B is not None else torch.Tensor()

            # repeat x_in along batch dimension to pair with weights for each influential neuron
            x_stacked = x_in.repeat(num_infl, 1, 1, 1)
            # reshape to batch the convolution for all influential neurons
            # x batched dimensions: 1 x in_channels * num_infl x h_in x w_in
            # W batched dimensions: in_channels * num_infl x 1 x h_in x w_in
            x_batch = x_stacked.view(1, in_channels * num_infl, h_in, w_in)
            W_batch = W.view(
                in_channels * num_infl,
                1,
                kernel_size,
                kernel_size,
            )

            # take the depthwise convolution
            z = F.conv2d(
                x_batch,
                W_batch,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels * num_infl,
            )

            # reshape to correct dimensions: num_infl x in_channels x h_out x w_out
            z = z.view(num_infl, in_channels, h_out, w_out)

            # ignore negative values
            z = torch.where(z > 0, z, torch.zeros(z.shape))

            # find the max value per channel
            maxvals = F.max_pool2d(z, kernel_size=h_out).view(num_infl, in_channels)

            # order channels by greatest max value contribution
            ordsmat_vals, ordsmat_indices = torch.sort(maxvals, descending=True)
            # take the cumsum and normalize by the total sum to find the threshold goal
            cumsum = torch.cumsum(ordsmat_vals, dim=1)
            totalsum = cumsum[:, -1].unsqueeze(-1)
            goals = cumsum / totalsum

            # find the channels within the threshold goal
            bool_accept = goals <= threshold
            accept = torch.sum(bool_accept, dim=1)
            # if accept == in_channels, all values taken as contributors
            # subtract 1 in this case to avoid IndexError when adding additional accept
            accept = torch.where(accept < in_channels, accept, accept - 1)

            # add additional accept, ie accept + 1
            bool_accept[range(num_infl), accept] = True

            # normalize by total sum of all accepted contributions
            ordsmat_vals /= cumsum[range(num_infl), accept].unsqueeze(-1)

            # grab accepted contributor values and indices
            ordsmat_vals = ordsmat_vals[bool_accept]
            contrib_idx = ordsmat_indices[bool_accept]
            # repeat each influential neuron once for each of its accepted contributors
            infl_idx = np.repeat(infl_neurons, accept + 1)

            # construct synapse weights and counts
            synapse_weights = sp.coo_matrix(
                (ordsmat_vals, (infl_idx, contrib_idx)),
                shape=(out_channels, in_channels),
            )
            synapse_weights.eliminate_zeros()

            synapse_counts = sp.coo_matrix(
                (
                    np.ones(synapse_weights.data.shape),
                    (synapse_weights.row, synapse_weights.col),
                ),
                shape=(out_channels, in_channels),
                dtype=int,
            )

            # construct neuron counts by summing over columns of synapse counts
            neuron_counts = sp.coo_matrix(
                sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
            )

        return neuron_counts, synapse_counts, synapse_weights

    def contrib_max2d(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix]:
        """
        Draws synaptic connections between the given influential neurons in a 2D
        max pooling layer and their contributors in a previous layer

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
        layer : List[str], optional
            not used, placeholder for uniformity in arguments
        threshold : float, optional
            not used, placeholder for uniformity in arguments

        Returns
        -------
        neuron_counts : sp.coo_matrix
        synapse_counts : sp.coo_matrix
        synapse_weights : sp.coo_matrix
        """

        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_max2d requires x_in and y_out to have exactly one layer key each"
            )

        return self.contrib_identity(x_in, y_out, infl_neurons)

    def contrib_adaptive_avg_pool2d(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix]:
        """
        Draws synaptic connections between the given influential neurons in a 2D
        adaptive average pooling layer and their contributors in a previous layer

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
        layer : List[str], optional
            not used, placeholder for uniformity in arguments
        threshold : float, optional
            not used, placeholder for uniformity in arguments

        Returns
        -------
        neuron_counts : sp.coo_matrix
        synapse_counts : sp.coo_matrix
        synapse_weights : sp.coo_matrix
        """

        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_adaptive_avg_pool2d requires x_in and y_out to have exactly "
                + "one layer key each"
            )

        return self.contrib_identity(x_in, y_out, infl_neurons)

    def contrib_resnetadd(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[Tuple[sp.coo_matrix, sp.coo_matrix], sp.coo_matrix, sp.coo_matrix]:
        """
        Draws synaptic connections between the given influential neurons in a ResNet add
        layer and their contributors in a previous layer
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
        layer : List[str], optional
            not used, placeholder for uniformity in arguments
        threshold : float, optional
            not used, placeholder for uniformity in arguments

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
                "contrib_resnetadd requires y_out to have exactly one layer key and "
                + "x_in to have exactly two layer keys"
            )

        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x1_ldx, x2_ldx = sorted(x_in.keys())

        # define dimensions
        num_channels, h, w = y_out[0].shape

        with torch.no_grad():
            # combine activations from input layers along new dimension
            x_in = torch.cat((x_in[x1_ldx], x_in[x2_ldx]))
            # find the max val from each potential contributor channel in both input layers
            maxvals = x_in.view(2, num_channels, h * w).max(dim=-1)[0][:, infl_neurons]

        # construct neuron counts and synapse counts/weights for each input layer
        # when the max channel val in the first input layer is greater than the second,
        # fully attribute influence to the corresponding channel in the first layer
        nc1, sc1, sw1 = self.contrib_identity(
            {x1_ldx: x_in[0].unsqueeze(0)},
            {y_ldx: y_out},
            infl_neurons[maxvals[0] > maxvals[1]],
        )

        # when the max channel val in the second input layer is greater than or equal to
        # the first, fully attribute influence to the corresponding channel in the second layer
        nc2, sc2, sw2 = self.contrib_identity(
            {x2_ldx: x_in[1].unsqueeze(0)},
            {y_ldx: y_out},
            infl_neurons[maxvals[0] <= maxvals[1]],
        )

        # return neuron counts as tuple, synapse counts/weights as block diagonal matrices
        neuron_counts = (nc1, nc2)
        synapse_counts = sp.block_diag((sc1, sc2))
        synapse_weights = sp.block_diag((sw1, sw2))

        return neuron_counts, synapse_counts, synapse_weights

    def contrib_identity(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix]:
        """
        Pass through to keep influential neurons from one layer fixed into the
        next.

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
        layer : List[str], optional
            not used, placeholder for uniformity in arguments
        threshold : float, optional
            not used, placeholder for uniformity in arguments

        Returns
        -------
        neuron_counts
        synapse_counts
        synapse_weights
        """
        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_identity requires x_in and y_out to have exactly one layer key each"
            )
        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x_ldx = list(x_in.keys())[0]
        x_in = x_in[x_ldx]

        if y_out[0].shape[0] != x_in[0].shape[0]:
            raise NotImplementedError(
                "contrib_identity requires x_in and y_out to have the same number of channels"
            )

        with torch.no_grad():
            # define dimensions
            num_channels = y_out[0].shape[0]

            # get number of influential neurons
            num_infl = infl_neurons.shape[0]

        # construct synapse_weights and counts
        synapse_weights = sp.coo_matrix(
            (np.ones(num_infl), (infl_neurons, infl_neurons)),
            shape=(num_channels, num_channels),
        )

        synapse_counts = sp.coo_matrix(
            (np.ones(num_infl),(infl_neurons, infl_neurons)),
            shape=(num_channels, num_channels),
            dtype=int,
        )

        # construct neuron counts by summing along columns of synapse counts
        neuron_counts = sp.coo_matrix(
            sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
        )

        return neuron_counts, synapse_counts, synapse_weights
