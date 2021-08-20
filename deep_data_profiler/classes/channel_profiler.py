from deep_data_profiler.classes.profile import Profile
from deep_data_profiler.classes.torch_profiler import TorchProfiler
from deep_data_profiler.utils import get_index, matrix_convert, submatrix_generator
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Tuple, Union
import warnings


class ChannelProfiler(TorchProfiler):
    def influence_generator(
        self,
        activations: Dict[str, torch.Tensor],
        norm: int = None,
    ) -> Callable[[int, float], Tuple[sp.coo_matrix, sp.coo_matrix]]:
        """
        Parameters
        ----------
        activations : dict of tensors
        norm : int, optional, default=None
            Specify norm=1 or norm=2 to select influential neurons by L1- or L2-norm, respectively.
            Defaults to select influential neurons by max. values

        Returns
        -------
        influential_neurons : function
            A function that will pick out the most influential neurons
            in a layer up to some threshold
        """

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

                    if norm is not None:
                        if len(t.shape) == 4:
                            # we have a 3-tensor so take the matrix norm
                            m = torch.linalg.norm(t, ord=norm, dim=(2, 3))
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

                    # if accept == m.shape[1], all values taken as influential
                    # subtract 1 in this case to avoid IndexError when adding additional accept
                    accept = torch.where(accept < m.shape[1], accept, accept - 1)

                    # normalize by final accepted cumsum
                    ordsmat_vals /= cumsum[:, accept]

                    # add additional accept, ie accept + 1
                    # use range to enumerate over batch size entries of accept
                    bool_accept[range(len(accept)), accept] = True

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

                    influential_neurons = influential_weights.bool().int()

                return (
                    matrix_convert(influential_neurons),
                    matrix_convert(influential_weights),
                )
            else:
                return sp.coo_matrix((0, 0)), sp.coo_matrix((0, 0))

        return influential_neurons

    def create_profile(
        self,
        x: torch.Tensor,
        layers_to_profile: Union[list, Tuple] = None,
        threshold: float = 0.1,
        norm: int = None,
    ) -> Profile:

        """
        Generate a profile for a single input data x

        Parameters
        ----------
        x : torch.Tensor
            input to model being profiled\
        layers_to_profile : list or tuple
            list of specific layers to profile or tuple with first,last layers
            (exclusive of last) to profile and all layers inbetween
        threshold : float, optional, default=0.1
            Percentage of contribution to track in a profile.
        norm : int, optional, default=None
            Specify norm=1 or norm=2 to select influential neurons by L1- or L2-norm, respectively.
            Defaults to select influential neurons by max. values

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

        func = getattr(self.__class__, self.layerdict[ldx][1])
        # get list of influential indices
        infl_idx = torch.Tensor(neuron_counts.col).long()
        # return ncs, scs, sws
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
        threshold : float

        Returns
        -------
        neuron_counts : sp.coo_matrix
        synapse_counts : sp.coo_matrix
        synapse_weights : sp.coo_matrix
        """
        # use the same profiling for channels or neurons for linear
        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_linear cannot handle more than one dict key for x_in or y_out"
            )
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

            if len(xdims) > 1:
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
            # if accept == x_in.shape[1], all values taken as contributors
            # subtract 1 in this case to avoid IndexError when adding additional accept
            accept = torch.where(accept < x_in.shape[1], accept, accept - 1)

            # normalize
            ordsmat_vals /= cumsum[range(len(accept)), accept].unsqueeze(-1)
            # add additional accept, ie accept + 1
            bool_accept[range(len(accept)), accept] = True

            # find accepted synapses, all other values zero.
            # note: ordered by largest norm value, but "unordered" by neuron
            unordered_synapses = torch.where(
                bool_accept, ordsmat_vals, torch.zeros(ordsmat_vals.shape)
            )
            # re-order to mantain proper neuron ordering
            sws_compact = unordered_synapses.gather(1, ordsmat_indices.argsort(1))

            # sum contribution per channel if x_in is a conv layer
            if len(xdims) > 1:
                sws_compact = sws_compact.view(len(j), xdims[0], -1)
                sws_compact = torch.sum(sws_compact, dim=-1)

            # fullify synapse weights to outdims x indims
            synapse_weights = torch.zeros(ydims[0], xdims[0])
            synapse_weights[j] = sws_compact

            synapse_counts = synapse_weights.bool().int()

            # construct neuron counts
            neuron_counts = torch.sum(synapse_counts, dim=0).unsqueeze(0).int()

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
                "contrib_conv2d cannot handle more than one dict key for x_in or y_out"
            )
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

            ch = infl_neurons

            # this is not [:,ch] because W dims are [#filters=outdim, #channels = indim, ker_size,ker_size] - don't erase this
            W = W[ch]
            B = B[ch] if B is not None else torch.Tensor()

            y_true = y_out[0, ch, :, :].detach()

            # reshape to batch convolution for num_influential neurons
            W2 = W.view(
                num_infl * in_channels,
                kernel_size,
                kernel_size,
            )
            x_in_stacked = x_in.repeat(num_infl, 1, 1, 1)
            # take the depthwise convolution .unsqueeze(1)
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

        return (
            matrix_convert(neuron_counts),
            matrix_convert(synapse_counts),
            matrix_convert(synapse_weights),
        )

    def contrib_max2d(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str] = None,
        threshold: float = None,
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
        layer : None or List[str]
            not used, placeholder for uniformity in arguments
        threshold : None or float
            not used, placeholder for uniformity in arguments

        Returns
        -------
        neuron_counts : sp.coo_matrix
        synapse_counts : sp.coo_matrix
        synapse_weights : sp.coo_matrix
        """

        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_max2d cannot handle more than one dict key for x_in or y_out"
            )

        return self.contrib_identity(x_in, y_out, infl_neurons)

    def contrib_adaptive_avg_pool2d(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str] = None,
        threshold: float = None,
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
        layer : None or List[str]
            not used, placeholder for uniformity in arguments
        threshold : None or float
            not used, placeholder for uniformity in arguments

        Returns
        -------
        neuron_counts : sp.coo_matrix
        synapse_counts : sp.coo_matrix
        synapse_weights : sp.coo_matrix
        """
        # grab tensors from dictionaries
        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_adaptive_avg_pool2d cannot handle more than one dict key for "
                + "x_in or y_out"
            )

        return self.contrib_identity(x_in, y_out, infl_neurons)

    def contrib_resnetadd(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str] = None,
        threshold: float = None,
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

        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        dims = y_out.shape

        x1_ldx, x2_ldx = sorted(x_in.keys())
        x1_in = x_in[x1_ldx]
        x2_in = x_in[x2_ldx]

        with torch.no_grad():
            maxvals = torch.zeros(2, dims[1])
            for i, x in enumerate((x1_in, x2_in)):
                channel_vals = x.view(x.shape[:2] + (-1,))
                maxvals[i] = torch.max(channel_vals, dim=-1)[0]
            maxvals = maxvals[:, infl_neurons]

            ch_x1 = infl_neurons[maxvals[0] > maxvals[1]]
            ch_x2 = infl_neurons[maxvals[0] <= maxvals[1]]

            for i, ch in enumerate((ch_x1, ch_x2)):
                neuron_counts[i] = torch.zeros(dims[:2], dtype=torch.int)
                synapse_counts[i] = torch.zeros(dims[1], dims[1], dtype=torch.int)
                synapse_weights[i] = torch.zeros(dims[1], dims[1])

                neuron_counts[i][:, ch] = 1
                synapse_counts[i][ch, ch] = 1
                synapse_weights[i][ch, ch] = 1

        neuron_counts = tuple(matrix_convert(neuron_counts[i]) for i in range(2))
        synapse_counts = sp.block_diag(
            [matrix_convert(synapse_counts[i]) for i in range(2)]
        )
        synapse_weights = sp.block_diag(
            [matrix_convert(synapse_weights[i]) for i in range(2)]
        )

        return neuron_counts, synapse_counts, synapse_weights

    def contrib_identity(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str] = None,
        threshold: float = None,
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
        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x_ldx = list(x_in.keys())[0]
        x_in = x_in[x_ldx]

        xdims = x_in.shape
        ydims = y_out.shape

        ch = infl_neurons
        neuron_counts = torch.zeros(xdims[:2], dtype=torch.int)
        synapse_counts = torch.zeros(ydims[1], xdims[1], dtype=torch.int)
        synapse_weights = torch.zeros(ydims[1], xdims[1])

        neuron_counts[:, ch] += 1
        synapse_counts[ch, ch] += 1
        synapse_weights[ch, ch] = 1

        return (
            matrix_convert(neuron_counts),
            matrix_convert(synapse_counts),
            matrix_convert(synapse_weights),
        )
