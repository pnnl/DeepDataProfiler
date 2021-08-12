from deep_data_profiler.classes.profile import Profile
from deep_data_profiler.classes.torch_profiler import TorchProfiler
from deep_data_profiler.utils import get_index, matrix_convert, submatrix_generator
import numpy as np
import scipy.sparse as sp
import torch
from typing import Callable, Dict, List, Tuple, Union
import warnings


class ElementProfiler(TorchProfiler):
    def influence_generator(
        self,
        activations: Dict[str, torch.Tensor],
        use_abs: bool = False,
    ) -> Callable[[int, float], Tuple[sp.coo_matrix, sp.coo_matrix]]:
        """
        Parameters
        ----------
        activations : dict of tensors
        use_abs : boolean, optional, default=False
            If True, use the absolute value of element activations to determine influence

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

                    if use_abs:
                        t = torch.abs(t)

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

                    if len(t.shape) == 4:
                        # m_idx (idx of max neuron per channel) are flat, so start with ch x h*w
                        neuron_weights = torch.zeros(
                            t.shape[1], t.shape[2] * t.shape[3]
                        )
                        # assign infl. weight to max neuron in each infl. channel
                        neuron_weights[range(t.shape[1]), m_idx] = influential_weights
                        # reshape weights to ch x h x w
                        influential_weights = neuron_weights.view(t.shape[1:])

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
        use_abs: bool = False,
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
        use_abs : boolean, optional, default=False
            If True, use the absolute value of element activations to determine influence

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
        ) = self.build_dicts(x, layers_to_profile, param=threshold, use_abs=use_abs)

        return Profile(
            neuron_counts=neuron_counts,
            neuron_weights=neuron_weights,
            synapse_counts=synapse_counts,
            synapse_weights=synapse_weights,
            num_inputs=1,
            neuron_type="element",
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
        flat_idx = neuron_counts.nonzero()
        if func is ElementProfiler.contrib_linear:
            infl_idx = flat_idx[1]
        # unravel to 3D for conv. layer
        else:
            row_idx, col_idx = np.unravel_index(flat_idx[1], y_out[ldx].shape[-2:])
            infl_idx = np.stack((flat_idx[0], row_idx, col_idx)).T
        infl_idx = torch.Tensor(infl_idx).long()

        # call contrib function
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

        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_linear cannot handle more than one dict"
                + "key for x_in or y_out"
            )

        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x_ldx = list(x_in.keys())[0]
        x_in = x_in[x_ldx]

        j = infl_neurons

        with torch.no_grad():
            linear = self.model.available_modules()[layer[0]]

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

            # fullify synapse weights to outdims x indims
            if len(xdims) == 1:
                synapse_weights = torch.zeros(ydims[0], xdims[0])
            else:
                synapse_weights = torch.zeros(ydims[0], xdims[0] * xdims[1] * xdims[2])
            synapse_weights[j] = sws_compact

            synapse_counts = synapse_weights.bool().int()

            # construct neuron counts
            neuron_counts = torch.sum(synapse_counts, dim=0).unsqueeze(0).int()
            if len(xdims) > 1:
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
                "contrib_conv2d cannot handle more than one dict"
                + "key for x_in or y_out"
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

            # get channel, i, j indices (unravel if flat)
            if len(infl_neurons.shape) == 1:
                ch, i, j = get_index(infl_neurons, kernel_size)
            else:
                ch, i, j = infl_neurons.unbind(dim=1)

            # this is not [:,ch] because W dims are [#filters=outdim, #channels = indim, ker_size,ker_size] - don't erase this
            W = W[ch]
            B = B[ch] if B is not None else torch.Tensor()

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
            synapse_weights = torch.sparse_coo_tensor(indices, values, size).coalesce()

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
        layer: List[str],
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
        layer : List[str]
            list containing single key in self.model.available_modules() dictionary
        threshold : None or float
            not used, placeholder for uniformity in arguments.

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

        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x_ldx = list(x_in.keys())[0]
        x_in = x_in[x_ldx]

        with torch.no_grad():
            xdims = x_in.shape
            ydims = y_out.shape

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
            synapse_weights = torch.sparse_coo_tensor(indices, values, size).coalesce()

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

        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x_ldx = list(x_in.keys())[0]
        x_in = x_in[x_ldx]

        with torch.no_grad():
            xdims = x_in.shape
            ydims = y_out.shape

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
            synapse_weights = torch.sparse_coo_tensor(indices, values, size).coalesce()

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

        neuron_counts = dict()
        synapse_counts = dict()
        synapse_weights = dict()

        with torch.no_grad():
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

        ch, i, j = infl_neurons.unbind(dim=1)
        neuron_counts = torch.zeros(xdims[1:], dtype=torch.int)
        neuron_counts[ch, i, j] += 1

        indices = torch.stack((ch, i, j, ch, i, j))
        values = torch.ones((len(infl_neurons)))

        size = ydims[1:] + xdims[1:]

        synapse_counts = torch.sparse_coo_tensor(indices, values, size, dtype=torch.int)
        synapse_weights = torch.sparse_coo_tensor(indices, values, size)

        return (
            matrix_convert(neuron_counts),
            matrix_convert(synapse_counts),
            matrix_convert(synapse_weights),
        )
