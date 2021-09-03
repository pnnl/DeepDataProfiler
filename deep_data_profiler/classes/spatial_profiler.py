from deep_data_profiler.classes.profile import Profile
from deep_data_profiler.classes.channel_profiler import ChannelProfiler
from deep_data_profiler.utils import get_index, submatrix_generator
import numpy as np
import scipy.sparse as sp
import torch
from typing import Callable, Dict, List, Tuple, Union


class SpatialProfiler(ChannelProfiler):
    def influence_generator(
        self,
        activations: Dict[str, torch.Tensor],
        use_quantile: bool = True,
        max_infl: int = 100,
        norm: int = 2,
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
            layer_number: int,
            threshold: float,
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
                    t = activations[nd]

                    if len(t.shape) == 4:
                        spatials = t.view(t.shape[:2] + (-1,))
                        m = torch.linalg.norm(spatials, ord=norm, dim=1)
                    else:
                        m = torch.where(t > 0, t, torch.zeros(t.shape))

                    ordsmat_vals, ordsmat_indices = torch.sort(m, descending=True)

                    if use_quantile:
                        bool_accept = ordsmat_vals >= torch.quantile(
                            ordsmat_vals, threshold
                        )

                    else:
                        # take the cumsum and normalize by total contribution per dim
                        cumsum = torch.cumsum(ordsmat_vals, dim=1)
                        totalsum = cumsum[:, -1]

                        # find the indices within the threshold goal, per dim
                        bool_accept = cumsum / totalsum <= threshold

                    bool_accept[:, max_infl - 1 :] = False
                    accept = torch.sum(bool_accept, dim=1)

                    # if accept == m.shape[1], all values taken as influential
                    # subtract 1 in this case to avoid IndexError when adding additional accept
                    accept = torch.where(accept < m.shape[1], accept, accept - 1)

                    # add additional accept, ie accept + 1
                    # use range to enumerate over batch size entries of accept
                    bool_accept[range(len(accept)), accept] = True

                    if not use_quantile:
                        # normalize by final accepted cumsum
                        ordsmat_vals /= cumsum[:, accept]

                    influential_weights = sp.coo_matrix(
                        (
                            ordsmat_vals[bool_accept],
                            (torch.zeros(accept + 1), ordsmat_indices[bool_accept]),
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
        infl_threshold: float = 0.97,
        contrib_threshold: float = 0.1,
        use_quantile: bool = True,
        max_infl: int = 100,
        norm: int = 2,
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
            infl_threshold=infl_threshold,
            contrib_threshold=contrib_threshold,
            use_quantile=use_quantile,
            max_infl=max_infl,
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

            # sum contribution per spatial if x_in is a conv layer
            if len(xdims) > 1:
                sws_compact = sws_compact.view(len(j), xdims[0], -1)
                sws_compact = torch.sum(sws_compact, dim=1)

            # fullify synapse weights to outdims x indims
            if len(xdims) == 1:
                synapse_weights = sp.csr_matrix((ydims[0], xdims[0]))
            else:
                synapse_weights = sp.csr_matrix((ydims[0], xdims[1] * xdims[2]))

            synapse_weights[j] = sws_compact
            synapse_weights = synapse_weights.tocoo()
            synapse_weights.eliminate_zeros()

            synapse_counts = sp.coo_matrix(
                (
                    np.ones(synapse_weights.data.shape),
                    (synapse_weights.row, synapse_weights.col),
                ),
                shape=synapse_weights.shape,
                dtype=int,
            )

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

            # get i, j from flat spatial indices
            i, j = np.unravel_index(infl_neurons, (h_out, w_out))

            # find receptive fields for each influential neuron
            # xmat dimensions: num_infl x in_channels x kernel_size x kernel_size
            rfield = submatrix_generator(x_in, stride, kernel_size, padding)(i, j)

            # prepare weight matrix and receptive field matrices for batched matrix multiplication
            W_batch = W.view(out_channels, in_channels, -1).permute(2, 0, 1)
            rf_batch = rfield.view(num_infl, in_channels, -1).T

            # multiply each spatial from weight matrix with corresponding spatial from receptive field
            wx = torch.bmm(W_batch, rf_batch).T

            # calculate sum of weighted spatials (equal to y_out spatial activation)
            y = torch.sum(wx, dim=-1)
            # caluclate strength of and unit vector in direction of influential spatials
            y_norm = torch.linalg.norm(y, dim=-1)
            y_uvec = y.T / y_norm

            # project weighted spatials onto direction of the output spatial
            wx_proj = torch.bmm(wx.transpose(1, 2), y_uvec.T.unsqueeze(-1)).squeeze(-1)

            # order spatials in receptive field by abs(magnitude in direction of output spatial)
            ordproj_vals, ordproj_idx = torch.sort(
                torch.abs(wx_proj), dim=1, descending=True
            )

            # use indices from spatial ordering to order the weighted spatial vectors
            # (preparing to take partial sum)
            gather_idx = ordproj_idx.unsqueeze(1).repeat((1, out_channels, 1))
            ordwx = wx.gather(-1, gather_idx)

            # partial sums of ordered weighted spatials from receptive field
            cumsum = torch.cumsum(ordwx, dim=-1)
            # calculate distance between partial sum vector and output spatial
            cumsum_dist = y.unsqueeze(-1) - cumsum
            # accept as few spatials as needed to bring distance between partial sum
            # vector and output spatial vector within TH% of magnitude of output spatial
            bool_accept = torch.bmm(
                y_uvec.T.unsqueeze(1), cumsum_dist
            ).squeeze() >= threshold * y_norm.unsqueeze(-1).repeat(1, kernel_size ** 2)
            accept = torch.sum(bool_accept, dim=-1)
            # if accept == kernel_size**2, all values taken as contributors
            # subtract 1 in this case to avoid IndexError when adding additional accept
            accept = torch.where(accept < kernel_size ** 2, accept, accept - 1)

            # add additional accept, ie accept + 1
            bool_accept[range(num_infl), accept] = True

            # normalize magnitude of projection as a fraction of magnitude of output spatial
            ordproj_vals /= y_norm.unsqueeze(-1)

            # find accepted synapses, all other values zero.
            unordered_synapses = torch.where(
                bool_accept, ordproj_vals, torch.zeros(ordproj_vals.shape)
            )

            # re-order to mantain proper neuron ordering
            # note: ordsort_vals is num_infl x kernel_size**2, and each row
            #   is equal to range(kernel_size**2)
            ordsort_vals, ordsort_idx = ordproj_idx.sort()
            sws_compact = unordered_synapses.gather(-1, ordsort_idx).squeeze()

            # convert ordered flattened contributor indices to full indices in x_in
            ordsi = (
                ordsort_vals // kernel_size + stride * np.expand_dims(i, -1) - padding
            )
            ordsj = (
                ordsort_vals % kernel_size + stride * np.expand_dims(j, -1) - padding
            )

            # copy influential indices along new dimension to index with contributors
            infl_idx = infl_neurons.unsqueeze(-1).repeat(1, kernel_size ** 2).squeeze()

            # identify and remove padding indices
            if padding > 0:
                valid_idx = (
                    (ordsi >= 0) & (ordsi < h_in) & (ordsj >= 0) & (ordsj < w_in)
                )
                ordsi = ordsi[valid_idx]
                ordsj = ordsj[valid_idx]
                infl_idx = infl_idx[valid_idx]
                sws_compact = sws_compact[valid_idx]

            contrib_idx = np.ravel_multi_index((ordsi, ordsj), (h_in, w_in)).squeeze()

            synapse_weights = sp.coo_matrix(
                (sws_compact, (infl_idx, contrib_idx)),
                shape=(h_out * w_out, h_in * w_in),
            )
            synapse_weights.eliminate_zeros()

            synapse_counts = sp.coo_matrix(
                (
                    np.ones(synapse_weights.data.shape),
                    (synapse_weights.row, synapse_weights.col),
                ),
                shape=(h_out * w_out, h_in * w_in),
                dtype=int,
            )

            neuron_counts = sp.coo_matrix(
                sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
            )

        return neuron_counts, synapse_counts, synapse_weights

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

        if len(y_out) != 1 or len(x_in) != 1:
            raise NotImplementedError(
                "contrib_max2d cannot handle more than one dict key for x_in or y_out"
            )

        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x_ldx = list(x_in.keys())[0]
        x_in = x_in[x_ldx]

        with torch.no_grad():
            maxpool = self.model.available_modules()[layer[0]]

            # assumption is that kernel size, stride are equal in both dimensions
            # and padding preserves input size
            kernel_size = maxpool.kernel_size
            stride = maxpool.stride
            padding = maxpool.padding

            # define dimensions, in channels and out channels
            xdims = x_in.shape
            ydims = y_out.shape
            in_channels, h_in, w_in = xdims[1:]
            out_channels, h_out, w_out = ydims[1:]

            # get number of influential neurons
            num_infl = infl_neurons.shape[0]

            # get i, j from flat spatial indices
            i, j = np.unravel_index(infl_neurons, (h_out, w_out))

            # get flat spatial indices of receptive field
            flat_idx = torch.Tensor(
                np.tile(np.arange(kernel_size ** 2), (num_infl, 1))
            ).long()

            # convert flattened receptive field indices to full indices in x_in
            ordsi = flat_idx // kernel_size + stride * np.expand_dims(i, -1) - padding
            ordsj = flat_idx % kernel_size + stride * np.expand_dims(j, -1) - padding

            # copy influential indices along new dimension to index with contributors
            infl_idx = infl_neurons.unsqueeze(-1).repeat(1, kernel_size ** 2).squeeze()

            # identify and remove padding indices
            if padding > 0:
                valid_idx = (
                    (ordsi >= 0) & (ordsi < h_in) & (ordsj >= 0) & (ordsj < w_in)
                )
                ordsi = ordsi[valid_idx]
                ordsj = ordsj[valid_idx]
                infl_idx = infl_idx[valid_idx]

            contrib_idx = np.ravel_multi_index((ordsi, ordsj), (h_in, w_in)).flatten()
            infl_idx = infl_idx.view(-1)

            # import ipdb;ipdb.set_trace()
            synapse_weights = sp.coo_matrix(
                (np.ones(contrib_idx.shape), (infl_idx, contrib_idx)),
                shape=(h_out * w_out, h_in * w_in),
            )

            synapse_counts = sp.coo_matrix(
                (
                    np.ones(contrib_idx.shape),
                    (synapse_weights.row, synapse_weights.col),
                ),
                shape=(h_out * w_out, h_in * w_in),
                dtype=int,
            )

            neuron_counts = sp.coo_matrix(
                sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
            )

        return neuron_counts, synapse_counts, synapse_weights

    def contrib_adaptive_avg_pool2d(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str],
        threshold: float = 0.1,
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
        threshold : float

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

        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x_ldx = list(x_in.keys())[0]
        x_in = x_in[x_ldx]

        with torch.no_grad():

            # define dimensions, in channels and out channels
            xdims = x_in.shape
            ydims = y_out.shape
            in_channels, h_in, w_in = xdims[1:]
            out_channels, h_out, w_out = ydims[1:]

            avgpool = self.model.available_modules()[layer[0]]

            # Grab dimensions of avgpool from parameters
            output_size = avgpool.output_size[0]
            input_size = h_in
            stride = input_size // output_size
            kernel_size = input_size - (output_size - 1) * stride

            # get number of influential neurons
            num_infl = infl_neurons.shape[0]

            # get i, j from flat spatial indices
            i, j = np.unravel_index(infl_neurons, (h_out, w_out))

            rfield = submatrix_generator(x_in, stride, kernel_size)(i, j)

            wx = rfield.view(num_infl, in_channels, kernel_size ** 2) * (
                1 / kernel_size ** 2
            )

            # calculate sum of weighted spatials (equal to y_out spatial activation)
            y = torch.sum(wx, dim=-1)
            # caluclate strength of and unit vector in direction of influential spatials
            y_norm = torch.linalg.norm(y, dim=-1)
            y_uvec = y.T / y_norm

            # project weighted spatials onto direction of the output spatial
            wx_proj = torch.bmm(wx.transpose(1, 2), y_uvec.T.unsqueeze(-1)).squeeze(-1)

            # normalize
            wx_proj = wx_proj.squeeze() / y_norm

            # get flat spatial indices of receptive field
            flat_idx = torch.Tensor(
                np.tile(np.arange(kernel_size ** 2), (num_infl, 1))
            ).long()

            # convert flattened receptive field indices to full indices in x_in
            ordsi = flat_idx // kernel_size + stride * np.expand_dims(i, -1)
            ordsj = flat_idx % kernel_size + stride * np.expand_dims(j, -1)

            # copy influential indices along new dimension to index with contributors
            infl_idx = infl_neurons.unsqueeze(-1).repeat(1, kernel_size ** 2).squeeze()

            # convert full indices in x_in to flattened indices in x_in
            contrib_idx = np.ravel_multi_index((ordsi, ordsj), (h_in, w_in)).squeeze()

            # construct synapse weights
            synapse_weights = sp.coo_matrix(
                (wx_proj.view(-1), (infl_idx, contrib_idx)),
                shape=(h_out * w_out, h_in * w_in),
            )
            synapse_weights.eliminate_zeros

            # construct synapse_counts
            synapse_counts = sp.coo_matrix(
                (
                    np.ones(synapse_weights.data.shape),
                    (synapse_weights.row, synapse_weights.col),
                ),
                shape=(h_out * w_out, h_in * w_in),
                dtype=int,
            )
            # construct neuron_counts
            neuron_counts = sp.coo_matrix(
                sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
            )

        return neuron_counts, synapse_counts, synapse_weights

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
        num_channels, h, w = y_out[0].shape

        x1_ldx, x2_ldx = sorted(x_in.keys())

        with torch.no_grad():
            x_in = torch.cat((x_in[x1_ldx], x_in[x2_ldx]))

            num_infl = infl_neurons.shape[0]

            # get i, j from flat spatial indices
            i, j = np.unravel_index(infl_neurons, (h, w))

            # only take spatials corresponding to influentials in y_out
            x = x_in[:, :, i, j].view(2, num_channels, num_infl)

            # calculate sum of spatials (equal to y_out spatial activation)
            y = torch.sum(x, dim=0)
            # caluclate strength of and unit vector in direction of influential spatials
            y_norm = torch.linalg.norm(y, dim=0)
            y_uvec = y / y_norm

            # strength of projection of each contributor on to the influential
            proj = torch.bmm(x.permute(2, 0, 1), y_uvec.T.unsqueeze(-1)).squeeze().T
            # normalize
            proj /= y_norm

        nc1, sc1, sw1 = self.contrib_identity(
            {x1_ldx: x_in[0].unsqueeze(0)}, {y_ldx: y_out}, infl_neurons
        )
        nc2, sc2, sw2 = self.contrib_identity(
            {x2_ldx: x_in[1].unsqueeze(0)}, {y_ldx: y_out}, infl_neurons
        )

        # set proper synapse weights
        sw1 = sp.coo_matrix((proj[0], (sw1.row, sw1.col)), shape=sw1.shape)
        sw2 = sp.coo_matrix((proj[1], (sw2.row, sw2.col)), shape=sw2.shape)

        neuron_counts = (nc1, nc2)
        synapse_counts = sp.block_diag((sc1, sc2))
        synapse_weights = sp.block_diag((sw1, sw2))

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

        in_channels, h_in, w_in = x_in[0].shape
        out_channels, h_out, w_out = y_out[0].shape

        num_infl = infl_neurons.shape[0]

        synapse_weights = sp.coo_matrix(
            (torch.ones(num_infl), (infl_neurons, infl_neurons)),
            shape=(h_out * w_out, h_in * w_in),
        )

        synapse_counts = sp.coo_matrix(
            (
                torch.ones(num_infl),
                (infl_neurons, infl_neurons),
            ),
            shape=(h_out * w_out, h_in * w_in),
            dtype=int,
        )

        neuron_counts = sp.coo_matrix(
            sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
        )

        return neuron_counts, synapse_counts, synapse_weights
