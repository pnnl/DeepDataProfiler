from deep_data_profiler.classes.profile import Profile
from deep_data_profiler.classes.torch_profiler import TorchProfiler
from deep_data_profiler.utils import get_index, submatrix_generator
import numpy as np
import scipy.sparse as sp
import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings


class ElementProfiler(TorchProfiler):
    """
    ElementProfiler identifies influential elements of an activation tensor. Influential
    element neurons are identified by their value or absolute value. Contributing
    neurons in the previous layer are the elements in the receptive field with the
    greatest values whose sum reaches a specified threshold, or percentage of the
    value of the influential element.
    """

    def influence_generator(
        self,
        activations: Dict[str, torch.Tensor],
        use_abs: bool = False,
    ) -> Callable[[int, float], Tuple[sp.coo_matrix, sp.coo_matrix]]:
        """
        Parameters
        ----------
        activations : dict of tensors
        use_abs : boolean, default=False
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
            threshold : float, default=0.1

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
                    t = torch.where(
                        torch.eq(hd > 0, nd > 0).to(self.device),
                        nd,
                        torch.zeros(nd.shape, device=self.device),
                    )

                    if use_abs:
                        # take the absolute value of all activations
                        t = torch.abs(t)

                    # check if module is a conv. layer
                    is_conv = len(t.shape) == 4

                    # define dimensions
                    if is_conv:
                        num_channels, h, w = t[0].shape
                    else:
                        num_elements = t[0].shape[0]

                    # if module is a conv. layer, consider the max val element from each channel
                    if is_conv:
                        channel_vals = t.view(1, num_channels, h * w)
                        m, m_idx = torch.max(channel_vals, dim=-1)
                    # otherwise if module is fully connected, consider all elements
                    else:
                        m = t

                    if not use_abs:
                        # ignore negative elements when not taking absolute value
                        m = torch.where(
                            m > 0, m, torch.zeros(m.shape, device=self.device)
                        )

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

                    # send values and indices to cpu if necessary
                    if self.device != "cpu":
                        ordsmat_vals = ordsmat_vals.cpu()
                        ordsmat_indices = ordsmat_indices.cpu()
                        if is_conv:
                            m_idx = m_idx.cpu()

                    # define shape depending on if module is conv. or FC
                    if is_conv:
                        shape = (num_channels, h * w)
                        # trace max val from accepted channel back to element index
                        indices = (ordsmat_indices, m_idx[:, ordsmat_indices].squeeze())
                    else:
                        shape = (1, num_elements)
                        indices = (np.zeros(accept + 1), ordsmat_indices)

                    # construct weights and counts sparse matrices
                    influential_weights = sp.coo_matrix(
                        (ordsmat_vals, indices), shape=shape
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
        threshold: float = 0.1,
        layers_to_profile: Union[list, Tuple] = None,
        use_abs: bool = False,
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
        use_abs : boolean, default=False
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
            activation_shapes,
        ) = self.build_dicts(
            x,
            infl_threshold=threshold,
            contrib_threshold=threshold,
            layers_to_profile=layers_to_profile,
            use_abs=use_abs,
        )

        return Profile(
            neuron_counts=neuron_counts,
            neuron_weights=neuron_weights,
            synapse_counts=synapse_counts,
            synapse_weights=synapse_weights,
            activation_shapes=activation_shapes,
            pred_dict=self.pred_dict,
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

        # get the appropriate contrib function for the module
        func = getattr(self.__class__, self.layerdict[ldx][1])
        # take index as is for FC layer
        if func is ElementProfiler.contrib_linear:
            infl_idx = torch.LongTensor(neuron_counts.col)
        # unravel (channel, flat spatial) to (channel, row, col) for conv. layer
        else:
            h, w = y_out[ldx][0].shape[1:]
            row_idx, col_idx = np.unravel_index(neuron_counts.col, (h, w))
            infl_idx = torch.LongTensor(
                np.stack((neuron_counts.row, row_idx, col_idx)).T,
            )

        # send influential indices to correct device
        infl_idx = infl_idx.to(self.device)

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
                in_elements = in_channels * h_in * w_in
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
            z = torch.where(z > 0, z, torch.zeros(z.shape, device=self.device))

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
            # if accept == in_elements, all values taken as contributors
            # subtract 1 in this case to avoid IndexError when adding additional accept
            accept = torch.where(accept < in_elements, accept, accept - 1)
            # add additional accept, ie accept + 1
            bool_accept[range(num_infl), accept] = True

            # normalize by total sum of all contributions
            ordsmat_vals /= cumsum[range(num_infl), accept].unsqueeze(-1)

            # grab accepted contributor values and indices
            ordsmat_vals = ordsmat_vals[bool_accept]
            contrib_idx = ordsmat_indices[bool_accept]

            # send indices and values to cpu if necessary
            if self.device != "cpu":
                accept = accept.cpu()
                ordsmat_vals = ordsmat_vals.cpu()
                contrib_idx = contrib_idx.cpu()
                infl_neurons = infl_neurons.cpu()

            # repeat each influential neuron once for each of its accepted contributors
            infl_idx = np.repeat(infl_neurons, accept + 1)

            # define shape of synapse counts/weights
            shape = (out_elements, in_elements)

            # construct synapse weights and counts
            synapse_weights = sp.coo_matrix(
                (ordsmat_vals, (infl_idx, contrib_idx)),
                shape=shape,
            )
            synapse_weights.eliminate_zeros()

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

            # reshape neuron counts to in_channels x h_in * w_in if x_in is a conv layer
            if conv_input:
                neuron_counts = neuron_counts.reshape((in_channels, h_in * w_in))

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
            dimensions: num_influential x 3 (channel, row, col)
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

            # get channel, row, col indices of influential neurons
            ch, i, j = infl_neurons.unbind(dim=1)

            # W dims are [#filters=out_channels, #channels=in_channels, kernel_size, kernel_size]
            W = W[ch]
            B = B[ch] if B is not None else torch.Tensor()

            # find receptive fields for each influential neuron
            # rfield dimensions: num_infl x in_channels x kernel_size x kernel_size
            rfield = submatrix_generator(x_in, stride, kernel_size, padding)(i, j)
            # convolve weights and receptive fields
            z = W * rfield
            # order neurons in receptive field by greatest contribution
            # ordsmat dimensions: num_infl x in_channels*kernel_size^2
            ordsmat_vals, ordsmat_indices = torch.sort(
                z.view(num_infl, in_channels * kernel_size**2),
                dim=1,
                descending=True,
            )

            # take the cumsum and normalize by the total sum to find the threshold goal
            cumsum = torch.cumsum(ordsmat_vals, dim=1)
            totalsum = cumsum[:, -1].unsqueeze(-1)
            goals = cumsum / totalsum

            # find the indices within the threshold goal
            bool_accept = goals <= threshold
            accept = torch.sum(bool_accept, dim=1)
            # if accept == kernel_size**2, all values taken as contributors
            # subtract 1 in this case to avoid IndexError when adding additional accept
            accept = torch.where(accept < kernel_size**2, accept, accept - 1)

            # add additional accept, ie accept + 1
            bool_accept[range(len(accept)), accept] = True

            # normalize by total sum of all accepted contributions
            ordsmat_vals /= cumsum[range(num_infl), accept].unsqueeze(-1)

            # convert flattened contributor indices to index in receptive field
            ords_ch, ords_row, ords_col = get_index(ordsmat_indices, kernel_size)
            # convert receptive field index to full index in x_in
            ordsi = ords_row + stride * i.unsqueeze(-1) - padding
            ordsj = ords_col + stride * j.unsqueeze(-1) - padding

            # grab accepted contributor values and (channel, row, col) indices
            ords_ch = ords_ch[bool_accept]
            ordsi = ordsi[bool_accept]
            ordsj = ordsj[bool_accept]
            ordsmat_vals = ordsmat_vals[bool_accept]

            # send indices and values to cpu if necessary
            if self.device != "cpu":
                accept = accept.cpu()
                ords_ch = ords_ch.cpu()
                ordsi = ordsi.cpu()
                ordsj = ordsj.cpu()
                ordsmat_vals = ordsmat_vals.cpu()
                ch = ch.cpu()
                i = i.cpu()
                j = j.cpu()

            # repeat each influential (channel, row, col) index once for each of its
            # accepted contributors
            ch = np.repeat(ch, accept + 1)
            i = np.repeat(i, accept + 1)
            j = np.repeat(j, accept + 1)

            # identify and remove padding indices, which have either row or col index
            # outside of the range [0, #row/col)
            if padding > 0:
                valid_idx = (
                    (ordsi >= 0) & (ordsi < h_in) & (ordsj >= 0) & (ordsj < w_in)
                )
                ords_ch = ords_ch[valid_idx]
                ordsi = ordsi[valid_idx]
                ordsj = ordsj[valid_idx]
                ordsmat_vals = ordsmat_vals[valid_idx]
                ch = ch[valid_idx]
                i = i[valid_idx]
                j = j[valid_idx]

            # flatten all influential and contributor indices
            infl_idx = np.ravel_multi_index((ch, i, j), (out_channels, h_out, w_out))
            contrib_idx = np.ravel_multi_index(
                (ords_ch, ordsi, ordsj), (in_channels, h_in, w_in)
            )

            # define shape of synapse counts/weights
            shape = (out_channels * h_out * w_out, in_channels * h_in * w_in)

            # construct synapse weights and counts
            synapse_weights = sp.coo_matrix(
                (ordsmat_vals, (infl_idx, contrib_idx)), shape=shape
            )
            synapse_weights.eliminate_zeros()

            synapse_counts = sp.coo_matrix(
                (
                    np.ones(synapse_weights.data.shape),
                    (synapse_weights.row, synapse_weights.col),
                ),
                shape=shape,
                dtype=int,
            )

            # construct neuron counts by summing over columns of synapse counts
            # and reshaping to in_channels x h_in * w_in
            neuron_counts = sp.coo_matrix(
                sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
            ).reshape(in_channels, h_in * w_in)

        return neuron_counts, synapse_counts, synapse_weights

    def contrib_max2d(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        infl_neurons: torch.Tensor,
        layer: List[str],
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
            dimensions: num_influential x 3 (channel, row, col)
        layer : List[str]
            list containing single key in self.model.available_modules() dictionary
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

        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x_ldx = list(x_in.keys())[0]
        x_in = x_in[x_ldx]

        with torch.no_grad():
            # grab max pooling module
            maxpool = self.model.available_modules()[layer[0]]

            # assumption is that kernel size, stride are equal in both dimensions
            # and padding preserves input size
            kernel_size = maxpool.kernel_size
            stride = maxpool.stride
            padding = maxpool.padding

            # define dimensions, in channels and out channels
            in_channels, h_in, w_in = x_in[0].shape
            out_channels, h_out, w_out = y_out[0].shape

            # get number of influential neurons
            num_infl = infl_neurons.shape[0]

            # get channel, row, col indices of influential neurons
            ch, i, j = infl_neurons.unbind(dim=1)

            # find receptive fields for each influential neuron
            # rfield dimensions: num_infl x kernel_size x kernel_size
            rfield = submatrix_generator(x_in, stride, kernel_size, padding)(i, j)[
                range(num_infl), ch
            ]
            # find max val from each receptive field
            maxval, maxidx = torch.max(rfield.view(num_infl, kernel_size**2), dim=1)

            # convert index of max vals in receptive field to full index in x_in
            maxi = (maxidx // kernel_size + stride * i) - padding
            maxj = (maxidx % kernel_size + stride * j) - padding

            # send indices and values to cpu if necessary
            if self.device != "cpu":
                maxi = maxi.cpu()
                maxj = maxj.cpu()
                maxval = maxval.cpu()
                ch = ch.cpu()
                i = i.cpu()
                j = j.cpu()

            # identify and remove padding indices, which have either row or col index
            # outside of the range [0, #row/col)
            if padding > 0:
                valid_idx = (maxi >= 0) & (maxi < h_in) & (maxj >= 0) & (maxj < w_in)
                maxi = maxi[valid_idx]
                maxj = maxj[valid_idx]
                maxval = maxval[valid_idx]
                ch = ch[valid_idx]
                i = i[valid_idx]
                j = j[valid_idx]

            # flatten all influential and contributor indices
            infl_idx = np.ravel_multi_index((ch, i, j), (out_channels, h_out, w_out))
            contrib_idx = np.ravel_multi_index(
                (ch, maxi, maxj), (in_channels, h_in, w_in)
            )

            # define shape of synapse counts/weights
            shape = (out_channels * h_out * w_out, in_channels * h_in * w_in)

            # construct synapse weights and counts
            synapse_weights = sp.coo_matrix(
                (maxval, (infl_idx, contrib_idx)), shape=shape
            )
            synapse_weights.eliminate_zeros()

            synapse_counts = sp.coo_matrix(
                (
                    np.ones(synapse_weights.data.shape),
                    (synapse_weights.row, synapse_weights.col),
                ),
                shape=shape,
                dtype=int,
            )

            # construct neuron counts by summing over columns of synapse counts
            # and reshaping to in_channels x h_in * w_in
            neuron_counts = sp.coo_matrix(
                sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
            ).reshape(in_channels, h_in * w_in)

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
            dimensions: num_influential x 3 (channel, row, col)
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
                "contrib_adaptive_avg_pool2d requires x_in and y_out to have exactly "
                + "one layer key each"
            )

        y_ldx = list(y_out.keys())[0]
        y_out = y_out[y_ldx]
        x_ldx = list(x_in.keys())[0]
        x_in = x_in[x_ldx]

        with torch.no_grad():
            # grab adaptive average pooling module
            avgpool = self.model.available_modules()[layer[0]]

            # define dimensions, in channels and out channels
            in_channels, h_in, w_in = x_in[0].shape
            out_channels, h_out, w_out = y_out[0].shape

            # grab dimensions of avgpool from hyperparameters
            output_size = avgpool.output_size[0]
            input_size = h_in
            stride = input_size // output_size
            kernel_size = input_size - (output_size - 1) * stride

            # get number of influential neurons
            num_infl = infl_neurons.shape[0]

            # get channel, row, col indices of influential neurons
            ch, i, j = infl_neurons.unbind(dim=1)

            # find receptive field for each influential neuron
            # rfield dimensions: num_infl x kernel_size x kernel_size
            rfield = submatrix_generator(x_in, stride, kernel_size)(i, j)[
                range(num_infl), ch
            ]
            # only consider values > 0
            rfield = torch.where(
                rfield > 0, rfield, torch.zeros(rfield.shape, device=self.device)
            )

            # order neurons in receptive field by greatest normalized contribution
            ordsmat_vals, ordsmat_indices = torch.sort(
                rfield.view(num_infl, kernel_size**2) / kernel_size**2,
                dim=1,
                descending=True,
            )

            # take the cumsum and normalize by the total sum to find the threshold goal
            cumsum = torch.cumsum(ordsmat_vals, dim=1)
            totalsum = cumsum[:, -1].unsqueeze(-1)
            goals = cumsum / totalsum

            # find the indices within the threshold goal
            bool_accept = goals <= threshold
            accept = torch.sum(bool_accept, dim=1)
            # if accept == kernel_size**2, all values taken as contributors
            # subtract 1 in this case to avoid IndexError when adding additional accept
            accept = torch.where(accept < kernel_size**2, accept, accept - 1)

            # add additional accept, ie accept + 1
            bool_accept[range(num_infl), accept] = True

            # normalize by total sum of all accepted contributions
            ordsmat_vals /= cumsum[range(num_infl), accept].unsqueeze(-1)

            # convert contributor indices in receptive field to full index in x_in
            ordsi = ordsmat_indices // kernel_size + stride * i.unsqueeze(-1)
            ordsj = ordsmat_indices % kernel_size + stride * j.unsqueeze(-1)

            # grab accepted contributor values and (channel, row, col) indices
            ordsi = ordsi[bool_accept]
            ordsj = ordsj[bool_accept]
            ordsmat_vals = ordsmat_vals[bool_accept]

            # send indices and values to cpu if necessary
            if self.device != "cpu":
                accept = accept.cpu()
                ordsi = ordsi.cpu()
                ordsj = ordsj.cpu()
                ordsmat_vals = ordsmat_vals.cpu()
                ch = ch.cpu()
                i = i.cpu()
                j = j.cpu()

            # repeat each influential (channel, row, col) index once for each of its
            # accepted contributors
            ch = np.repeat(ch, accept + 1)
            i = np.repeat(i, accept + 1)
            j = np.repeat(j, accept + 1)

            # flatten all influential and contributor indices
            infl_idx = np.ravel_multi_index((ch, i, j), (out_channels, h_out, w_out))
            contrib_idx = np.ravel_multi_index(
                (ch, ordsi, ordsj), (in_channels, h_in, w_in)
            )

            # define shape of synapse counts/weights
            shape = (out_channels * h_out * w_out, in_channels * h_in * w_in)

            # construct synapse weights and counts
            synapse_weights = sp.coo_matrix(
                (ordsmat_vals, (infl_idx, contrib_idx)), shape=shape
            )
            synapse_weights.eliminate_zeros()

            synapse_counts = sp.coo_matrix(
                (
                    np.ones(synapse_weights.data.shape),
                    (synapse_weights.row, synapse_weights.col),
                ),
                shape=shape,
                dtype=int,
            )

            # construct neuron counts by summing over columns of synapse counts
            # and reshaping to in_channels x h_in * w_in
            neuron_counts = sp.coo_matrix(
                sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
            ).reshape(in_channels, h_in * w_in)

        return neuron_counts, synapse_counts, synapse_weights

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
            dimensions: num_influential x 3 (channel, row, col)
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

            # get number of influential neurons
            num_infl = infl_neurons.shape[0]

            # get channel, row, col indices
            ch, i, j = infl_neurons.unbind(dim=1)

            # only take elements corresponding to influentials in y_out
            vals = x_in[:, ch, i, j]

        # construct neuron counts and synapse counts/weights for each input layer
        # when the element val in the first input layer is greater than the second,
        # fully attribute influence to the corresponding element in the first layer
        nc1, sc1, sw1 = self.contrib_identity(
            {x1_ldx: x_in[0].unsqueeze(0)},
            {y_ldx: y_out},
            infl_neurons[vals[0] > vals[1]],
        )

        # when the element val in the second input layer is greater than or equal to the
        # first, fully attribute influence to the corresponding element in the second layer
        nc2, sc2, sw2 = self.contrib_identity(
            {x2_ldx: x_in[1].unsqueeze(0)},
            {y_ldx: y_out},
            infl_neurons[vals[0] <= vals[1]],
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
        layer: List[str] = None,
        threshold: Optional[float] = None,
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
            dimensions: num_influential x 3 (channel, row, col)
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

        if y_out[0].shape != x_in[0].shape:
            raise NotImplementedError(
                "contrib_identity requires x_in and y_out to have the same shape"
            )

        with torch.no_grad():
            # define dimensions
            num_channels, h, w = y_out[0].shape

            # get number of influential neurons
            num_infl = infl_neurons.shape[0]

            # get channel, row, col indices of influential neurons
            ch, i, j = infl_neurons.unbind(dim=1)

            # send indices to cpu if necessary
            if self.device != "cpu":
                ch = ch.cpu()
                i = i.cpu()
                j = j.cpu()

            # flatten influential indices
            infl_flat = np.ravel_multi_index((ch, i, j), (num_channels, h, w))

        # define shape of synapse counts and weights
        shape = (num_channels * h * w, num_channels * h * w)

        # construct synapse weights and counts
        synapse_weights = sp.coo_matrix(
            (np.ones(num_infl), (infl_flat, infl_flat)), shape=shape
        )

        synapse_counts = sp.coo_matrix(
            (np.ones(num_infl), (infl_flat, infl_flat)), shape=shape, dtype=int
        )

        # construct neuron counts by summing over columns of synapse counts
        # and reshaping to num_channels x h * w
        neuron_counts = sp.coo_matrix(
            sp.csc_matrix(synapse_counts).sum(axis=0),
            dtype=int,
        ).reshape(num_channels, h * w)

        return neuron_counts, synapse_counts, synapse_weights
