from deep_data_profiler.classes.profile import Profile
from deep_data_profiler.classes.torch_profiler import TorchProfiler
from deep_data_profiler.utils import get_index, submatrix_generator
import numpy as np
import scipy.sparse as sp
import torch
from typing import Callable, Dict, List, Optional, Tuple, Union


class SpatialProfiler(TorchProfiler):
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
        use_quantile : boolean, default=True
            If True, the function returned by the generator will interpret the threshold
            argument as a top quantile to draw influential neurons from,
            Otherwise, the threshold argument will be interpreted as the percentage of
            the sum of all neuron activations achieved by aggregating the
            influential neuron activations
        max_infl : int, default=100
            An upper bound on the number of influential neurons to be identified
        norm : int, default=2
            The order of the norm to be taken to sum the strengths of spatial activations,
            default is L2-norm

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
            nd = self.sght[layer_number][1]  # tail module in layer
            if nd:
                with torch.no_grad():
                    # grab tail module activations
                    t = activations[nd]

                    # check if module is a conv. layer
                    is_conv = len(t.shape) == 4

                    # define dimensions
                    if is_conv:
                        num_channels, h, w = t[0].shape
                    else:
                        num_elements = t[0].shape[0]

                    # if module is a conv. layer, the value of each spatial is its norm
                    if is_conv:
                        spatials = t.view(1, num_channels, h * w)
                        m = torch.linalg.norm(spatials, ord=norm, dim=1)
                    # otherwise if module is fully connected, ignore negative activations
                    else:
                        m = torch.where(t > 0, t, torch.zeros(1, num_elements))

                    # sort values to find strongest neurons
                    ordsmat_vals, ordsmat_indices = torch.sort(m, descending=True)

                    if use_quantile:
                        # accept neurons with value greater than the specified quantile
                        bool_accept = ordsmat_vals >= torch.quantile(
                            ordsmat_vals, threshold
                        )

                    else:
                        # take the cumsum and normalize by total contribution
                        cumsum = torch.cumsum(ordsmat_vals, dim=1)
                        totalsum = cumsum[:, -1]

                        # find the indices within the threshold goal
                        bool_accept = cumsum / totalsum <= threshold

                    # find number of accepted neurons
                    accept = bool_accept.sum()
                    # if accept == m.shape[1] (h * w if conv., num_elements if FC), all
                    # values taken as influential
                    # subtract 1 in this case to avoid IndexError when adding additional accept
                    if accept == m.shape[1]:
                        accept -= 1

                    # add additional accept, ie accept + 1
                    bool_accept[:, accept] = True

                    # accept no more than the specified upper bound on infl. neurons
                    bool_accept[:, max_infl:] = False

                    if not use_quantile:
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
        infl_threshold : float, default=0.97
            Percentage or quantile used to identify influential neurons
        contrib_threshold : float, default=0.1
            Percentage of contribution to track in a profile
        use_quantile : boolean, default=True
            If True, the infl_threshold argument will be interpreted as a top quantile
            to draw influential neurons from,
            Otherwise, the infl_threshold argument will be interpreted as the percentage
            of the sum of all neuron activations achieved by aggregating the
            influential neuron activations
        max_infl : int, default=100
            An upper bound on the number of influential neurons to be identified
        norm : int, default=2
            The order of the norm to be taken when identifying influential neurons to sum
            the strengths of spatial activations, default is L2-norm

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
            neuron_type="spatial",
        )

    def single_profile(
        self,
        x_in: Dict[int, torch.Tensor],
        y_out: Dict[int, torch.Tensor],
        neuron_counts: sp.coo_matrix,
        ldx: int,
        threshold: float = 0.1,
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
        threshold : float, default=0.1
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
        Draws synaptic connections between the given influential neurons in a linear/
        fully connected layer and their contributors in a previous layer

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
        # use the same profiling for channels or neurons for linear
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
            cumsum = torch.cumsum(ordsmat_vals, dim=1).detach()

            # find threshold goal
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
                # get flat spatial index of each contributor
                contrib_idx = np.unravel_index(contrib_idx, (in_channels, h_in * w_in))[
                    1
                ]
                shape = (out_elements, h_in * w_in)
            else:
                shape = (out_elements, in_elements)

            # construct synapse weights and counts
            synapse_weights = sp.coo_matrix(
                (ordsmat_vals, (infl_idx, contrib_idx)),
                shape=shape,
            )
            synapse_weights.eliminate_zeros()

            # sum contribution weight per spatial if x_in is a conv layer
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
        Draws synaptic connections between the given influential neurons in a 2D
        convolutional layer and their contributors in a previous layer

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

        if len(y_out) > 1 or len(x_in) > 1:
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

            # get row/col indices from flat spatial indices
            i, j = np.unravel_index(infl_neurons, (h_out, w_out))

            # special case: when there is only one spatial in the receptive field,
            # take it as the only contributor
            if kernel_size == 1:
                # find row/col index of contributor in x_in
                s = i * stride - padding
                t = j * stride - padding
                # convert row/col index to flat spatial index
                contrib_idx = np.ravel_multi_index((s, t), (h_in, w_in))
                infl_idx = infl_neurons
                # sole contributors have weight 1.0
                ordproj_vals = np.ones(num_infl)

            # standard case: look for contributors based on the strength of their weighted
            # projections in the direction of the influential output spatial
            else:
                # find receptive fields for each influential neuron
                # rfield dimensions: num_infl x in_channels x kernel_size x kernel_size
                rfield = submatrix_generator(x_in, stride, kernel_size, padding)(i, j)

                # prepare weight matrix and receptive field matrices for batched matrix multiplication
                # W batched dimensions: kernel_size^2 x out_channels x in_channels
                W_batch = W.view(out_channels, in_channels, -1).permute(2, 0, 1)
                # rfield batched shape: kernel_size^2 x in_channels x num_infl
                rf_batch = rfield.view(num_infl, in_channels, -1).T

                # multiply each spatial from weight matrix with corresponding spatial from receptive field
                # wx (weighted spatials) dimensions: num_infl x out_channels x kernel_size^2
                wx = torch.bmm(W_batch, rf_batch).T

                # calculate sum of weighted spatials (equal to y_out spatial activation)
                y = torch.sum(wx, dim=-1)
                # caluclate strength of and unit vector in direction of influential spatials
                y_norm = torch.linalg.norm(y, dim=-1)
                y_uvec = y.T / y_norm

                # project weighted spatials onto direction of the output spatial
                # wx_proj (strength of weighted spatial projections) dimensions: num_infl x kernel_size^2
                wx_proj = torch.bmm(wx.transpose(1, 2), y_uvec.T.unsqueeze(-1)).squeeze(
                    -1
                )

                # order spatials in receptive field by abs(magnitude in direction of output spatial)
                ordproj_vals, ordproj_idx = torch.sort(
                    torch.abs(wx_proj), dim=1, descending=True
                )

                # reorder the weighted spatial vectors in each receptive field from
                # strongest to weakest projection (preparing to take partial sum)
                gather_idx = ordproj_idx.unsqueeze(1).repeat((1, out_channels, 1))
                ordwx = wx.gather(-1, gather_idx)

                # partial sums of ordered weighted spatials from receptive field
                cumsum = torch.cumsum(ordwx, dim=-1)
                # calculate distance between partial sum vector and output spatial
                cumsum_dist = y.unsqueeze(-1) - cumsum
                # calculate strength of projected distance vector onto output spatial
                dist_proj = torch.bmm(y_uvec.T.unsqueeze(1), cumsum_dist).squeeze()
                # calculate the threshold distance goal (TH% of magnitude of output spatial)
                goal = threshold * y_norm.unsqueeze(-1).repeat(1, kernel_size ** 2)

                # accept as few spatials as needed to bring projected strength of
                # distance between partial sum vector and output spatial vector within the goal
                bool_accept = dist_proj >= goal
                accept = torch.sum(bool_accept, dim=-1)
                # if accept == kernel_size**2, all values taken as contributors
                # subtract 1 in this case to avoid IndexError when adding additional accept
                accept = torch.where(accept < kernel_size ** 2, accept, accept - 1)
                # update accept to be index of the first False (the distances between the
                # partial sum vectors and the output spatial are not always monotonically
                # decreasing so the following code handles this special case)
                accept = torch.argsort(bool_accept, descending=True, dim=-1)[
                    range(num_infl), accept
                ]
                # add additional accept, ie accept + 1, and don't accept any further
                bool_accept = torch.where(
                    torch.arange(kernel_size ** 2).unsqueeze(0).repeat(num_infl, 1)
                    <= accept.unsqueeze(-1),
                    torch.ones(num_infl, kernel_size ** 2),
                    torch.zeros(num_infl, kernel_size ** 2),
                ).bool()

                # normalize magnitude of projection as a fraction of magnitude of output spatial
                ordproj_vals /= y_norm.unsqueeze(-1)

                # convert ordered flattened contributor indices to full row/col indices in x_in
                ordsi = (
                    ordproj_idx // kernel_size
                    + stride * np.expand_dims(i, -1)
                    - padding
                )
                ordsj = (
                    ordproj_idx % kernel_size + stride * np.expand_dims(j, -1) - padding
                )

                # grab accepted contributor values and indices
                ordsi = ordsi[bool_accept]
                ordsj = ordsj[bool_accept]
                ordproj_vals = ordproj_vals[bool_accept]
                # repeat each influential neuron once for each of its accepted contributors
                infl_idx = np.repeat(infl_neurons, accept + 1)

                # identify and remove padding indices, which have either row or col index
                # outside of the range [0, #row/col)
                if padding > 0:
                    valid_idx = (
                        (ordsi >= 0) & (ordsi < h_in) & (ordsj >= 0) & (ordsj < w_in)
                    )
                    ordsi = ordsi[valid_idx]
                    ordsj = ordsj[valid_idx]
                    ordproj_vals = ordproj_vals[valid_idx]
                    infl_idx = infl_idx[valid_idx]

                # convert row/col indices to flat spatial indices
                contrib_idx = np.ravel_multi_index(
                    (ordsi, ordsj), (h_in, w_in)
                ).squeeze()

            # construct synapse weights and counts
            synapse_weights = sp.coo_matrix(
                (ordproj_vals, (infl_idx, contrib_idx)),
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
            dimensions: num_influential
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

            # get row/col indices from flat spatial indices
            i, j = np.unravel_index(infl_neurons, (h_out, w_out))

            # special case: when there is only one spatial in the receptive field,
            # take it as the only contributor
            if kernel_size == 1:
                # find row/col index of contributor in x_in
                s = i * stride - padding
                t = j * stride - padding
                # convert row/col index to flat spatial index
                contrib_idx = np.ravel_multi_index((s, t), (h_in, w_in))
                infl_idx = infl_neurons
            # standard case: take all spatials in receptive field as equal contributors
            else:
                # get flat spatial indices of receptive field
                flat_idx = np.tile(np.arange(kernel_size ** 2), (num_infl, 1))

                # convert flattened receptive field indices to full indices in x_in
                ordsi = (
                    flat_idx // kernel_size + stride * np.expand_dims(i, -1) - padding
                )
                ordsj = (
                    flat_idx % kernel_size + stride * np.expand_dims(j, -1) - padding
                )

                # repeat each influential neuron once for each of its accepted contributors
                infl_idx = np.repeat(infl_neurons, kernel_size ** 2)

                # identify and remove padding indices, which have either row or col index
                # outside of the range [0, #row/col)

                if padding > 0:
                    valid_idx = (
                        (ordsi >= 0) & (ordsi < h_in) & (ordsj >= 0) & (ordsj < w_in)
                    )
                    ordsi = ordsi[valid_idx]
                    ordsj = ordsj[valid_idx]
                    infl_idx = infl_idx[valid_idx.flatten()]

                # convert row/col indices to flat spatial indices
                contrib_idx = np.ravel_multi_index(
                    (ordsi, ordsj), (h_in, w_in)
                ).flatten()
                infl_idx = infl_idx.view(-1)

            # construct synapse counts and weights
            synapse_weights = sp.coo_matrix(
                (
                    np.ones(contrib_idx.shape) / (kernel_size ** 2),
                    (infl_idx, contrib_idx),
                ),
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

            # construct neuron counts by summing over columns of synapse counts
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

            # get row/col indices from flat spatial indices
            i, j = np.unravel_index(infl_neurons, (h_out, w_out))

            # special case: when there is only one spatial in the receptive field,
            # take it as the only contributor
            if kernel_size == 1:
                s = i * stride
                t = j * stride
                contrib_idx = np.ravel_multi_index((s, t), (h_in, w_in))
                infl_idx = infl_neurons
                ordproj_vals = np.ones(num_infl)
            # standard case: take all contributors weighted by the strength of their
            # projections in the direction of the influential output spatial
            else:
                # find receptive fields for each influential neuron
                # rfield dimensions: num_infl x in_channels x kernel_size x kernel_size
                rfield = submatrix_generator(x_in, stride, kernel_size)(i, j)

                # reshape and normalize
                wx = rfield.view(num_infl, in_channels, kernel_size ** 2) / (
                    kernel_size ** 2
                )

                # calculate sum of normalized spatials (equal to y_out averaged spatial activation)
                y = torch.sum(wx, dim=-1)
                # caluclate strength of and unit vector in direction of influential spatials
                y_norm = torch.linalg.norm(y, dim=-1)
                y_uvec = y.T / y_norm

                # project normalized spatials onto direction of the output spatial
                wx_proj = torch.bmm(wx.transpose(1, 2), y_uvec.T.unsqueeze(-1)).squeeze(
                    -1
                )

                # normalize by strength of output spatial to find contribution percentages
                wx_proj = wx_proj.squeeze() / y_norm

                # get flat spatial indices of receptive field
                flat_idx = np.tile(np.arange(kernel_size ** 2), (num_infl, 1))

                # convert flattened receptive field indices to full indices in x_in
                ordsi = flat_idx // kernel_size + stride * np.expand_dims(i, -1)
                ordsj = flat_idx % kernel_size + stride * np.expand_dims(j, -1)

                # repeat each influential neuron once for each of its accepted contributors
                infl_idx = np.repeat(infl_neurons, kernel_size ** 2)

                # convert row/col indices to flat indices
                contrib_idx = np.ravel_multi_index(
                    (ordsi, ordsj), (h_in, w_in)
                ).squeeze()

                ordproj_vals = wx_proj.view(-1)

            # construct synapse weights and counts
            synapse_weights = sp.coo_matrix(
                (ordproj_vals, (infl_idx, contrib_idx)),
                shape=(h_out * w_out, h_in * w_in),
            )
            synapse_weights.eliminate_zeros

            synapse_counts = sp.coo_matrix(
                (
                    np.ones(synapse_weights.data.shape),
                    (synapse_weights.row, synapse_weights.col),
                ),
                shape=(h_out * w_out, h_in * w_in),
                dtype=int,
            )
            # construct neuron_counts by summing over columns of synapse counts
            neuron_counts = sp.coo_matrix(
                sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
            )

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

            # get number of influential neurons
            num_infl = infl_neurons.shape[0]

            # get row/col indices from flat spatial indices
            i, j = np.unravel_index(infl_neurons, (h, w))

            # only take spatials corresponding to influentials in y_out
            x = x_in[:, :, i, j].view(2, num_channels, num_infl)

            # calculate sum of spatials (equal to y_out spatial activation)
            y = torch.sum(x, dim=0)
            # caluclate strength of and unit vector in direction of influential spatials
            y_norm = torch.linalg.norm(y, dim=0)
            y_uvec = y / y_norm

            # strength of projection of each contributor onto the influential spatial
            proj = torch.bmm(x.permute(2, 0, 1), y_uvec.T.unsqueeze(-1)).squeeze().T
            # normalize by strength of influential spatial
            proj /= y_norm

        # construct neuron counts and synapse counts/weights for each input layer
        nc1, sc1, sw1 = self.contrib_identity(
            {x1_ldx: x_in[0].unsqueeze(0)}, {y_ldx: y_out}, infl_neurons
        )
        nc2, sc2, sw2 = self.contrib_identity(
            {x2_ldx: x_in[1].unsqueeze(0)}, {y_ldx: y_out}, infl_neurons
        )

        # set proper synapse weights according to normalized projection strength
        sw1 = sp.coo_matrix((proj[0], (sw1.row, sw1.col)), shape=sw1.shape)
        sw2 = sp.coo_matrix((proj[1], (sw2.row, sw2.col)), shape=sw2.shape)

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

        if y_out.shape != x_in.shape:
            raise NotImplementedError(
                "contrib_identity requires x_in and y_out to have the same shape"
            )

        # define dimensions
        if len(y_out[0].shape) == 3:
            num_channels, h, w = y_out[0].shape
            num_spatials = h * w
        else:
            num_spatials = y_out[0].shape[0]

        # get number of influential neurons
        num_infl = infl_neurons.shape[0]

        # construct synapse weights and counts
        synapse_weights = sp.coo_matrix(
            (torch.ones(num_infl), (infl_neurons, infl_neurons)),
            shape=(num_spatials, num_spatials),
        )

        synapse_counts = sp.coo_matrix(
            (
                torch.ones(num_infl),
                (infl_neurons, infl_neurons),
            ),
            shape=(num_spatials, num_spatials),
            dtype=int,
        )

        # construct neuron counts by summing along columns of synapse counts
        neuron_counts = sp.coo_matrix(
            sp.csc_matrix(synapse_counts).sum(axis=0), dtype=int
        )

        return neuron_counts, synapse_counts, synapse_weights
