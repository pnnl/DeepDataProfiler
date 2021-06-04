import numpy as np
import scipy.sparse as sp
import torch
from typing import Callable, Tuple, Union
from collections import OrderedDict

# __all__ = ['DDPCounter', 'get_index', 'submatrix_generator', 'DefaultOrderedDict']


class DDPCounter:
    """
    Useful singleton class for keeping track of layers
    """

    def __init__(self, start=0, inc=1):
        self.counter = start
        self._inc = inc

    def inc(self):
        self.counter += self._inc
        return self.counter

    def __call__(self):
        return self.counter


# Helper Functions


def get_index(
    b: torch.Tensor, k: int, first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return 3dim index for a flattened tensor built from kxk planes

    Parameters
    ----------
    b : torch.Tensor
        1D indices
    k : int
        kernel size
    first : bool, optional, default=True
        if True then output will be channel,row,column tuple
        otherwise output will be row,column,channel tuple

    """
    s = k ** 2
    ch = b // s
    r = (b % s) // k
    c = (b % s) % k
    if first:
        return ch, r, c
    else:
        return r, c, ch


def submatrix_generator(
    x_in: Union[np.ndarray, torch.Tensor], stride: int, kernel: int, padding: int = 0
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Returns a function which creates the subtensor of x_in used to compute value of output at i,j index

    Parameters
    ----------
    x_in : numpy.ndarray or torch.Tensor
        dimensions assumed reference: channel, row, column
    stride : int
    kernel : int
        dimension of a square plane of filter or pooling size.
    padding : int, optional, default=0
        padding is assumed equal on all four dimensions

    Returns
    -------
    submat : function
    """
    with torch.no_grad():
        if padding > 0:
            # xp = torch.nn.functional.pad(x_in[:, :, :, :], (padding, padding, padding, padding), value=0)
            xp = torch.nn.functional.pad(
                x_in, (padding, padding, padding, padding), value=0
            )
        else:
            xp = x_in
        if len(xp.shape) == 4:
            temp = xp.squeeze(0)
        elif len(xp.shape) == 3:
            temp = xp
        else:
            print(
                "submatrix_generator not implemented for x_in dimensions other than 3 or 4"
            )
            return None

        def submat(i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:

            num_idxs = len(i)
            num_ch = temp.shape[0]

            # vector with repeating entries 0 through kernel-1
            # we add this to i and j to index each element in the kernel x kernel submatrix
            k = torch.arange(kernel).repeat(num_idxs)

            # vector with kernel*kernel entries of each channel, repeated num_idxs times
            # used to take a tubelike slice through all channels for each submatrix
            ch = (
                torch.arange(num_ch).repeat_interleave(kernel * kernel).repeat(num_idxs)
            )

            # get the idx of each row in the submatrix
            # this is stride*i:stride*i+kernel for each idx in i
            i = torch.repeat_interleave(stride * i, kernel) + k
            # repeat each row idx kernel times (to match pairwise with every col idx)
            i = i.repeat_interleave(kernel)
            # reshape to repeat each submatrix's list of kernel^2 row idxs once for each channel
            # (in order to take the receptive field through all channels)
            i = i.view(num_idxs, kernel * kernel).repeat(1, num_ch).view(-1)

            # get the idx of each col in the submatrix
            # this is stride*j:stride*j+kernel for each idx in j
            j = torch.repeat_interleave(stride * j, kernel) + k
            # reshape to repeat each submatrix's list of kernel col idxs once for each row idx and channel
            # (pairwise match with each row idx for all num_ch copies of the list of row idxs)
            j = j.view(num_idxs, kernel).repeat(1, kernel * num_ch).view(-1)

            # return submatrices as #submatrices x #channels x kernel x kernel
            return temp[ch, i, j].view(num_idxs, num_ch, kernel, kernel)

    return submat


def matrix_convert(x: torch.Tensor) -> sp.coo_matrix:
    """
    Converts any neuron/synapse weight/count to a 2D scipy sparse matrix

    Parameters
    ----------
    x : torch.Tensor
    """
    dims = x.shape
    # element-wise synapse count/weight
    if x.is_sparse and len(dims) == 6:
        x = x.coalesce()
        idx = x.indices()
        # get flat index of y_out neuron
        row = np.ravel_multi_index(
            (idx[0], idx[1], idx[2]), (dims[0], dims[1], dims[2])
        )
        # get flat index of x_in neuron
        col = np.ravel_multi_index(
            (idx[3], idx[4], idx[5]), (dims[3], dims[4], dims[5])
        )
        return sp.coo_matrix(
            (x.values(), (row, col)),
            shape=(dims[0] * dims[1] * dims[2], dims[3] * dims[4] * dims[5]),
        )
    # 3dim: element-wise neuron count/weight
    # 4dim: element-wise synapse count/weight between linear and conv
    if len(dims) == 3 or len(dims) == 4:
        x = x.view(dims[0], -1)

    return sp.coo_matrix(x)


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if default_factory is not None and not isinstance(default_factory, Callable):
            raise TypeError("first argument must be callable")
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = (self.default_factory,)
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy

        return type(self)(self.default_factory, copy.deepcopy(self.items()))

    def __repr__(self):
        return "OrderedDefaultDict(%s, %s)" % (
            self.default_factory,
            OrderedDict.__repr__(self),
        )


def get_children(model: torch.nn.Module):
    # just get the children, with no regard
    # for the ordering
    children = list(model.children())
    all_ops = []
    if children == []:
        # if no children, just model
        return model
    else:
        # recursively search through children,
        # as in seq_graph()
        for c in children:
            if len(list(c.named_children())) == 0:
                all_ops.append(get_children(c))
            else:
                all_ops.extend(get_children(c))
    return all_ops
