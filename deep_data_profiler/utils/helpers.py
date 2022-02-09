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
    s = k**2
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

        def submat(i, j):
            s = stride * i
            t = stride * j
            return temp[:, s : s + kernel, t : t + kernel]

        def batched_submat(i, j):
            return torch.stack(
                [submat(idx.item(), jdx.item()) for idx, jdx in zip(i, j)]
            )

    return batched_submat


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
