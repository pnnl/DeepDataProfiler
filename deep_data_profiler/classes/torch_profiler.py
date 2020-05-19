from collections import Counter, defaultdict, deque, OrderedDict
import concurrent.futures
import torch
import torch.nn as nn
import inspect
from torch.nn.modules import activation, dropout
import numpy as np
from deep_data_profiler import DeepDataProfilerError
import warnings
import traceback
from deep_data_profiler.classes.profile import Profile
from deep_data_profiler.utils import TorchHook, DDPCounter, submatrix_generator, get_index

class TorchProfiler():

    """
    Torch Profiler wraps a PyTorch model into a TorchHook model which can register activations
    as it evaluates data. Using the activations, inputs to the model may be profiled.

    Attributes
    ----------
    activation_classes : list
        List of activation classes in PyTorch
    dropout_classes : list
        List of dropout classes in PyTorch
    implemented_classes : list
        List of classes in PyTorch which have contrib functions implemented
    contrib_functions : list
        List of contrib functions ordered by their corresponding classes
        in implemented_classes
    model : torch.nn.Sequential()
        model to be profiled
    """

    def __init__(self, model, threshold=0.5, device=torch.device('cpu')):

        super().__init__()
        self.activation_classes = [m[1] for m in inspect.getmembers(activation, inspect.isclass) if
                                   m[1].__module__ == 'torch.nn.modules.activation']
        self.dropout_classes = [m[1] for m in inspect.getmembers(dropout, inspect.isclass) if
                                m[1].__module__ == 'torch.nn.modules.dropout']

        self.implemented_classes = [torch.nn.Linear,
                                    torch.nn.MaxPool2d,
                                    torch.nn.AdaptiveAvgPool2d,
                                    torch.nn.Conv2d]
        self.contrib_functions = ['_contrib_linear',
                                  '_contrib_max2d',
                                  '_contrib_adaptive_avg_pool2d',
                                  '_contrib_conv2d']

        self.model = TorchHook(model)
        self.hooks = self.model.available_modules()

    def create_layers(self, nlayers=0):
        """
        Create a dictionary of layers to profile.

        Parameters
        ----------
        nlayers : int, optional
            if 0 all available modules (except dropout) in the model will be used, otherwise up to n layers will
            be constructed.

        Returns
        -------
        layerdict : dict
            A dictionary of layers keyed by their position from the end of the model.
            The logit layer will be keyed 0.

        Note
        ----
        Because a weighted module is often followed by an activation module
        those modules are combined into a single layer to be profiled. As a consequence
        the number of layers generated will typically be less than the number of available
        modules in the model.
        When this method is called the modules used in the layers are given hooks so that their activations
        may be accessed.

        """
        hooks = self.hooks
        if nlayers == 0:
            # this will be greater than we need because activation layers will join with implemented layers.
            nlayers = len(hooks)
        namelist = set()

        layerdict = OrderedDict()

        revhooks = reversed(hooks)
        layeridx = DDPCounter(start=0)
        tmplyer = deque()
        for kdx in revhooks:
            if layeridx() == nlayers:
                tmplyer.appendleft(kdx)
                namelist.add(kdx)
                layeridx.inc()
                layerdict[layeridx()] = [list(tmplyer), 0]
                break
            this_type = type(hooks[kdx])
            if this_type in self.dropout_classes:
                continue
            elif this_type in self.activation_classes:
                tmplyer.appendleft(kdx)
                namelist.add(kdx)
                continue
            elif this_type in self.implemented_classes:
                tmplyer.appendleft(kdx)
                namelist.add(kdx)
                layeridx.inc()
                layerdict[layeridx()] = [list(tmplyer), self.contrib_functions[self.implemented_classes.index(this_type)]]
                tmplyer = deque()
                continue
            else:
                print(f'profiler not implemented for layer of type: {type(hooks[kdx])}')
                tmplyer.appendleft(kdx)
                namelist.add(kdx)
                layeridx.inc()
                layerdict[layeridx()] = [list(tmplyer), 0]
                break
        else:
            layeridx.inc()
            layerdict[layeridx()] = [0, 0]
        namelist = list(namelist)
        if len(namelist) > 0:
            self.model.add_hooks(namelist)
        return layerdict

    def _single_profile(self, x_in, y_out, ydx, layers, layerdict, ldx):
        func = getattr(self.__class__, layerdict[ldx][1])
        return func(self, x_in, y_out, ydx, layers)

    def create_profile(self, x, layerdict, n_layers=0, show_progress=False, parallel=False):
        """
        Generate a profile for a single input data x as it passes through the layers in layerdict

        Parameters
        ----------
        x : torch.Tensor
            input to model being profiled
        layerdict : dict
            keyed by index starting from the logit layer and working backwards, each layer contains at most one weighted module
        n_layers : int
            optional, number of layers to profile - useful if layerdict contains more layers than is wanted to profile
        show_progress : bool
            Will print the layer number currently being computed
        parallel : bool
            Set to True if wish to parallelize the profiling of many neurons.

        Returns
        -------
        profile.Profile
            profile contains neuron_counts, synapse_counts, and synapse_weights across layers in layerdict. Corresponding number of images = 1

        Note
        ----
        layerdict has n layers keyed 1 thru n
        layer n is the last layer which only holds the input for layer n-1
        below we create profiles starting with a layer 0 corresponding to the
        single neuron corresponding to the maximal logit in the last layer

        Parallel process uses concurrent.futures module.

        """
        with torch.no_grad():
            y, actives = self.model.forward(x)

            neuron_counts = defaultdict(Counter)
            synapse_counts = defaultdict(Counter)
            synapse_weights = defaultdict(set)

            # initialize profile with index of maximal logit from last layer
            neuron = tuple([int((torch.argmax(y[0])).detach().numpy())])
            neuron_counts[0].update([neuron])
            synapse_counts[0].update([(neuron, neuron, 0)])
            synapse_weights[0].update([(neuron, neuron, 1)])

            if n_layers == 0 or n_layers >= len(layerdict):
                n = len(layerdict)
            else:
                n = n_layers + 1
            for ldx in range(1, n):
                try:
                    if show_progress:
                        print(f'Layer #{ldx}')
                    # first retrieve x_in
                    inlayers, incontrib = layerdict[ldx + 1]
                    if incontrib == 0 and inlayers == 0:
                        x_in = x
                    else:
                        x_in = actives[inlayers[-1]]

                    # next retrieve y_out
                    layers, contrib = layerdict[ldx]
                    y_out = actives[layers[-1]]

                    if parallel == True:

                        with concurrent.futures.ProcessPoolExecutor() as executor:
                            results = [executor.submit(self._single_profile, x_in, y_out, ydx, layers, layerdict, ldx) for ydx in neuron_counts[ldx - 1]]

                            for f in concurrent.futures.as_completed(results):
                                nc, sc, sw = f.result()
                                neuron_counts[ldx].update(nc)
                                synapse_counts[ldx].update(sc)
                                synapse_weights[ldx].update(sw)
                    else:
                        for ydx in neuron_counts[ldx - 1]:
                            nc, sc, sw = self._single_profile(x_in, y_out, ydx, layers, layerdict, ldx)
                            neuron_counts[ldx].update(nc)
                            synapse_counts[ldx].update(sc)
                            synapse_weights[ldx].update(sw)

                except Exception as ex:
                    traceback.print_exc()
                    break

            return Profile(neuron_counts=neuron_counts,
                           synapse_counts=synapse_counts,
                           synapse_weights=synapse_weights, num_inputs=1)

        # return neuron_counts,synapse_counts,synapse_weights

    def _contrib_max2d(self, x_in, y_out, ydx, layer, threshold=None):
        '''
        Return the contributing synapse for a torch.nn.Max2D layer

        Parameters
        ----------
        x_in : torch.Tensor
            dimensions: batchsize,channels,height,width
        y_out : torch.Tensor
            dimensions: batchsize,channels,height,width
        ydx : tuple
            (channel,row,column) position of an output neuron
        layer : list(str)
            list containing single key in self.model.available_modules() dictionary
        threshold : None or float
            not used, placeholder for uniformity in arguments.

        Returns
        -------
        neuron_counts
        synapse_counts
        synapse_weights
        '''
        neuron_counts = Counter()
        synapse_counts = Counter()
        synapse_weights = set()
        maxpool = self.model.available_modules()[layer[0]]

        # Grab dimensions of maxpool from parameters
        stride = maxpool.stride
        kernel_size = maxpool.kernel_size

        if len(ydx) == 1:
            ch, i, j = get_index(ydx[0], kernel_size)
        else:
            ch, i, j = ydx

        xmat = submatrix_generator(x_in, stride, kernel_size)(i, j)[ch]

        c, v = max(list(enumerate(xmat.flatten())), key=lambda x: x[1])
        assert np.allclose(v.detach().numpy(), y_out[0, ch, i, j].detach().numpy(), rtol=1e-04, atol=1e-4), f'maxpool failure: {v - y_out[0,ch,i,j]}'

        neuron = (ch, c // kernel_size + stride * i, c % kernel_size + stride * j)
        neuron_counts.update([neuron])
        synapse_counts.update([(neuron, ydx)])
        synapse_weights.update([(neuron, ydx, 1)])  # 1 is just a placeholder since it is a direct contribution
        return neuron_counts, synapse_counts, synapse_weights

    def _contrib_adaptive_avg_pool2d(self, x_in, y_out, ydx, layer, threshold=0.1):
        '''
        Return the contributing synapses for a torch.nn.AdaptiveAveragePool layer

        Parameters
        ----------
        x_in : torch.Tensor
            dimensions: batchsize,channels,height,width
        y_out : torch.Tensor
            dimensions: batchsize,channels,height,width
        ydx : tuple
            (channel,row,column) position of an output neuron
        layer : list(str)
            list containing single key in self.model.available_modules() dictionary
        threshold : float

        Returns
        -------
        neuron_counts
        synapse_counts
        synapse_weights
        '''
        neuron_counts = Counter()
        synapse_counts = Counter()
        synapse_weights = set()
        avgpool = self.model.available_modules()[layer[0]]

        '''Grab the dimensions used by an adaptive pooling layer'''
        output_size = avgpool.output_size[0]
        input_size = x_in.shape[-1]
        stride = (input_size // output_size)
        kernel_size = input_size - (output_size - 1) * stride

        if len(ydx) == 1:
            ch, i, j = get_index(ydx[0], kernel_size)
        else:
            ch, i, j = ydx

        scalar = 1 / (kernel_size**2)  # multiplier for computing average
        goal = threshold * y_out[0, ch, i, j]

        xmat = submatrix_generator(x_in, stride, kernel_size)(i, j)[ch]
        ordsmat = sorted([(v * scalar, c) for c, v in enumerate(xmat.flatten()) if v != 0], key=lambda t: t[0], reverse=True)
        if len(ordsmat) > 0:
            cumsum = torch.cumsum(torch.Tensor([v[0] for v in ordsmat]), dim=0).detach()
            assert np.allclose(cumsum[-1].detach().numpy(), y_out[0, ch, i, j].detach().numpy(), rtol=1e-04, atol=1e-4), f'avgpool failure: {cumsum[-1] - y_out[0,ch,i,j]}'

            for idx, t in enumerate(cumsum):
                if t > goal:
                    break
            totalsum = cumsum[-1]
            for v, c in ordsmat[:idx + 1]:
                neuron = (ch, c // kernel_size + stride * i, c % kernel_size + stride * j)
                neuron_counts.update([neuron])
                synapse_counts.update([(neuron, ydx)])
                weight = round(float((v / totalsum).detach().numpy()), 6)
                synapse_weights.update([(neuron, ydx, weight)])
        return neuron_counts, synapse_counts, synapse_weights

    def _contrib_conv2d(self, x_in, y_out, ydx, layers, threshold=0.1):
        '''
        Profile a single output neuron from a 2d conv layer

        Pattern
        -------
        x_in : torch.tensor
            dimensions: batchsize,channels,height,width
        y_out : torch.Tensor
            dimensions: batchsize,channels,height,width
        ydx : tuple
            (channel,row,column) 3d position of an output neuron
        layers : list([str,str])
            list containing keys in self.model.available_modules() dictionary
            for conv2d these will refer to a convolutional module and an activation module
        threshold : float

        Returns
        -------
        neuron_counts
        synapse_counts
        synapse_weights

        Note
        ----
        Only implemented for convolution using filters with same height and width
        and strides equal in both dimensions and padding equal in all dimensions

        Synapse profiles for conv2d are indexed by 3 sets of tuples one for each neuron
        and on one for the index of the filter used.
        '''

        neuron_counts = Counter()
        synapse_counts = Counter()
        synapse_weights = set()
        conv, actf = layers
        conv = self.model.available_modules()[conv]
        actf = self.model.available_modules()[actf]

        # assumption is that kernel size, stride are equal in both dimensions
        # and padding preserves input size
        kernel_size = conv.kernel_size[0]
        stride = conv.stride[0]
        padding = conv.padding[0]
        W = conv._parameters['weight']
        B = conv._parameters['bias']

        if len(ydx) == 1:
            d, i, j = get_index(ydx[0], kernel_size)
        else:
            d, i, j = ydx

        # TODO make the lines below loop over a list of ydx positions
        try:
            y_true = y_out[0, d, i, j].detach().numpy()
        except:
            print(d, i, j)
        goal = threshold * y_true
        if goal <= 0:
            warnings.warn(f'output neuron at position {ydx} is less than 0')
        xmat = submatrix_generator(x_in, stride, kernel_size, padding=padding)(i, j)  # TODO generate sbmat before the loop

        z = torch.mul(W[d], xmat)
        ordsmat = sorted([(v, idx) for idx, v in enumerate(z.flatten()) if v != 0], key=lambda t: t[0], reverse=True)
        if len(ordsmat) > 0:
            cumsum = torch.cumsum(torch.Tensor([v[0] for v in ordsmat]), dim=0).detach() + B[d]
            values = [actf(v) for v in cumsum]
            assert np.allclose(values[-1].detach().numpy(), y_true, rtol=1e-04, atol=1e-4), f'conv2d failure: {values[-1].detach().numpy() - y_true}'

            for idx, v in enumerate(values):
                if v >= goal:
                    break
            totalsum = cumsum[-1] - B[d]  # this is the sum of all the values before bias and activation
            for v, jdx in ordsmat[:idx + 1]:
                wdx = get_index(jdx, kernel_size)
                neuron = tuple(np.array(wdx, dtype=int) + np.array((0, stride * i - padding, stride * j - padding), dtype=int))  # need the unpadded index
                neuron_counts.update([neuron])  # this shows where the index was in original input
                synapse_counts.update([(neuron, ydx, wdx)])
                weight = round(float((v / totalsum).detach().numpy()), 6)
                synapse_weights.update([(neuron, ydx, weight)])
        return neuron_counts, synapse_counts, synapse_weights

    def _contrib_linear(self, x_in, y_out, ydx, layers, threshold=0.1):
        '''
        Profile a single output neuron from a linear layer

        Pattern
        -------
        x_in : torch.tensor
            2dimensions: batchsize,i
        y_out : torch.Tensor
            2dimensions: batchsize,j
        ydx : tuple
            1d position of an output neuron
        layers : list()
            list containing keys in self.model.available_modules() dictionary
            for linear layer these will refer to a linear module and an activation module
            or just a linear module
        threshold : float

        Returns
        -------
        neuron_counts
        synapse_counts
        synapse_weights
        '''
        j = ydx[0]
        neuron_counts = Counter()
        synapse_counts = Counter()
        synapse_weights = set()

        if len(layers) == 1:
            linear = layers[0]
            def actf(x): return x
        else:
            linear, actf = layers
            actf = self.model.available_modules()[actf]
        linear = self.model.available_modules()[linear]

        W = linear._parameters['weight']
        B = linear._parameters['bias']

        # TODO make the lines below loop over a list of ydx positions
        goal = threshold * y_out[0, j]
        if goal <= 0:
            warnings.warn(f'output neuron at position {ydx} is less than 0')
        xdims = x_in[0].shape
        if len(xdims) > 1:
            holdx = torch.Tensor(x_in)
            x_in = x_in[0].flatten().unsqueeze(0)
        z = x_in[0] * W[j]

        # Confirm we haven't changed anything
        y_predict = (torch.sum(z) + B[j]).detach().numpy()
        y_true = y_out[0, j].detach().numpy()
        assert np.allclose(y_predict, y_true, rtol=1e-04, atol=1e-4), f'linear failed {y_predict-y_true} {layers}'

        ordsmat = sorted([(v, idx) for idx, v in enumerate(z) if v != 0], key=lambda t: t[0], reverse=True)
        if len(ordsmat):
            cumsum = torch.cumsum(torch.Tensor([v[0] for v in ordsmat]), dim=0).detach() + B[j]
            values = [actf(v) for v in cumsum]
            assert np.allclose(values[-1].detach().numpy(), y_true, rtol=1e-04, atol=1e-4), f'linear failed[2]: {values[-1].detach().numpy() - y_true}'

            for idx, v in enumerate(values):
                if v >= goal:
                    break
            totalsum = cumsum[-1] - B[j]
            for v, jdx in ordsmat[:idx + 1]:
                if len(xdims) > 1:
                    neuron = get_index(jdx, xdims[-1])  # is this producing 3 tuple?
                else:
                    neuron = tuple([jdx])
                neuron_counts.update([neuron])  # this shows where the index was in original input
                synapse_counts.update([(neuron, ydx)])
                weight = round(float((v / totalsum).detach().numpy()), 6)
                synapse_weights.update([(neuron, ydx, weight)])
        return neuron_counts, synapse_counts, synapse_weights
