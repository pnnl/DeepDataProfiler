import torch
import deep_data_profiler as ddp
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List


class TorchProfilerSpectral(ddp.TorchProfiler):
    def create_svd(
        self, layers_to_find: Optional[List[int]] = None
    ) -> Dict[int, Tuple[str, torch.svd]]:
        """
        Create a dictionary of the Singular Value Decomposition
        of a layer's weights.
        Args:
            layers_to_find (list, optional): Optional list of layers
                to find influential SVD neurons for.

        Returns
        -------
        svd_dict : dict
            A dictionary of SVD of weights, keyed by their position from
            the beginning of the model, according to the order the layers
            of the model are listed.
        """
        implemented_classes = [
            torch.nn.Linear,
            torch.nn.Conv2d,
        ]
        # ordered dictionary to feed to influential profiler
        svd_dict = OrderedDict()

        SGnodes = sorted(self.SG.nodes, key=lambda nd: nd[0])
        if layers_to_find is None:
            ltf = range(1, len(SGnodes))
        elif isinstance(layers_to_find, list):
            ltf = [
                lyr
                for lyr in layers_to_find
                if lyr >= 1 or lyr <= len(SGnodes) - 1
            ]
        else:  # a tuple is expected
            start = max(1, layers_to_find[0])
            end = min(len(SGnodes), layers_to_find[1])
            ltf = range(start, end)

        for ndx in ltf:
            nd = SGnodes[ndx]
            layertype = type(self.model.available_modules()[nd[1]])

            if layertype in implemented_classes:
                # grab the weights for the layer
                X = self.hooks[nd[1]]._parameters["weight"].detach()
                # if a Cond2d layer, 'unfold'
                if layertype is torch.nn.Conv2d:
                    X = torch.flatten(X, start_dim=1, end_dim=-1)
                # take SVD and put into dict
                svd = torch.svd(X, compute_uv=True)
                svd_dict[nd[0]] = (self.supernodes[nd[1]], svd.U)
        return svd_dict
