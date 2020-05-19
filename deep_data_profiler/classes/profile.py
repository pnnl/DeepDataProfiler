from collections import Counter, defaultdict
import numpy as np
import copy

class Profile:

    """Summary
    """
    
    def __init__(self,profile=None, neuron_counts=None,
                    synapse_counts=None,
                    synapse_weights=None,
                    num_inputs=0):
        """Summary
        
        Parameters
        ----------
        profile : ddp.Profile, optional
            for instantiating a new Profile using a previous profile
        neuron_counts : defaultdict(Counter()), optional
        synapse_counts : defaultdict(Counter()), optional
        synapse_weights : defaultdict(Counter()), optional
            these are the outputs from a path profiling method.
        num_inputs : int, optional
            Number of inputs represented by the profile

        Note
        ----
        The format for the inputs is very strict so that it can be used to store 
        the results of a profiling process but there is no type checking. If the
        input is not in the correct format, the metrics could fail or return inaccurate
        values.
        """
        if profile is not None:
            self._neuron_counts = copy.deepcopy(profile.neuron_counts)
            self._synapse_counts = copy.deepcopy(profile.synapse_counts)
            self._synapse_weights = copy.deepcopy(profile.synapse_weights)
            self._num_inputs = profile._num_inputs
        else :
            self._neuron_counts = neuron_counts or defaultdict(Counter)
            self._synapse_counts = synapse_counts or defaultdict(Counter)
            self._synapse_weights = synapse_weights or defaultdict(set)
            self._num_inputs = num_inputs 

    @property
    def neuron_counts(self):
        return self._neuron_counts

    @property
    def synapse_counts(self):
        return self._synapse_counts
    
    @property
    def synapse_weights(self):
        return self._synapse_weights
              
    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def total(self):
        return sum([sum(self._neuron_counts[layer].values()) for layer in self._neuron_counts])

    @property
    def size(self):
        return sum([len(set(self.neuron_counts[layer].keys())) for layer in self.neuron_counts])

    def __eq__(self,other):
        return bool(self._neuron_counts == other.neuron_counts and
                self._synapse_counts == other.synapse_counts and
                self._synapse_weights == other.synapse_weights and
                self._num_inputs == other.num_inputs)

    def __iter__(self):
        return self.neuron_counts.keys()

    def __add__(self,other):
        new_profile = Profile(profile=self)
        new_profile._num_inputs += other.num_inputs
        for layer in other.neuron_counts:
            new_profile._neuron_counts[layer].update(other.neuron_counts[layer])
            new_profile._synapse_counts[layer].update(other.synapse_counts[layer])
            new_profile._synapse_weights[layer].update(other.synapse_weights[layer])        
        return new_profile

    def __iadd__(self,other):
        """
        Adds in place the neuron_counts and synapse_sets of other to self.
        Add in place.
        
        Parameters
        ----------
        other : Profile
        
        Returns
        -------
        self : Profile
            
        """            
        for layer in other.neuron_counts:
            self._neuron_counts[layer].update(other.neuron_counts[layer])
            self._synapse_counts[layer].update(other.synapse_counts[layer])
            self._synapse_weights[layer].update(other.synapse_weights[layer])
        self._num_inputs += other.num_inputs
        return self


def jaccard_simple(set1,set2):
    """
    Computes the jaccard similarity of two sets = size of their 
    intersection / size of their union
    
    Parameters
    ----------
    set1 : set or iterable
    set2 : set or iterable
    
    Returns
    -------
     : float
    """
    if len(set1) == 0 or len(set2) == 0:
        return 0
    s1 = set(set1);s2 = set(set2)
    return len(s1 & s2)/len(s1 | s2)

def instance_jaccard(profile1, profile2, neuron=False):
    """
    Computes the proportion of synapses(or neurons/neurons) of profile1 that 
    belongs to profile2 synapses(or neurons/neurons)

    Parameters
    ----------
    profile1 : Profile 
        Typically a single image profile
    profile2 : Profile 
        Typically an aggregated profile of many images
    neuron : bool
        Set to True if wish to compute proportions in terms of neurons instead
         of synapses
    
    Returns
    -------
     : float
        The proportion of profile1 in profile2.
    """
    if profile1.num_inputs == 0 or profile2.num_inputs == 0:
        return 0
    if neuron:
        aprofile = set([(layer,neuron) for layer in profile1.neuron_counts for neuron in profile1.neuron_counts[layer] ])
        bprofile = set([(layer,neuron) for layer in profile2.neuron_counts for neuron in profile2.neuron_counts[layer] ])
    else:
        aprofile = set([(layer,synapse) for layer in profile1.synapse_counts for synapse in profile1.synapse_counts[layer] ])
        bprofile = set([(layer,synapse) for layer in profile2.synapse_counts for synapse in profile2.synapse_counts[layer] ])
    return len(aprofile & bprofile)/len(aprofile)

def avg_jaccard(profile1, profile2, neuron=False):
    """
    Computes the jaccard similarity at each layer using synapse sets (or 
    neuron sets) then averages the values.

    Parameters
    ----------
    profile1 : Profile
    profile2 : Profile
    neuron : bool, optional, default=False
        Set to true if wish to compute the iou on the neuron sets instead
        of the synapse sets
    
    Returns
    -------
     : float
        Mean Intersection-over-Union (IOU) across layers of synapse (neuron) sets
        in Profile object. The final logit layer is not considered in this 
        calculation.

    See also
    --------
    jaccard_simple
    """
    if profile1.num_inputs == 0 or profile2.num_inputs == 0:
        return 0
    iou = []
    layers = sorted(list(profile1.neuron_counts.keys()))[1:]
    if neuron:
        aprofile = profile1.neuron_counts
        bprofile = profile2.neuron_counts
    else:
        aprofile = {layer: { synapse for synapse in profile1.synapse_counts[layer] } for layer in layers}
        bprofile = {layer: { synapse for synapse in profile2.synapse_counts[layer] } for layer in layers}
    for layer in layers:            
        iou.append(jaccard_simple(aprofile[layer],bprofile[layer]))
    return np.mean(iou)

def jaccard(profile1, profile2, neuron=False):
    """
    Computes the jaccard similarity metric between two profiles using 
    the aggregation of all synapse sets (or neuron set across all layers
    
    Parameters
    ----------
    profile1 : Profile
    profile2 : Profile
    neuron : bool, optional, default=False
        Set to true if wish to compute the jaccard on the neuron sets instead
        of the synapse sets
    
    Returns
    -------
     : float

    See also
    --------
    jaccard_simple
    """
    if profile1.num_inputs == 0 or profile2.num_inputs == 0:
        return 0
    if neuron:
        aprofile = [(layer,neuron) for layer in profile1.neuron_counts for neuron in profile1.neuron_counts[layer] ]
        bprofile = [(layer,neuron) for layer in profile2.neuron_counts for neuron in profile2.neuron_counts[layer] ]
    else:
        aprofile = [(layer,synapse) for layer in profile1.synapse_counts for synapse in profile1.synapse_counts[layer] ]
        bprofile = [(layer,synapse) for layer in profile2.synapse_counts for synapse in profile2.synapse_counts[layer] ]
    return jaccard_simple(aprofile,bprofile)

def order_neuron_counts(profile):
    """
    Generates a dictionary keyed by layer pointing at a list of tuples
    (neuron,count) reverse ordered by count.
    
    Parameters
    ----------
    profile : Profile

    Returns
    -------
    count_dict : dict
    """
    count_dict = dict()
    for layer in profile.neuron_counts.keys():
        count_dict[layer] = sorted([(k,v) for k,v in profile.neuron_counts[layer].items()],key=lambda x : x[1],reverse=True)
    return count_dict

