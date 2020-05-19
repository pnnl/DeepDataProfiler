
import torch

class DDPCounter:
    '''
    Useful singleton class for keeping track of layers
    '''
    def __init__(self,start=0,inc=1):
        self.counter = start
        self._inc = inc

    def inc(self):
        self.counter += self._inc
        return self.counter

    def __call__(self):
        return self.counter

###### Helper Functions

def get_index(b,k,first=True):	
    '''
    Return 3dim index for a flattened tensor built from kxk planes
    
    Parameters
    ----------
    b : int
        1D index
    k : int
        kernel size
    first : bool
        if True then output will be channel,row,column tuple
        otherwise output will be row,column,channel tuple

    '''
    s = k**2
    ch = int(b//s)
    r = int((b%s)//k)
    c = int((b%s)%k)
    if first:
        return ch,r,c
    else:
        return r,c,ch

def submatrix_generator(x_in,stride,kernel,padding=0):  
    '''
    Returns a function which creates the subtensor of x_in used to compute value of output at i,j index
    
    Parameters
    ----------
    x_in : numpy.ndarray or torch.Tensor
        dimensions assumed reference: channel, row, column
    stride : int
    kernel : int
        dimension of a square plane of filter or pooling size.
    padding : int
        padding is assumed equal on all four dimensions
    '''
    if padding >0:
        xp = torch.nn.functional.pad(x_in[:,:,:,:],(padding,padding,padding,padding),value=0)
    else:
        xp = x_in
    if len(xp.shape) == 4:
        temp = xp.squeeze(0)
    elif len(xp.shape) == 3:
        temp = xp
    else:
        print('submatrix_generator not implemented for x_in dimensions other than 3 or 4')
        return None
    def submat(i,j):
        s = stride*i
        t = stride*j
        return temp[:,s:s+kernel,t:t+kernel]
    return submat


