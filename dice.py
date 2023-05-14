import torch
import numpy as np
import itertools
from typing import Union

def is_integer(x) -> bool:
    '''
    Checks whether x is a python, torch, or numpy integers. 
    '''
    return type(x) in [int, torch.int8, torch.int16, torch.int32, torch.int64,
                       np.int8, np.int16, np.int32, np.int64]

def is_1d_iterable(x) -> bool:
    '''
    Checks whether x is an iterable, a 1-dimensional torch.Tensor, or numpy.ndarray.
    '''
    return (type(x) in [list, tuple, range, set] 
            or type(x) in [torch.Tensor, np.ndarray] and len(x.shape) == 1)

def is_0d_iterable(x) -> bool:
    '''
    Checks whether x is a 0-dimensional torch.Tensor, or numpy.ndarray.
    '''
    return type(x) in [np.ndarray, torch.Tensor] and len(x.shape) == 0

def _iterable_to_tuple(x) -> tuple:
    '''
    Takes a 0/1-D iterable, or an integer and changes it into a tuple.
    '''
    if is_1d_iterable(x): 
        x = tuple(torch.as_tensor(x).tolist())
        for item in x:
            if not is_integer(item):
                raise RuntimeError('each item of input should be integer')
    elif is_0d_iterable(x):
        if is_integer(x.item()): x = x.item(),
        else: raise RuntimeError('each item of input should be integer')
    elif is_integer(x): x = tuple(torch.as_tensor([x]).tolist())
    else: raise RuntimeError('input should be integer or 1-D iterable of integers')
    return x

def dice(tensor:torch.Tensor, length:Union[int, torch.Tensor, list, tuple], 
         dicing_dims:Union[int, torch.Tensor, list, tuple, None]=None,
         clone:bool=True, return_slice:bool=False) -> tuple:
    '''
    dices a tensor. 
    returns a tuple of diced tensors, or a tuple filled with tuples of slice objects.
    '''
    tensor = tensor.clone() if clone else tensor
    dims = torch.arange(len(tensor.shape))
    if dicing_dims is None: dicing_dims = dims
    else: dicing_dims = _iterable_to_tuple(dicing_dims)
    length = list(_iterable_to_tuple(length))
    if not (len(length) == 1 or len(length) == len(dicing_dims)):
        raise RuntimeError('number of length arg and dicing_dims should be same' \
                           + ' or length is integer.')
    if len(length) == 1: length = [length[0]] * len(dicing_dims)
    for dim in dims:
        if dim not in dicing_dims: length.insert(dim, tensor.shape[dim])
    length = torch.tensor(length).type(torch.int)
    dice_nums = torch.ceil(torch.tensor(tensor.shape) / length).type(torch.int)
    slicer = list()
    for dim in dims:
        slice_idxs = (torch.arange(dice_nums[dim]) * length[dim]).tolist() + [None]
        slicer.append(
            [slice(i, j) for i, j in zip(slice_idxs[:-1], slice_idxs[1:])])
    slices = tuple(itertools.product(*slicer))
    return slices if return_slice else tuple([tensor[slice_] for slice_ in slices])
