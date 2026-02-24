import numpy as np
from .grad import dag
from .utils import check
from ._tensor import _Tensor


def from_numpy(array, requires_grad=False):
    # fast path to create tensor from numpy array without copy
    # /!\ Warning ! changes to the original array will be reflected in the tensor and vice versa
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input should be a np.ndarray but got {type(array)}.")
    
    if array.flags.writeable==False:
        raise RuntimeError("Writing to a tensor created from a read-only NumPy array is not supported.")

    new_tensor = _Tensor(array=array, requires_grad=requires_grad)
    if requires_grad:
        new_tensor.node_id = dag.create_node(parents_id=[None], operation=None, result=new_tensor, is_leaf=True)
    return new_tensor

def tensor(array, requires_grad=True, dtype=None):
    # Instantiating tensor from numpy array copies the underlying buffer. For no-copy use from_numpy.
    new_tensor = _Tensor(array=array, requires_grad=requires_grad, dtype=dtype)
    if requires_grad:
        new_tensor.node_id = dag.create_node(parents_id=[None], operation=None, result=new_tensor, is_leaf=True)
    return new_tensor


def randn(shape, requires_grad=False, dtype=np.float32):
    if not isinstance(shape, tuple):
        raise TypeError(f"Expected a shape of type tuple, got {type(shape)} instead")
    if 0 in shape:
        raise ValueError("At least one of the dimension is empty")
    return tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)


def uniform(low, high, shape, requires_grad=False, dtype=np.float32):
    if not isinstance(shape, tuple):
        raise TypeError(f"Expected a shape of type tuple, got {type(shape)} instead")
    if not check.is_scalar(low) or not check.is_scalar(high):
        raise TypeError(f"Wrong type for bounds, got low={type(shape)}, high={type(shape)}")
    if 0 in shape:
        raise ValueError("At least one of the dimension is empty")
    return tensor(np.random.uniform(low=low, high=high, size=shape).astype(dtype), requires_grad=requires_grad)


def ones(shape, requires_grad=False, dtype=np.float32):
    if not isinstance(shape, tuple):
        raise TypeError(f"Expected a shape of type tuple, got {type(shape)} instead")
    if 0 in shape:
        raise ValueError("At least one of the dimension is empty")
    return tensor(np.ones(shape).astype(dtype), requires_grad=requires_grad)


def zeros(shape, requires_grad=False, dtype=np.float32):
    if not isinstance(shape, tuple):
        raise TypeError(f"Expected a shape of type tuple, got {type(shape)} instead")
    if 0 in shape:
        raise ValueError("At least one of the dimension is empty")
    return tensor(np.zeros(shape).astype(dtype), requires_grad=requires_grad)



