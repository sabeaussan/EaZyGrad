from __future__ import annotations

from typing import Any

import numpy as np
from .grad import dag
from .utils import check
from ._tensor import _Tensor


def from_numpy(array: np.ndarray, requires_grad: bool = False) -> _Tensor:
    """
    Create a tensor that shares storage with an existing NumPy array.

    Parameters
    ----------
    array : numpy.ndarray
        Source array. The tensor reuses the same underlying storage.
    requires_grad : bool, default=False
        Whether to track operations on the returned tensor for automatic
        differentiation.

    Returns
    -------
    _Tensor
        Tensor view over ``array``.

    Raises
    ------
    TypeError
        If ``array`` is not a NumPy array.
    RuntimeError
        If ``array`` is read-only.

    Notes
    -----
    Mutations are shared between the NumPy array and the returned tensor.

    See Also
    --------
    `torch.from_numpy <https://pytorch.org/docs/stable/generated/torch.from_numpy.html>`_
    """
    # fast path to create tensor from numpy array without copy
    # /!\ Warning ! Shared storage, changes to the original array will be reflected in the tensor and vice versa
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input should be a np.ndarray but got {type(array)}.")
    
    if array.flags.writeable==False:
        raise RuntimeError("Writing to a tensor created from a read-only NumPy array is not supported.")

    new_tensor = _Tensor(array=array, requires_grad=requires_grad)
    if requires_grad:
        new_tensor.node_id = dag.create_node(parents_id=[None], operation=None, result=new_tensor, is_leaf=True)
    return new_tensor

def tensor(array: Any, requires_grad: bool = False, dtype: Any = None) -> _Tensor:
    """
    Create a tensor from a Python scalar, list, or NumPy array.

    Parameters
    ----------
    array : scalar, list, or numpy.ndarray
        Input data used to build the tensor.
    requires_grad : bool, default=False
        Whether to track operations on the returned tensor for automatic
        differentiation.
    dtype : numpy.dtype or type, optional
        Requested tensor dtype. If omitted, the dtype is inferred from the
        input.

    Returns
    -------
    _Tensor
        Newly created tensor.

    Notes
    -----
    Unlike :func:`from_numpy`, this constructor copies NumPy inputs.

    See Also
    --------
    `torch.tensor <https://pytorch.org/docs/stable/generated/torch.tensor.html>`_
    """
    # Instantiating tensor from numpy array copies the underlying buffer. For no-copy use from_numpy.
    new_tensor = _Tensor(array=array, requires_grad=requires_grad, dtype=dtype)
    if requires_grad:
        new_tensor.node_id = dag.create_node(parents_id=[None], operation=None, result=new_tensor, is_leaf=True)
    return new_tensor


def randn(*shape: int, requires_grad: bool = False, dtype: Any = np.float32) -> _Tensor:
    """
    Return a tensor filled with samples from the standard normal distribution.

    Parameters
    ----------
    *shape : int
        Output shape.
    requires_grad : bool, default=False
        Whether to track gradients for the returned tensor.
    dtype : numpy.dtype or type, default=numpy.float32
        Output dtype.

    Returns
    -------
    _Tensor
        Randomly initialized tensor.

    See Also
    --------
    `torch.randn <https://pytorch.org/docs/stable/generated/torch.randn.html>`_
    """
    if not isinstance(shape, tuple):
        raise TypeError(f"Expected a shape of type tuple, got {type(shape)} instead")
    if 0 in shape:
        raise ValueError("At least one of the dimension is empty")
    return tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)


def uniform(
    *shape: int,
    low: float = 0.0,
    high: float = 1.0,
    requires_grad: bool = False,
    dtype: Any = np.float32,
) -> _Tensor:
    """
    Return a tensor filled with samples from a uniform distribution.

    Parameters
    ----------
    *shape : int
        Output shape.
    low : float, default=0.0
        Lower bound of the distribution.
    high : float, default=1.0
        Upper bound of the distribution.
    requires_grad : bool, default=False
        Whether to track gradients for the returned tensor.
    dtype : numpy.dtype or type, default=numpy.float32
        Output dtype.

    Returns
    -------
    _Tensor
        Randomly initialized tensor.

    See Also
    --------
    `torch.rand <https://pytorch.org/docs/stable/generated/torch.rand.html>`_
    """
    if not isinstance(shape, tuple):
        raise TypeError(f"Expected a shape of type tuple, got {type(shape)} instead")
    if not check.is_scalar(low) or not check.is_scalar(high):
        raise TypeError(f"Wrong type for bounds, got low={type(low)}, high={type(high)}")
    if 0 in shape:
        raise ValueError("At least one of the dimension is empty")
    return tensor(np.random.uniform(low=low, high=high, size=shape).astype(dtype), requires_grad=requires_grad)


def ones(*shape: int, requires_grad: bool = False, dtype: Any = np.float32) -> _Tensor:
    """
    Return a tensor filled with ones.

    Parameters
    ----------
    *shape : int
        Output shape.
    requires_grad : bool, default=False
        Whether to track gradients for the returned tensor.
    dtype : numpy.dtype or type, default=numpy.float32
        Output dtype.

    Returns
    -------
    _Tensor
        Tensor filled with ones.

    See Also
    --------
    `torch.ones <https://pytorch.org/docs/stable/generated/torch.ones.html>`_
    """
    if not isinstance(shape, tuple):
        raise TypeError(f"Expected a shape of type tuple, got {type(shape)} instead")
    if 0 in shape:
        raise ValueError("At least one of the dimension is empty")
    return tensor(np.ones(shape).astype(dtype), requires_grad=requires_grad)


def zeros(*shape: int, requires_grad: bool = False, dtype: Any = np.float32) -> _Tensor:
    """
    Return a tensor filled with zeros.

    Parameters
    ----------
    *shape : int
        Output shape.
    requires_grad : bool, default=False
        Whether to track gradients for the returned tensor.
    dtype : numpy.dtype or type, default=numpy.float32
        Output dtype.

    Returns
    -------
    _Tensor
        Tensor filled with zeros.

    See Also
    --------
    `torch.zeros <https://pytorch.org/docs/stable/generated/torch.zeros.html>`_
    """
    if not isinstance(shape, tuple):
        raise TypeError(f"Expected a shape of type tuple, got {type(shape)} instead")
    if 0 in shape:
        raise ValueError("At least one of the dimension is empty")
    return tensor(np.zeros(shape).astype(dtype), requires_grad=requires_grad)
