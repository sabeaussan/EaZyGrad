from __future__ import annotations

from typing import Any

import numpy as np
from .grad import operations, dag
from .utils import check

class _Tensor:
    """
    Dense tensor object used throughout EaZyGrad.

    `_Tensor` plays the same role as `torch.Tensor` in PyTorch: it stores
    array data, tracks whether gradients are required, and records graph edges
    when differentiable operations are applied.

    Notes
    -----
    This class is intentionally lightweight and educational. Most users should
    construct tensors through the factory functions in :mod:`eazygrad`, such as
    :func:`eazygrad.tensor` or :func:`eazygrad.from_numpy`, rather than calling
    `_Tensor` directly.

    See Also
    --------
    `PyTorch tensor docs <https://pytorch.org/docs/stable/tensors.html>`_
    """

    def __init__(self, array: Any, requires_grad: bool, dtype: Any = None) -> None:
        # /!\ Warning ! array is not copied if this __init__ function is used instead of tensor factories
        self._array = check.input_array_type(array, dtype)
        self.ndim = self._array.ndim
        self.dtype = self._array.dtype
        # Does not allow grad computation for integer tensors
        if requires_grad and not np.issubdtype(self.dtype, np.floating):
            raise TypeError("Only tensors with floating point dtype can require gradients.")
        self.requires_grad = requires_grad and dag.grad_enable
        # Grad atttributes : 
        # - acc_grad is a temporary buffer to compute backward
        # - grad contains the value used for gradient descent
        self.grad = None
        self.acc_grad = np.float32(0.0)
        # the node_id reflects the creation node of the tensor
        self.node_id = None

    def __len__(self) -> int:
        return self._array.shape[0]

    def __eq__(self, other: _Tensor) -> _Tensor:
        return _Tensor(self._array==other._array, requires_grad=False, dtype=np.bool)
    

    def __setitem__(self, key: Any, value: Any) -> None:
        # overloads the array[key]=value operator
        try:
            self._array[key] = value
        except ValueError as e:
            if "assignment destination is read-only" in str(e):
                raise RuntimeError("This tensor is read-only because it has been cached for the backward pass.")
            else:
                raise e

    def __getitem__(self, key: Any) -> _Tensor:
        # overloads the value=array[key] operator
        result = _Tensor(self._array[key], requires_grad=self.requires_grad)
        if self.requires_grad:
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Slice(shape=self._array.shape, key=key, dtype=result.dtype), result=result)
        return result

    def __add__(self, other: _Tensor | float | int) -> _Tensor:
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result_arr = np.add(self._array, other)
            result = _Tensor(result_arr, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Add(), result=result)
        elif isinstance(other, _Tensor):
            requires_grad = self.requires_grad or other.requires_grad
            result_arr = np.add(self._array, other._array)
            result = _Tensor(result_arr, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.Add(), result=result)
        else:
            raise RuntimeError(f"Other should be either a tensor or a scalar, got {type(other)}.")
        return result

    def __radd__(self, other: _Tensor | float | int) -> _Tensor:
        return self.__add__(other)

    def __sub__(self, other: _Tensor | float | int) -> _Tensor:
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(self._array - other, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Sub(), result=result)
        elif isinstance(other, _Tensor):
            requires_grad = self.requires_grad or other.requires_grad
            result = _Tensor(self._array - other._array, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.Sub(), result=result)
        else:
            raise RuntimeError(f"Other should be either a tensor or a scalar, got {type(other)}.")
        return result

    def __rsub__(self, other: _Tensor | float | int) -> _Tensor:
        return -1 * (self - other)

    def __neg__(self) -> _Tensor:
        return self.__mul__(-1)
    
    def __mul__(self, other: _Tensor | float | int) -> _Tensor:
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(self._array * other, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Mul(arr=self._array, scalar=other), result=result)
        elif isinstance(other, _Tensor):
            requires_grad = self.requires_grad or other.requires_grad
            result = _Tensor(self._array * other._array, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.Mul(arr1=self._array, arr2=other._array), result=result)
        else:
            raise RuntimeError(f"Other should be either a tensor or a scalar, got {type(other)}.")
        return result

    def __rmul__(self, other: _Tensor | float | int) -> _Tensor:
        return self.__mul__(other)

    def __truediv__(self, other: _Tensor | float | int) -> _Tensor:
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(self._array / other, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Div(arr=self._array, scalar=other), result=result)
        elif isinstance(other, _Tensor):
            requires_grad = self.requires_grad or other.requires_grad
            result = _Tensor(self._array / other._array, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.Div(arr1=self._array, arr2=other._array), result=result)
        else:
            raise RuntimeError(f"Other should be either a tensor or a scalar, got {type(other)}.")
        return result

    def __rtruediv__(self, other: float | int) -> _Tensor:
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(other / self._array, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.RDiv(arr=self._array, scalar=other), result=result)
        else:
            raise RuntimeError(f"Other should be a scalar, got {type(other)}.")
        return result

    def __pow__(self, other: float | int) -> _Tensor:
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(self._array ** other, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Pow(arr=self._array, exponent=other), result=result)
        else:
            raise RuntimeError(f"Other should be a scalar, got {type(other)}.")
        return result
    
    def matmul(self, other: _Tensor) -> _Tensor:
        """
        Matrix-multiply this tensor with another tensor.

        Parameters
        ----------
        other : _Tensor
            Right-hand side tensor.

        Returns
        -------
        _Tensor
            Result of the matrix multiplication.

        Notes
        -----
        This method follows NumPy's ``matmul`` broadcasting rules and requires
        both operands to be at least 1-dimensional.

        See Also
        --------
        `torch.matmul <https://pytorch.org/docs/stable/generated/torch.matmul.html>`_
        """
        if isinstance(other, _Tensor):
            if other._array.ndim==0 or self._array.ndim==0:
                raise RuntimeError(f"Both arguments to matmul need to be at least 1D, but got {len(other._array.shape)}D and {len(self._array.shape)}D.")
            requires_grad = self.requires_grad or other.requires_grad
            # Numpy handles all the broadcasting rules for matmul and different shape cases
            result_arr = np.matmul(self._array, other._array)
            result = _Tensor(result_arr, requires_grad=requires_grad)
            # Need to select the right operation depending on the shape of the two arrays
            if requires_grad:
                if self._array.ndim == 1 and other._array.ndim == 1:
                    # Inner product
                    result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.InnerProduct(arr1=self._array, arr2=other._array), result=result)
                else:
                    result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.MatMul(arr1=self._array, arr2=other._array), result=result)
            return result
        else:
            raise RuntimeError(f"Other should be a tensor, got {type(other)}.")

    def __matmul__(self, other: _Tensor) -> _Tensor:
        return self.matmul(other)

    def reshape(self, *shape: int) -> _Tensor:
        """
        Return a reshaped view of the tensor.

        Parameters
        ----------
        *shape : int
            Target shape. At most one dimension may be ``-1``.

        Returns
        -------
        _Tensor
            Reshaped tensor view.

        Notes
        -----
        The returned tensor shares storage with the input whenever NumPy can
        provide a view.

        See Also
        --------
        `torch.reshape <https://pytorch.org/docs/stable/generated/torch.reshape.html>`_
        """
        # Return a view of the input array with given shape
        # Share the same data buffer as the original Tensor
        result = _Tensor(self._array.reshape(*shape), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Reshape(shape=self._array.shape), result=result)
        return result

    def mean(self, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> _Tensor:
        """
        Compute the mean of the tensor along one or more axes.

        Parameters
        ----------
        dim : int or tuple of int, optional
            Axis or axes to reduce. If omitted, all dimensions are reduced.
        keepdims : bool, default=False
            Whether reduced dimensions are retained with size 1.

        Returns
        -------
        _Tensor
            Tensor containing the reduced mean.

        See Also
        --------
        `torch.mean <https://pytorch.org/docs/stable/generated/torch.mean.html>`_
        """
        # dim is int or tuple of ints
        if isinstance(dim, int):
            dim = (dim,)
        requires_grad = self.requires_grad
        if dim is None:
            dim = tuple(np.arange(len(self.shape)))
        # Forces type promotion to float64 for mean/sum ops
        result = _Tensor(
            self._array.mean(axis=dim, keepdims=keepdims, dtype=np.float64).astype(self.dtype), 
            requires_grad=requires_grad
        )
        if requires_grad:
            # avoid native python casting to float64
            # will be promoted to f64 if needed
            div_factor = np.float32(1 / np.prod([self.shape[ax] for ax in dim]))
            dtype = result.dtype
            result.node_id = dag.create_node(
                parents_id=[self.node_id],
                operation=operations.Mean(shape=self._array.shape, div_factor=div_factor, dim=dim, keepdims=keepdims, dtype=dtype),
                result=result,
            )
        return result

    def sum(self, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> _Tensor:
        """
        Compute the sum of the tensor along one or more axes.

        Parameters
        ----------
        dim : int or tuple of int, optional
            Axis or axes to reduce. If omitted, all dimensions are reduced.
        keepdims : bool, default=False
            Whether reduced dimensions are retained with size 1.

        Returns
        -------
        _Tensor
            Tensor containing the reduced sum.

        See Also
        --------
        `torch.sum <https://pytorch.org/docs/stable/generated/torch.sum.html>`_
        """
        if dim is None:
            # avoid backprop error is keepdims is True and dim = None
            dim = tuple(np.arange(len(self.shape)))
        requires_grad = self.requires_grad
        result = _Tensor(
            self._array.sum(axis=dim, keepdims=keepdims, dtype=np.float64).astype(self.dtype), 
            requires_grad=requires_grad
        )
        if requires_grad:
            dtype = result.dtype
            result.node_id = dag.create_node(
                parents_id=[self.node_id],
                operation=operations.Sum(shape=self._array.shape, dim=dim, keepdims=keepdims, dtype=dtype),
                result=result,
            )
        return result

    @property
    def shape(self) -> tuple[int, ...]:
        """tuple of int: Shape of the underlying tensor array."""
        return self._array.shape

    def unsqueeze(self, *dim: int) -> _Tensor:
        """
        Insert one or more singleton dimensions.

        Parameters
        ----------
        *dim : int
            Positions where singleton dimensions are inserted. If omitted, a
            singleton dimension is inserted at axis 0.

        Returns
        -------
        _Tensor
            Tensor with expanded dimensionality.

        See Also
        --------
        `torch.unsqueeze <https://pytorch.org/docs/stable/generated/torch.unsqueeze.html>`_
        """
        if len(dim) == 0:
            dim = 0
        result = _Tensor(np.expand_dims(self._array, axis=dim), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.ExpandDims(dim=dim), result=result)
        return result

    def squeeze(self, *dim: int) -> _Tensor:
        """
        Remove singleton dimensions from the tensor.

        Parameters
        ----------
        *dim : int, optional
            Specific singleton dimensions to remove. If omitted, all singleton
            dimensions are removed.

        Returns
        -------
        _Tensor
            Tensor with squeezed dimensionality.

        See Also
        --------
        `torch.squeeze <https://pytorch.org/docs/stable/generated/torch.squeeze.html>`_
        """
        if len(dim) == 0:
            dim = None
        result = _Tensor(np.squeeze(self._array, axis=dim), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Squeeze(dim=dim), result=result)
        return result

    def swapdims(self, dim1: int, dim2: int) -> _Tensor:
        """
        Swap two dimensions of the tensor.

        Parameters
        ----------
        dim1 : int
            First dimension.
        dim2 : int
            Second dimension.

        Returns
        -------
        _Tensor
            Tensor with the two dimensions exchanged.

        See Also
        --------
        `torch.swapdims <https://pytorch.org/docs/stable/generated/torch.swapdims.html>`_
        """
        result = _Tensor(np.swapaxes(self._array, dim1, dim2), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.SwapDims(dim1=dim1, dim2=dim2), result=result)
        return result

    def T(self) -> _Tensor:
        """
        Swap the last two dimensions of the tensor.

        Returns
        -------
        _Tensor
            Tensor with the trailing two axes transposed.
        """
        return self.swapdims(-1, -2)

    def clear_grad(self) -> None:
        """
        Clear the stored gradient of the tensor.

        Returns
        -------
        None
        """
        self.grad = None

    def numpy(self, force: bool = True) -> np.ndarray:
        """
        Return the tensor contents as a NumPy array copy.

        Parameters
        ----------
        force : bool, default=True
            Compatibility argument. Only ``True`` is supported.

        Returns
        -------
        numpy.ndarray
            Copy of the underlying tensor data.

        Notes
        -----
        Unlike `torch.Tensor.numpy()`, EaZyGrad always returns a copy and does
        not expose shared storage back to NumPy.
        """
        # Always returns a copy unlike pytorch
        if not force:
            raise NotImplementedError("Unlike Pytorch, always force a copy, no shared storage.")
        return self._array.copy()
    
    def float(self) -> _Tensor:
        """
        Cast the tensor to ``numpy.float32``.

        Returns
        -------
        _Tensor
            Tensor cast to ``float32``.
        """
        requires_grad = self.requires_grad
        result = _Tensor(self._array.astype(np.float32, copy=False), requires_grad=requires_grad)
        if requires_grad:
            result.node_id = dag.create_node(
                parents_id=[self.node_id], 
                operation=operations.Copy(dtype=self.dtype), 
                result=result
            )
        return result
    
    def double(self) -> _Tensor:
        """
        Cast the tensor to ``numpy.float64``.

        Returns
        -------
        _Tensor
            Tensor cast to ``float64``.
        """
        requires_grad = self.requires_grad
        result = _Tensor(self._array.astype(np.float64, copy=False), requires_grad=requires_grad)
        if requires_grad:
            result.node_id = dag.create_node(
                parents_id=[self.node_id], 
                operation=operations.Copy(dtype=self.dtype), 
                result=result
            )
        return result
    
    def int(self) -> _Tensor:
        """
        Cast the tensor to ``numpy.int32``.

        Returns
        -------
        _Tensor
            Tensor cast to ``int32``.
        """
        requires_grad = self.requires_grad
        result = _Tensor(self._array.astype(np.int32, copy=False), requires_grad=requires_grad)
        if requires_grad:
            result.node_id = dag.create_node(
                parents_id=[self.node_id], 
                operation=operations.Copy(dtype=self.dtype), 
                result=result
            )
        return result
    
    def long(self) -> _Tensor:
        """
        Cast the tensor to ``numpy.int64``.

        Returns
        -------
        _Tensor
            Tensor cast to ``int64``.
        """
        requires_grad = self.requires_grad
        result = _Tensor(self._array.astype(np.int64, copy=False), requires_grad=requires_grad)
        if requires_grad:
            result.node_id = dag.create_node(
                parents_id=[self.node_id], 
                operation=operations.Copy(dtype=self.dtype), 
                result=result
            )
        return result

    def to(self, dtype: Any) -> _Tensor:
        """
        Cast the tensor to a supported dtype.

        Parameters
        ----------
        dtype : numpy.dtype or type
            Target dtype. Supported values are ``numpy.float32``,
            ``numpy.float64``, ``numpy.int32``, and ``numpy.int64``.

        Returns
        -------
        _Tensor
            Tensor cast to the requested dtype.

        See Also
        --------
        `torch.Tensor.to <https://pytorch.org/docs/stable/generated/torch.Tensor.to.html>`_
        """
        # no-op
        if dtype == self.dtype:
            return self
        
        match dtype:
            case np.float32:
                return self.float()
            case np.float64:
                return self.double()
            case np.int32:
                return self.int()
            case np.int64:
                return self.long()
            case _:
                raise NotImplementedError(f"Unsupported dtype : {dtype}")
    
    def __array__(self, dtype: Any = None) -> np.ndarray:
        # Numpy array interface, to support `numpy.asarray(tensor) -> ndarray`
        if dtype is None:
            return self.numpy()
        else:
            return self.numpy().astype(dtype, copy=False)

    def detach(self) -> _Tensor:
        """
        Return a tensor detached from the computation graph.

        Notes
        -----
        This method is currently not implemented because the present graph
        ownership model can leak memory when detached tensors are created.
        """
        raise NotImplementedError("As implemented currently, it can lead to memory leaks.")
        return _Tensor(self._array, requires_grad=False)

    def backward(self, vector: np.ndarray | None = None, retain_graph: bool = False) -> None:
        """
        Backpropagate gradients from this tensor.

        Parameters
        ----------
        vector : numpy.ndarray, optional
            Gradient of a scalar objective with respect to this tensor. This is
            required when the tensor is not scalar.
        retain_graph : bool, default=False
            Whether to keep traversed graph nodes after the backward pass.

        Returns
        -------
        None

        Notes
        -----
        If ``vector`` is omitted, the tensor must be scalar and a gradient of
        1 is used.

        See Also
        --------
        `torch.Tensor.backward <https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html>`_
        """
        # vector is the gradient the gradient of the differentiated function w.r.t. self
        # Expect a numpy array of same shape and dtype
        if vector is None and not np.prod(self._array.shape) == 1:
            raise RuntimeError("Can't compute  propagation if root _Tensor is not a scalar and no vector are provided")
        elif vector is not None and self._array.shape != vector.shape:
            raise RuntimeError("Can't compute  propagation if root _Tensor and vector are not the same size")
        elif vector is not None:
            if not isinstance(vector, np.ndarray):
                raise TypeError("The vector passed to backward should be a numpy ndarray.")
            # TODO : à tester
            if vector.dtype != self.dtype:
                raise RuntimeError(f"The dtype of vector should match self.dtype, got {vector.dtype} instead of {self.dtype}")
            if vector.shape != self.shape:
                raise RuntimeError(f"The shape of vector should match self.shape, got {vector.shape} instead of {self.shape}")
            self.acc_grad = vector
        else:
            self.acc_grad = np.float32(1.0)
        dag.backward(self.node_id, retain_graph=retain_graph)

    def plot_dag(self, full_graph: bool = False) -> None:
        """
        Render the computation graph rooted at this tensor.

        Parameters
        ----------
        full_graph : bool, default=False
            Whether to render the full global graph. Only the rooted subgraph
            is currently supported.

        Returns
        -------
        None
        """
        dag.plot(self.node_id, full_graph)


    def __str__(self) -> str:
        if not self.requires_grad:
            return f"eazygrad.tensor({self._array.tolist()}, dtype={self.dtype})"
        else:
            return f"eazygrad.tensor({self._array.tolist()}, dtype={self.dtype}, requires_grad={self.requires_grad})"
    
    def __repr__(self) -> str:
        # For printing nested tensor
        if not self.requires_grad:
            return f"eazygrad.tensor({self._array.tolist()}, dtype={self.dtype})"
        else:
            return f"eazygrad.tensor({self._array.tolist()}, dtype={self.dtype}, requires_grad={self.requires_grad})"
