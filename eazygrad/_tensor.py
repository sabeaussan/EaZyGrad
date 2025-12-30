import numpy as np
from .grad import operations, dag
from .utils import check


class _Tensor:

    def __init__(self, array, requires_grad, dtype=None):
        # TODO : Warning ! array is not copied if this __init__ function is used instead of tensor factory
        self._array = check.input_array_type(array, dtype)
        self.ndim = self._array.ndim
        self.dtype = self._array.dtype
        self.requires_grad = requires_grad
        self.grad = None
        # the node_id reflects the creation node of the tensor
        self.node_id = None # /!\ TODO : what happens if the tensor belongs to multiple nodes ?-> node can be parent of multiple nodes, but child of only one

    def __len__(self):
        return self._array.shape[0]
    
    # TODO : à tester
    def __setitem__(self, key, value):
        try:
            self._array[key] = value
        except ValueError as e:
            if "assignment destination is read-only" in str(e):
                raise RuntimeError("This tensor is read-only because it has been cached for the backward pass.")
            else:
                raise e

    def __getitem__(self, key):
        result = _Tensor(self._array[key], requires_grad=self.requires_grad)
        result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Slice(self._array.shape, key), result=result)
        return result

    def __add__(self, other):
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(self._array + other, requires_grad=requires_grad)
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Add(self._array, other), result=result)
        elif isinstance(other, _Tensor):
            requires_grad = self.requires_grad or other.requires_grad
            result = _Tensor(self._array + other._array, requires_grad=requires_grad)
            result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.Add(self._array.shape, other._array.shape), result=result)
        else:
            raise NotImplementedError
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(self._array - other, requires_grad=requires_grad)
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Sub(self._array, other), result=result)
        elif isinstance(other, _Tensor):
            requires_grad = self.requires_grad or other.requires_grad
            result = _Tensor(self._array - other._array, requires_grad=requires_grad)
            result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.Sub(self._array.shape, other._array.shape), result=result)
        else:
            raise NotImplementedError
        return result

    def __rsub__(self, other):
        return -1 * (self - other)

    def __neg__(self):
        return self.__mul__(-1)

    def __mul__(self, other):
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(self._array * other, requires_grad=requires_grad)
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Mul(self._array, other), result=result)
        elif isinstance(other, _Tensor):
            requires_grad = self.requires_grad or other.requires_grad
            result = _Tensor(self._array * other._array, requires_grad=requires_grad)
            result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.Mul(self._array, other._array), result=result)
        else:
            raise NotImplementedError
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(self._array / other, requires_grad=requires_grad)
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Div(self._array, other), result=result)
        elif isinstance(other, _Tensor):
            requires_grad = self.requires_grad or other.requires_grad
            result = _Tensor(self._array / other._array, requires_grad=requires_grad)
            result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.Div(self._array, other._array), result=result)
        else:
            raise NotImplementedError
        return result

    def __rtruediv__(self, other):
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(other / self._array, requires_grad=requires_grad)
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.RDiv(self._array, other), result=result)
        elif isinstance(other, _Tensor):
            raise NotImplementedError
        else:
            raise NotImplementedError
        return result

    def __pow__(self, other):
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(self._array ** other, requires_grad=requires_grad)
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Pow(self._array, other), result=result)
        elif isinstance(other, _Tensor):
            raise NotImplementedError("Raising a _Tensor to a power of type _Tensor is not supported, feel free to implement that ;)")
        else:
            raise NotImplementedError
        return result

    def matmul(self, other):
        if check.is_scalar(other):
            raise RuntimeError("Can't apply matrix multiplication with a scalar")
        elif isinstance(other, _Tensor):
            if len(other._array.shape)==0 or len(self._array.shape)==0:
                raise RuntimeError(f"Both arguments to matmul need to be at least 1D, but got {len(other._array.shape)}D and {len(self._array.shape)}D.")
            requires_grad = self.requires_grad or other.requires_grad
            result = _Tensor(np.matmul(self._array, other._array), requires_grad=requires_grad)
            result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.MatMul(self._array, other._array), result=result)
            return result
        else:
            raise RuntimeError("Expected a _Tensor")

    def __matmul__(self, other):
        return self.matmul(other)

    def reshape(self, *shape):
        # TODO : does it copy the array ?
        result = _Tensor(self._array.reshape(shape), requires_grad=self.requires_grad)
        result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Reshape(self._array.shape), result=result)
        return result

    def mean(self, dim = None, keepdims = False):
        # dim is int or tuple of ints
        if isinstance(dim, int):
            dim = (dim,)
        requires_grad = self.requires_grad
        if dim is None:
            dim = tuple(np.arange(len(self.shape)))
        result = _Tensor(
            self._array.astype(np.float64, copy=False).mean(axis=dim, keepdims=keepdims).astype(np.float32), 
            requires_grad=requires_grad
        )
        div_factor = 1 / np.prod([self.shape[ax] for ax in dim])
        result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Mean(self._array.shape, div_factor, (dim, keepdims)), result=result)
        return result

    def sum(self, dim = None, keepdims = False):
        if dim is None:
            # avoid backprop error is keepdims is True and dim = None
            dim = tuple(np.arange(len(self.shape)))
        requires_grad = self.requires_grad
        result = _Tensor(
            self._array.astype(np.float64, copy=False).sum(axis=dim, keepdims=keepdims).astype(np.float32), 
            requires_grad=requires_grad
        )
        result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Sum(self._array.shape, (dim, keepdims)), result=result)
        return result

    @property
    def shape(self):
        return self._array.shape

    def expand_dims(self, *dim):
        result = _Tensor(np.expand_dims(self._array, axis=dim), requires_grad=self.requires_grad)
        result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.ExpandDims(dim), result=result)
        return result

    def squeeze(self, *dim):
        result = _Tensor(np.squeeze(self._array, axis=dim), requires_grad=self.requires_grad)
        result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Squeeze(dim), result=result)
        return result

    def swapdims(self, dim1, dim2):
        result = _Tensor(np.swapaxes(self._array, dim1, dim2), requires_grad=self.requires_grad)
        result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.SwapDims(dim1, dim2), result=result)
        return result

    def T(self):
        return self.swapdims(-1, -2)

    def clear_grad(self) -> None:
        self.grad = None

    def numpy(self):
        return self._array.copy()
    
    def float(self):
        requires_grad = self.requires_grad
        result = _Tensor(self._array.astype(np.float32, copy=False), requires_grad=requires_grad)
        if requires_grad:
            result.node_id = dag.create_node(
                parents_id=[self.node_id], 
                operation=operations.Copy(self.dtype), 
                result=result
            )
        return result
    
    def double(self):
        requires_grad = self.requires_grad
        result = _Tensor(self._array.astype(np.float64, copy=False), requires_grad=requires_grad)
        if requires_grad:
            result.node_id = dag.create_node(
                parents_id=[self.node_id], 
                operation=operations.Copy(self.dtype), 
                result=result
            )
        return result
    
    def detach(self):
        # TODO : if node does not require grad, node is not in the graph ?
        return _Tensor(self._array.copy(), requires_grad=False)

    def backward(self, vector = None, retain_graph = False):
        # TODO : vector is numpy array, why ?
        # TODO : vector and array should not have the same shape ?
        if vector is None and not np.prod(self._array.shape) == 1:
            raise RuntimeError("Can't compute  propagation if root _Tensor is not a scalar and no vector are provided")
        elif vector is not None and self._array.shape != vector.shape:
            raise RuntimeError("Can't compute  propagation if root _Tensor and vector are not the same size")
        elif vector is not None:
            if not isinstance(vector, np.ndarray):
                raise TypeError("The vector passed to backward should be a numpy ndarray.")
            self.grad = vector
        else:
            self.grad = None
        dag.backward(self.node_id, retain_graph=retain_graph)

    def __str__(self):
        return self._array.__str__()
    
    def __repr__(self):
        # For printing tensor nested
        return self._array.__repr__()
