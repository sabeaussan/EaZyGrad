import numpy as np
from .grad import operations, dag
from .utils import check


class _Tensor:

    def __init__(self, array, requires_grad, dtype=None):
        # /!\ Warning ! array is not copied if this __init__ function is used instead of tensor factories
        self._array = check.input_array_type(array, dtype)
        self.ndim = self._array.ndim
        self.dtype = self._array.dtype
        if requires_grad and not np.issubdtype(dtype, np.floating):
            raise TypeError("Only tensors with floating point dtype can require gradients.")
        self.requires_grad = requires_grad and dag.grad_enable
        self.grad = None
        # the node_id reflects the creation node of the tensor
        self.node_id = None

    def __len__(self):
        return self._array.shape[0]

    def __eq__(self, other):
        return _Tensor(self._array==other._array, requires_grad=False, dtype=np.bool)
    

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
        if self.requires_grad:
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Slice(shape=self._array.shape, key=key, dtype=result.dtype), result=result)
        return result

    def __add__(self, other, out=None):
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            if out is not None:
                np.add(self._array, other, out=out)
                result_arr = out
            else:
                result_arr = np.add(self._array, other)
            result = _Tensor(result_arr, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Add(arr=self._array.shape, dtype=result.dtype), result=result)
        elif isinstance(other, _Tensor):
            requires_grad = self.requires_grad or other.requires_grad
            if out is not None:
                np.add(self._array, other._array, out=out)
                result_arr = out
            else:
                result_arr = np.add(self._array, other._array)
            result = _Tensor(result_arr, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.Add(shape1=self._array.shape, shape2=other._array.shape, dtype=result.dtype), result=result)
        else:
            raise RuntimeError(f"Other should be either a tensor or a scalar, got {type(other)}.")
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(self._array - other, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Sub(arr=self._array.shape, dtype=result.dtype), result=result)
        elif isinstance(other, _Tensor):
            requires_grad = self.requires_grad or other.requires_grad
            result = _Tensor(self._array - other._array, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id, other.node_id], operation=operations.Sub(shape1=self._array.shape, shape2=other._array.shape, dtype=result.dtype), result=result)
        else:
            raise RuntimeError(f"Other should be either a tensor or a scalar, got {type(other)}.")
        return result

    def __rsub__(self, other):
        return -1 * (self - other)

    def __neg__(self):
        return self.__mul__(-1)
    
    def __mul__(self, other):
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

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
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

    def __rtruediv__(self, other):
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(other / self._array, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.RDiv(arr=self._array, scalar=other), result=result)
        else:
            raise RuntimeError(f"Other should be a scalar, got {type(other)}.")
        return result

    def __pow__(self, other):
        if check.is_scalar(other):
            requires_grad = self.requires_grad
            result = _Tensor(self._array ** other, requires_grad=requires_grad)
            if requires_grad:
                result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Pow(arr=self._array, exponent=other), result=result)
        else:
            RuntimeError(f"Other should be a scalar, got {type(other)}.")
        return result
    

    def matmul(self, other, out=None):
        if isinstance(other, _Tensor):
            if other._array.ndim==0 or self._array.ndim==0:
                raise RuntimeError(f"Both arguments to matmul need to be at least 1D, but got {len(other._array.shape)}D and {len(self._array.shape)}D.")
            requires_grad = self.requires_grad or other.requires_grad
            # Numpy handles all the broadcasting rules for matmul and different shape cases
            if out is not None:
                # out is a numpy array 
                np.matmul(self._array, other._array, out=out)
                result_arr = out
            else:
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

    def __matmul__(self, other):
        return self.matmul(other)

    def reshape(self, *shape):
        # Return a view of the input array with given shape
        # Share the same data buffer as the original Tensor
        result = _Tensor(self._array.reshape(shape), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Reshape(shape=self._array.shape), result=result)
        return result

    def mean(self, dim = None, keepdims = False):
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

    def sum(self, dim = None, keepdims = False):
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
    def shape(self):
        return self._array.shape

    def unsqueeze(self, *dim):
        if len(dim) == 0:
            dim = 0
        result = _Tensor(np.expand_dims(self._array, axis=dim), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.ExpandDims(dim=dim), result=result)
        return result

    def squeeze(self, *dim):
        if len(dim) == 0:
            dim = None
        result = _Tensor(np.squeeze(self._array, axis=dim), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.Squeeze(dim=dim), result=result)
        return result

    def swapdims(self, dim1, dim2):
        result = _Tensor(np.swapaxes(self._array, dim1, dim2), requires_grad=self.requires_grad)
        if self.requires_grad:
            result.node_id = dag.create_node(parents_id=[self.node_id], operation=operations.SwapDims(dim1=dim1, dim2=dim2), result=result)
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
                operation=operations.Copy(dtype=self.dtype), 
                result=result
            )
        return result
    
    def double(self):
        requires_grad = self.requires_grad
        result = _Tensor(self._array.astype(np.float64, copy=False), requires_grad=requires_grad)
        if requires_grad:
            result.node_id = dag.create_node(
                parents_id=[self.node_id], 
                operation=operations.Copy(dtype=self.dtype), 
                result=result
            )
        return result
    
    def int(self):
        requires_grad = self.requires_grad
        result = _Tensor(self._array.astype(np.int32, copy=False), requires_grad=requires_grad)
        if requires_grad:
            result.node_id = dag.create_node(
                parents_id=[self.node_id], 
                operation=operations.Copy(dtype=self.dtype), 
                result=result
            )
        return result
    
    def long(self):
        requires_grad = self.requires_grad
        result = _Tensor(self._array.astype(np.int64, copy=False), requires_grad=requires_grad)
        if requires_grad:
            result.node_id = dag.create_node(
                parents_id=[self.node_id], 
                operation=operations.Copy(dtype=self.dtype), 
                result=result
            )
        return result

    def to(self, dtype):
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
    
    def __array__(self, dtype=None):
        # Numpy array interface, to support `numpy.asarray(tensor) -> ndarray`
        if dtype is None:
            return self.numpy()
        else:
            return self.numpy().astype(dtype, copy=False)

    def detach(self):
        return _Tensor(self._array, requires_grad=False)

    def backward(self, vector = None, retain_graph = False):
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
            self.grad = vector
        else:
            self.grad = None
        dag.backward(self.node_id, retain_graph=retain_graph)

    def plot_dag(self, full_graph=False):
        dag.plot(self.node_id, full_graph)


    def __str__(self):
        if not self.requires_grad:
            return f"eazygrad.tensor({self._array.tolist()}, dtype={self.dtype})"
        else:
            return f"eazygrad.tensor({self._array.tolist()}, dtype={self.dtype}, requires_grad={self.requires_grad})"
    
    def __repr__(self):
        # For printing tensor nested
        if not self.requires_grad:
            return f"eazygrad.tensor({self._array.tolist()}, dtype={self.dtype})"
        else:
            return f"eazygrad.tensor({self._array.tolist()}, dtype={self.dtype}, requires_grad={self.requires_grad})"
