import numpy as np
from ..utils import check
from .._tensor import _Tensor
from ..grad import operations, dag
from .math import exp
import numba as nb
# import line_profiler
#
# TODO : revoir les messages d'erreurs pour mieux indiquer ce qui devrai etre un tensor
# TODO : maybe define a common wrapper or higher order function to take care of scalar check and co

def _validate_dim_arg(dim):
	if not isinstance(dim, int):
		raise ValueError("Dim argument should be an integer, got {}".format(type(dim)))

# TODO : generic but temporary implementation of logsumexp
# Slow, will be replaced with a numba friendly version
def _logsumexp_generic(f64_array, dim):
	M = f64_array.max(axis=dim, keepdims=True)
	logsumexp = np.exp(f64_array-M).sum(axis=dim, keepdims=True)
	logsumexp = np.log(logsumexp)
	logsumexp += M
	return logsumexp

@nb.njit(cache=True, fastmath=True, parallel=True)
def _fast_logsumexp(x2d):
    N, K = x2d.shape
    out = np.empty(N, dtype=np.float64)

    for i in nb.prange(N):
        # 1) max
        m = -np.inf
        for j in range(K):
            v = x2d[i, j]
            if v > m:
                m = v

        # Handle all -inf row
        if m == -np.inf:
            out[i] = -np.inf
            continue

        # 2) sum exp(x - m)
        s = 0.0
        for j in range(K):
            s += np.exp(x2d[i, j] - m)

        out[i] = np.log(s) + m

    return out

# @line_profiler.profile
def logsumexp(input, dim, keepdims=False):
	if check.is_scalar(input):
		raise TypeError("Expected a tensor")
	elif isinstance(input, _Tensor):
		ndim = input.ndim
		requires_grad = input.requires_grad
		if ndim == 0:
			result = _Tensor(input.numpy(), requires_grad=requires_grad)
			if requires_grad : 
				result.node_id = dag.create_node(
					parents_id = [input.node_id], 
					operation = operations.Copy(input.dtype), 
					result = result
				)
			return result
			
		# if ndim > 3:
		# 	raise NotImplementedError("Logsumexp for tensor with more than 3 dimensions is not supported.")

		_validate_dim_arg(dim)
		dtype = input.dtype
		reshaped = False
		# Maybe type promotion for exp and sum
		f64_array = input._array.astype(np.float64, copy=False)
		

		if dim != -1 or dim != ndim-1:
			f64_array = np.moveaxis(f64_array, dim, -1)
			new_shape = f64_array.shape
			axis_moved = True

		if ndim > 2:
			f64_array = f64_array.reshape(-1, f64_array.shape[-1])
			reshaped = True


		if not f64_array.flags.c_contiguous:
			f64_array = np.ascontiguousarray(f64_array)

		if ndim==1:
			logsumexp = _fast_logsumexp(np.expand_dims(f64_array, axis=0))
		else:
			logsumexp = _fast_logsumexp(f64_array)

		if reshaped:
			logsumexp = logsumexp.reshape(new_shape[:-1])

		if keepdims and logsumexp.ndim < ndim:
			logsumexp = np.expand_dims(logsumexp, axis=dim)

		
		# Recast to input dtype
		result = _Tensor(logsumexp, requires_grad=requires_grad, dtype=dtype)
		if requires_grad : 
			result.node_id = dag.create_node(
				parents_id = [input.node_id], 
				operation = operations.LogSumExp(f64_array, logsumexp, dim), 
				result = result
			)
	else :
		raise NotImplementedError
	return result 


def softmax(input, dim):
	if check.is_scalar(input):
		raise TypeError("Expected a tensor")
	elif isinstance(input, _Tensor):
		_validate_dim_arg(dim)
		# Type promotion for numerical stability
		input = input.double()
		shifted_input = input - logsumexp(input, dim, keepdims=True)
		result = exp(shifted_input).float()
	else :
		raise NotImplementedError
	return result

def log_softmax(input, dim):
	if check.is_scalar(input):
		raise TypeError("Expected a tensor")
	elif isinstance(input, _Tensor):
		_validate_dim_arg(dim)
		# Type promotion for numerical stability
		input = input.double()
		result = input - logsumexp(input, dim, keepdims=True).float()
	else :
		raise NotImplementedError
	return result