import numpy as np
from ..utils import check
from .._tensor import _Tensor
from ..grad import operations, dag
from .math import exp
import numba as nb
import line_profiler

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
def _logsumexp_lastaxis(x2d):
    """
    x2d: shape (N, K), contiguous C-order recommended for speed.
    returns: shape (N,), logsumexp over axis=1
    """
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
		_validate_dim_arg(dim)
		dtype = input.dtype
		# Maybe type promotion for exp and sum
		f64_array = input._array.astype(np.float64, copy=False)
		fast_path = input.ndim == 2 and (dim == 1 or dim == -1) and f64_array.flags.c_contiguous
		if fast_path:
			# Fast path for logsumexp over last axis of 2d array, which is common in softmax and cross entropy loss
			# Not generic, need some work
			logsumexp = _logsumexp_lastaxis(f64_array)
			if keepdims and logsumexp.ndim < input.ndim:
				logsumexp = np.expand_dims(logsumexp, axis=dim)
		else:
			# TODO : need to move dim axis to last and reshape to 2d
			# to replace this slow implementation
			logsumexp = _logsumexp_generic(f64_array, dim)
			if not keepdims:
				logsumexp = logsumexp.squeeze(dim)
		requires_grad = input.requires_grad
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