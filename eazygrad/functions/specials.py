import numpy as np
from ..utils import check
from .._tensor import _Tensor
from ..grad import operations, dag
from .math import exp

# TODO : revoir les messages d'erreurs pour mieux indiquer ce qui devrai etre un tensor
# TODO : define a common wrapper or higher order function to take care of scalar check and co
# TODO : rajouter max et min

def _validate_dim_arg(dim):
	if not isinstance(dim, int):
		raise ValueError("Dim argument should be an integer, got {}".format(type(dim)))

def logsumexp(input, dim, keepdims=False):
	if check.is_scalar(input):
		raise TypeError("Expected a tensor")
	elif isinstance(input, _Tensor):
		_validate_dim_arg(dim)
		dtype = input.dtype
		# Maybe type promotion for exp and sum
		f64_array = input._array.astype(np.float64, copy=False)
		M = np.max(f64_array, axis=dim, keepdims=True)
		sumexp = np.exp(f64_array-M).sum(axis=dim, keepdims=True) # TODO: check broadcastability
		logsumexp = (np.log(sumexp)+M)
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