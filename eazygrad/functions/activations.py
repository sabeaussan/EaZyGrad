import numpy as np
from ..utils import check
from .._tensor import _Tensor
from ..grad import operations, dag


def relu(input):
	if check.is_scalar(input):
		raise TypeError("Expected a tensor")
	elif isinstance(input, _Tensor):
		requires_grad = input.requires_grad
		result = _Tensor(np.maximum(0,input._array), requires_grad = requires_grad)
		if requires_grad : 
			result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.ReLU(input._array), result = result)
	else :
		raise NotImplementedError
	return result

def _stable_sigmoid_pos(x):
	return 1/(1+np.exp(-x))

def _stable_sigmoid_neg(x):
	return np.exp(x)/(1+np.exp(x))

def sigmoid(input):
	if check.is_scalar(input):
		raise TypeError("Expected a tensor")
	elif isinstance(input, _Tensor):
		# Sigmoid is numerically unstable
		# Compute sigmoid based on sign
		# Forces type promotion and copy before in-place op
		dtype = input.dtype
		f64_array = input._array.astype(np.float64)
		# Compute sigmoid inplace
		mask = f64_array>0
		f64_array[mask] = _stable_sigmoid_pos(f64_array[mask])
		f64_array[~mask] = _stable_sigmoid_neg(f64_array[~mask])
		result = _Tensor(f64_array, requires_grad = input.requires_grad, dtype=dtype)
		if input.requires_grad : 
			# /!\ array is now the sigmoid 
			result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Sigmoid(f64_array, dtype), result = result)
	else :
		raise NotImplementedError
	return result