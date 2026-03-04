import numpy as np
from ..utils import check
from .._tensor import _Tensor
from ..grad import operations, dag


def min(input, other):
	# Only for scalar Tensors
	if not isinstance(input, _Tensor) and not isinstance(other, _Tensor):
		raise TypeError(f"Expected inputs to be eazygrad tensors, but got {type(input)} and {type(other)}.")

	requires_grad = input.requires_grad
	if input._array <= other._array:
		min_val = input._array
		idx = 0
	else:
		min_val = other._array
		idx = 1
	result = _Tensor(min_val, requires_grad = requires_grad)
	if requires_grad:
		result.node_id = dag.create_node(
			parents_id = [input.node_id, other.node_id], 
			operation = operations.Min(idx=idx), 
			result = result
		)
	return result
	

def exp(input):
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")

	requires_grad = input.requires_grad
	# Type promotion for exp
	array = input._array
	result = _Tensor(np.exp(array), requires_grad = requires_grad)
	if requires_grad : 
		result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Exp(arr=input._array), result = result)

	return result

def log(input):
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")
	requires_grad = input.requires_grad
	# Type promotion for log
	array = input._array
	result = _Tensor(np.log(array), requires_grad = requires_grad)
	if requires_grad : 
		result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Log(arr=input._array), result = result)

	return result

def cos(input):
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")
	
	requires_grad = input.requires_grad
	result = _Tensor(np.cos(input._array), requires_grad = requires_grad)
	if requires_grad : 
		result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Cos(arr=input._array), result = result)
	return result

def sin(input):
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")
	requires_grad = input.requires_grad
	result = _Tensor(np.sin(input._array), requires_grad = requires_grad)
	if requires_grad : 
		result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Sin(arr=input._array), result = result)
	return result
