import numpy as np
from ..utils import check
from .._tensor import _Tensor
from ..grad import operations, dag


def min(input, other):
	if not isinstance(input, _Tensor) or not isinstance(other, _Tensor):
		raise TypeError(f"Expected inputs to be eazygrad tensors, but got {type(input)} and {type(other)}.")

	requires_grad = input.requires_grad or other.requires_grad
	min_val = np.minimum(input._array, other._array)
	result = _Tensor(min_val, requires_grad = requires_grad)
	if requires_grad:
		result.node_id = dag.create_node(
			parents_id = [input.node_id, other.node_id], 
			operation = operations.Min(arr1=input._array, arr2=other._array), 
			result = result
		)
	return result
	

def exp(input):
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")

	requires_grad = input.requires_grad
	array = input._array
	result = _Tensor(np.exp(array), requires_grad = requires_grad)
	if requires_grad : 
		result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Exp(arr=input._array), result = result)

	return result

def log(input):
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")
	requires_grad = input.requires_grad
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


def clip(input, low, high):
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")
	if not check.is_scalar(low) or not check.is_scalar(high):
		raise TypeError(f"Expected scalar bounds, got {type(low)} and {type(high)}")
	if low > high:
		raise ValueError(f"Expected low <= high, got low={low}, high={high}")

	requires_grad = input.requires_grad
	result = _Tensor(np.clip(input._array, low, high), requires_grad=requires_grad)
	if requires_grad:
		result.node_id = dag.create_node(
			parents_id=[input.node_id],
			operation=operations.Clip(arr=input._array, low=low, high=high),
			result=result,
		)
	return result
