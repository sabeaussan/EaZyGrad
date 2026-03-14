from __future__ import annotations

import numpy as np
from ..utils import check
from .._tensor import _Tensor
from ..grad import operations, dag


def min(input: _Tensor, other: _Tensor) -> _Tensor:
	"""
	Return the elementwise minimum of two tensors.

	Parameters
	----------
	input : _Tensor
		First input tensor.
	other : _Tensor
		Second input tensor.

	Returns
	-------
	_Tensor
		Elementwise minimum of ``input`` and ``other``.
	"""
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
	

def exp(input: _Tensor) -> _Tensor:
	"""
	Compute the elementwise exponential.

	Parameters
	----------
	input : _Tensor
		Input tensor.

	Returns
	-------
	_Tensor
		Tensor containing ``exp(input)``.
	"""
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")

	requires_grad = input.requires_grad
	array = input._array
	result = _Tensor(np.exp(array), requires_grad = requires_grad)
	if requires_grad : 
		result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Exp(arr=input._array), result = result)

	return result

def log(input: _Tensor) -> _Tensor:
	"""
	Compute the elementwise natural logarithm.

	Parameters
	----------
	input : _Tensor
		Input tensor. Values should be strictly positive.

	Returns
	-------
	_Tensor
		Tensor containing ``log(input)``.
	"""
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")
	requires_grad = input.requires_grad
	array = input._array
	result = _Tensor(np.log(array), requires_grad = requires_grad)
	if requires_grad : 
		result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Log(arr=input._array), result = result)

	return result

def cos(input: _Tensor) -> _Tensor:
	"""
	Compute the elementwise cosine.

	Parameters
	----------
	input : _Tensor
		Input tensor.

	Returns
	-------
	_Tensor
		Tensor containing ``cos(input)``.
	"""
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")
	
	requires_grad = input.requires_grad
	result = _Tensor(np.cos(input._array), requires_grad = requires_grad)
	if requires_grad : 
		result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Cos(arr=input._array), result = result)
	return result

def sin(input: _Tensor) -> _Tensor:
	"""
	Compute the elementwise sine.

	Parameters
	----------
	input : _Tensor
		Input tensor.

	Returns
	-------
	_Tensor
		Tensor containing ``sin(input)``.
	"""
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")
	requires_grad = input.requires_grad
	result = _Tensor(np.sin(input._array), requires_grad = requires_grad)
	if requires_grad : 
		result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Sin(arr=input._array), result = result)
	return result


def clip(input: _Tensor, low: float, high: float) -> _Tensor:
	"""
	Clip tensor values to a closed interval.

	Parameters
	----------
	input : _Tensor
		Input tensor.
	low : scalar
		Lower clipping bound.
	high : scalar
		Upper clipping bound.

	Returns
	-------
	_Tensor
		Clipped tensor.
	"""
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
