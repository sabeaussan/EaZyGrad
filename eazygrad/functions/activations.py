from __future__ import annotations

import numpy as np
from ..utils import check
from .._tensor import _Tensor
from ..grad import operations, dag


def relu(input: _Tensor) -> _Tensor:
	"""
	Apply the rectified linear unit activation elementwise.

	Parameters
	----------
	input : _Tensor
		Input tensor.

	Returns
	-------
	_Tensor
		Tensor with ``max(input, 0)`` applied elementwise.
	"""
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")
	
	requires_grad = input.requires_grad
	result = _Tensor(np.maximum(0,input._array), requires_grad = requires_grad)
	if requires_grad : 
		result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.ReLU(arr=input._array), result = result)
	return result


def _stable_sigmoid_pos(x: np.ndarray) -> np.ndarray:
	return 1/(1+np.exp(-x))

def _stable_sigmoid_neg(x: np.ndarray) -> np.ndarray:
	return np.exp(x)/(1+np.exp(x))

def sigmoid(input: _Tensor) -> _Tensor:
	"""
	Apply the sigmoid activation elementwise.

	Parameters
	----------
	input : _Tensor
		Input tensor.

	Returns
	-------
	_Tensor
		Tensor containing values in the interval ``[0, 1]``.

	Notes
	-----
	The implementation uses a numerically stable formulation based on the
	sign of the input.
	"""
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")

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
		result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Sigmoid(sigmoid=f64_array, dtype=dtype), result = result)
	return result


def tanh(input: _Tensor) -> _Tensor:
	"""
	Apply the hyperbolic tangent activation elementwise.

	Parameters
	----------
	input : _Tensor
		Input tensor.

	Returns
	-------
	_Tensor
		Tensor containing ``tanh(input)``.
	"""
	if not isinstance(input, _Tensor):
		raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")

	requires_grad = input.requires_grad
	tanh_array = np.tanh(input._array)
	result = _Tensor(tanh_array, requires_grad=requires_grad)
	if requires_grad:
		result.node_id = dag.create_node(
			parents_id=[input.node_id],
			operation=operations.Tanh(tanh=tanh_array),
			result=result,
		)
	return result
