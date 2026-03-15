from __future__ import annotations

import numpy as np
from .math import log
from ..utils import check
from .._tensor import _Tensor
from ..grad import operations, dag
from .specials import logsumexp

def mse_loss(predicted: _Tensor, target: _Tensor) -> _Tensor:
	"""
	Compute the mean squared error loss.

	Parameters
	----------
	predicted : _Tensor
		Predicted values.
	target : _Tensor
		Target values with the same shape as ``predicted``.

	Returns
	-------
	_Tensor
		Scalar loss tensor.
	"""
	return ((predicted - target) ** 2).mean()

def nll_loss(predicted: _Tensor, target: _Tensor) -> _Tensor:
	"""
	Compute the negative log-likelihood loss from log-probabilities.

	Parameters
	----------
	predicted : _Tensor
		Log-probabilities of shape ``(N, C)``.
	target : _Tensor
		Integer class indices of shape ``(N,)``.

	Returns
	-------
	_Tensor
		Scalar loss tensor.
	"""
	correct_probs = predicted[np.arange(predicted.shape[0]), target._array]
	return -correct_probs.mean()


def bce_with_logits_loss(logits: _Tensor, target: _Tensor) -> _Tensor:
	"""
	Compute binary cross-entropy loss from unnormalized logits.

	Parameters
	----------
	logits : _Tensor
		Input logits.
	target : _Tensor
		Target tensor with values typically in ``[0, 1]`` and the same shape as
		``logits``.

	Returns
	-------
	_Tensor
		Scalar loss tensor.

	Notes
	-----
	This implementation uses a numerically stable formulation and internally
	averages the per-element loss.
	"""

	if not isinstance(logits, _Tensor):
		raise TypeError(f"Expected predicted to be an eazygrad tensor, got {type(logits)}")
	if not isinstance(target, _Tensor):
		raise TypeError(f"Expected predicted to be an eazygrad tensor, got {type(target)}")
	
	# Maybe type promotion for exp and sum
	dtype = logits.dtype
	f64_array = logits._array.astype(np.float64, copy=False)
	# Numerically stable computation of BCE with logits
	bce = np.clip(f64_array, a_min=0, a_max=None) - f64_array*target._array + np.log1p(np.exp(-np.abs(f64_array)))
	requires_grad = logits.requires_grad
	# Recast to input dtype
	result = _Tensor(bce, requires_grad=requires_grad, dtype=dtype)
	if requires_grad : 
		result.node_id = dag.create_node(
			parents_id = [logits.node_id], 
			operation = operations.BinaryCrossEntropy(logits=f64_array, target=target._array), 
			result = result
		)
	return result.mean()

def bce_loss(predicted: _Tensor, target: _Tensor) -> _Tensor:
	"""
	Compute binary cross-entropy loss from probabilities.

	Parameters
	----------
	predicted : _Tensor
		Predicted probabilities.
	target : _Tensor
		Target tensor with the same shape as ``predicted``.

	Returns
	-------
	_Tensor
		Scalar loss tensor.
	"""
	return -(target * log(predicted) + (1 - target) * log(1 - predicted)).mean()

def cross_entropy_loss(predicted: _Tensor, target: _Tensor) -> _Tensor:
	"""
	Compute cross-entropy loss from class logits.

	Parameters
	----------
	predicted : _Tensor
		Logits of shape ``(N, C)``.
	target : _Tensor
		Targets of shape ``(N,)`` for integer class indices or shape ``(N, C)``
		for soft targets.

	Returns
	-------
	_Tensor
		Scalar loss tensor.

	Notes
	-----
	The loss is computed with a numerically stable log-sum-exp formulation.
	"""
	# Expect predicted to be Tensor with logits shape (N, C)
	# target should be Tensor with class indices or target distribution
	# -* if target is class indices shape is (N,)
	# -* if target is distribution shape is (N, C)
	if not isinstance(predicted, _Tensor):
		raise TypeError(f"Expected predicted to be an eazygrad tensor, got {type(predicted)}")
	if not isinstance(target, _Tensor):
		raise TypeError(f"Expected predicted to be an eazygrad tensor, got {type(target)}")
	if predicted.ndim != 2:
		raise ValueError("Expected predicted to have shape (N, C)")
	# Numerically stable computation of cross entropy loss
	if target.shape == (predicted.shape[0],):
		# Target is class indices
		if not np.issubdtype(target._array.dtype, np.integer):
			raise TypeError("Target class indices must be an integer tensor")
		target_idx = target._array.astype(np.int64, copy=False)
		if np.any(target_idx < 0) or np.any(target_idx >= predicted.shape[1]):
			raise ValueError("Target class indices are out of range")
		target_logits = predicted[np.arange(predicted.shape[0]), target_idx]
		cross_entropy = logsumexp(predicted, dim=-1, keepdims=False) - target_logits 
	elif target.shape == predicted.shape:
		# distribution / soft targets
		lse = logsumexp(predicted, dim=-1, keepdims=False)         # (N,)
		expected_logit = (target * predicted).sum(dim=-1)          # (N,)
		cross_entropy = lse - expected_logit
	else:
		raise ValueError("Target shape is not compatible with predicted shape")
	return cross_entropy.mean()
