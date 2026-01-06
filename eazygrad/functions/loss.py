import numpy as np
from .math import log
from ..utils import check
from .._tensor import _Tensor
from ..grad import operations, dag

def mse_loss(predicted, target):
	return ((predicted - target) ** 2).mean()

def nll_loss(predicted, target):
	correct_probs = predicted[np.arange(predicted.shape[0]), target._array]
	return -correct_probs.mean()


def bce_with_logits_loss(logits, target):
	if check.is_scalar(logits):
		raise TypeError("Expected a tensor")
	elif isinstance(logits, _Tensor):
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
				operation = operations.BinaryCrossEntropy(f64_array, target._array), 
				result = result
			)
	return result.mean()

def bce_loss(predicted, target):
	return -(target * log(predicted) + (1 - target) * log(1 - predicted)).mean()