import numpy as np
from ..utils import check
from .._tensor import _Tensor
from ..grad import operations, dag



def exp(input):
	if check.is_scalar(input):
		raise TypeError("Expected a tensor")
	elif isinstance(input, _Tensor):
		requires_grad = input.requires_grad
		# Type promotion for exp
		array = input._array
		result = _Tensor(np.exp(array), requires_grad = requires_grad)
		if requires_grad : 
			result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Exp(arr=input._array), result = result)
	else :
		raise NotImplementedError
	return result

def log(input):
	if check.is_scalar(input):
		raise TypeError("Expected a tensor")
	elif isinstance(input, _Tensor):
		requires_grad = input.requires_grad
		# Type promotion for log
		array = input._array
		result = _Tensor(np.log(array), requires_grad = requires_grad)
		if requires_grad : 
			result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Log(arr=input._array), result = result)
	else :
		raise NotImplementedError
	return result

def cos(input):
	if check.is_scalar(input):
		raise TypeError("Expected a tensor")
	elif isinstance(input, _Tensor):
		requires_grad = input.requires_grad
		result = _Tensor(np.cos(input._array), requires_grad = requires_grad)
		if requires_grad : 
			result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Cos(arr=input._array), result = result)
	else :
		raise NotImplementedError
	return result

def sin(input):
	if check.is_scalar(input):
		raise TypeError("Expected a tensor")
	elif isinstance(input, _Tensor):
		requires_grad = input.requires_grad
		result = _Tensor(np.sin(input._array), requires_grad = requires_grad)
		if requires_grad : 
			result.node_id = dag.create_node(parents_id = [input.node_id], operation = operations.Sin(arr=input._array), result = result)
	else :
		raise NotImplementedError
	return result
