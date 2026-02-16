import numpy as np
from ..utils import check
# Return always (grad,) because otherwise iterate over multiple of the grad if it is a list

# Primitive operations operating on Tensor. Define a single node in the computation graph

# TODO : check return dtypes, there is force cast to float32 no matter the original dtype

__all__ = [
    "Operation",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "RDiv",
    "Pow",
    "MatMul",
    "Sum",
    "Mean",
    "Exp",
    "Log",
    "ReLU",
    "Cos",
    "Sin",
    "Slice",
    "Reshape",
    "ExpandDims",
    "Squeeze",
    "SwapDims",
]


class Operation:

	# stateful 

	def __init__(self, *context): # context -> args
		self.context = []
		for ctx in context:
			if isinstance(ctx, np.ndarray):
				# protection against in-place ops
				# prevent modification of saved context
				ctx.flags.writeable=False
			self.context.append(ctx)

	def backward(self, grad_output):
		# return the result of the grad_output-jacobian product
		raise NotImplementedError
	
	def __repr__(self):
		return self.__class__.__name__

# Bizarre Add et Sub en terme de args
# Pas sur de la fiabilité
class Add(Operation):

	def backward(self, grad_output):
		# One of the operands might be a scalar which does not need grad
		# so we need to check wether we return one or two grads
		grad_inputs = []
		if grad_output is None :
			grad_inputs.append(np.ones(self.context[0], dtype=self.context[0].dtype))
			if len(self.context) > 1:
				grad_inputs.append(np.ones(self.context[1], dtype=self.context[0].dtype))
			return tuple(grad_inputs)
		grad_inputs.append(grad_output)
		if len(self.context) > 1:
			grad_inputs.append(grad_output)
		return tuple(grad_inputs)

class Sub(Operation):

	def backward(self, grad_output):
		# One of the operands might be a scalar which does not need grad
		# so we need to check wether we return one or two grads
		grad_inputs = []
		if grad_output is None :
			grad_inputs.append(np.ones(self.context[0], dtype=self.context[0].dtype))
			if len(self.context) > 1:
				grad_inputs.append(-np.ones(self.context[1], dtype=self.context[0].dtype))
			return tuple(grad_inputs)
		grad_inputs.append(grad_output)
		if len(self.context) > 1:
			grad_inputs.append(-grad_output)
		return tuple(grad_inputs)

class Mul(Operation):

	def backward(self, grad_output):
		# One of the operands might be a scalar which does not need grad
		# so we need to check wether we return one or two grads
		grad_inputs = [self.context[1]]
		if len(self.context) > 1:
			grad_inputs.append(self.context[0])	
		if grad_output is None :
			grad_inputs = map(lambda x : x*np.ones_like(x, dtype=x.dtype), grad_inputs)
			return tuple(grad_inputs)
		grad_inputs = map(lambda x : x*grad_output, grad_inputs)
		return tuple(grad_inputs)

class Div(Operation):

	def backward(self, grad_output):
		# Type promotion for division
		dtype = self.context[0].dtype
		x_64 = self.context[0].astype(np.float64, copy=False)
		y_64 = self.context[1]

		if not check.is_scalar(self.context[1]):
			y_64 = y_64.astype(np.float64, copy=False)
			# Possibly recast gradients to original dtype
			# if dtype is float64, no op
			grad_x = (1 / y_64).astype(dtype, copy=False)
		else:
			grad_x = (1 / y_64)

		grad_inputs = [grad_x]
		# One of the operands might be a scalar which does not need grad
		# so we need to check wether we return one or two grads
		if len(self.context) > 1:
			grad_y = (- x_64 / (y_64 ** 2)).astype(dtype, copy=False)
			grad_inputs.append(grad_y)

		if grad_output is None :
			return tuple(grad_inputs)
		
		grad_inputs = map(lambda x : x*grad_output, grad_inputs)
		return tuple(grad_inputs)

class RDiv(Operation):

	def backward(self, grad_output):
		# Type promotion for division
		dtype = self.context[0].dtype
		x_64 = self.context[0].astype(np.float64, copy=False)
		
		# Possibly recast gradients to original dtype
		# if dtype is float64, no op
		grad_x = (- self.context[1] / (x_64 ** 2)).astype(dtype, copy=False)

		if grad_output is None :
			return (grad_x, None)
		return (grad_x * grad_output, None)

class Pow(Operation):

	def backward(self, grad_output):
		if self.context[1] == 0:
			grad_x = np.zeros_like(self.context[0])
		else:
			grad_x = self.context[1] * (self.context[0]**(self.context[1]-1))
		if grad_output is None:
			return (grad_x,)	
		return (grad_x * grad_output,)
	
class InnerProduct(Operation):

	def backward(self, grad_output):
		grad_x = self.context[1]
		grad_y = self.context[0]
		if grad_output is None :
			return (grad_x, grad_y)
		else :
			return (grad_output * grad_x, grad_output * grad_y)


class MatMul(Operation):

	def backward(self, grad_output):
		if self.context[0].ndim > 1 :
			grad_x = np.swapaxes(self.context[1],-1,-2)
		# else:
		# 	grad_x = np.expand_dims(self.context[1], axis=0)

		if self.context[1].ndim > 1 :
			grad_y = np.swapaxes(self.context[0],-1,-2)
		# else:
		# 	grad_y = np.expand_dims(self.context[0], axis=1)

		if grad_output.ndim==0:
			grad_output = np.expand_dims(grad_output, axis=0)

		if grad_output is None :
			return (grad_x, grad_y)
		else :
			# print(grad_output.shape, grad_x.shape, grad_y.shape)
			return (grad_output@grad_x, grad_y@grad_output)


class Sum(Operation):
	def backward(self, grad_output):
		dim, keepdims = self.context[1]
		if grad_output is None :
			return (np.ones(self.context[0], dtype=np.float32),)
		else :
			# Check keepdims
			if not keepdims:
				grad_output = np.expand_dims(grad_output, axis=dim)
			return (grad_output*np.ones(self.context[0], dtype=grad_output.dtype),)

class Mean(Operation):
	def backward(self, grad_output):
		dim, keepdims = self.context[2]
		if grad_output is None :
			return (np.ones(self.context[0], dtype=np.float32)*self.context[1],)
		else :
			# Check keepdims
			if not keepdims:
				grad_output = np.expand_dims(grad_output, axis=dim)
			return (grad_output*np.ones(self.context[0], dtype=grad_output.dtype)*self.context[1],)


class Exp(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (np.exp(self.context[0]),)
		return (grad_output * np.exp(self.context[0]),)
	
class Sigmoid(Operation):

	def backward(self, grad_output):
		sigmoid = self.context[0]
		dtype = self.context[1]
		dx = (sigmoid*(1-sigmoid)).astype(dtype, copy=False)
		if grad_output is None :
			return (dx,)
		return (grad_output * dx,)

class Log(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (1/self.context[0],) 
		return (grad_output * 1/self.context[0],)

class ReLU(Operation):

	def backward(self, grad_output):
		r = self.context[0] > 0
		if grad_output is None :
			return (r,)
		return (r * grad_output,)

class Cos(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (-np.sin(self.context[0]),)
		else :
			return (-np.sin(self.context[0]) * grad_output,)

class Sin(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (np.cos(self.context[0]),)
		else :
			return (np.cos(self.context[0]) * grad_output,)

class Slice(Operation):

	def backward(self, grad_output):
		# TODO: pas hyper sur de ça
		# Si les grad[key] on été utilisé différemment est ce que c'est toujours juste ?
		key = self.context[1]
		grad = np.zeros(self.context[0], dtype=grad_output.dtype)
		if grad_output is None :
			grad[key] = 1
		else :
			grad[key] = grad_output
		return (grad,)

class Reshape(Operation):

	def backward(self, grad_output):
		grad = grad_output.reshape(self.context[0])
		return (grad,)

class ExpandDims(Operation):
	# Je crois que c'est juste un squeeze le reverse, pas un sum
	def backward(self, grad_output):
		grad = np.squeeze(grad_output, axis = self.context[0])
		return (grad,)

class Squeeze(Operation):

	def backward(self, grad_output):
		grad = np.expand_dims(grad_output, axis = self.context[0])
		return (grad,)

class SwapDims(Operation):

	# Je crois que c'est juste un squeeze le reverse, pas un sum
	def backward(self, grad_output):
		grad = np.swapaxes(grad_output, axis1 = self.context[1], axis2 = self.context[0])
		return (grad,)
	
class LogSumExp(Operation):

	def _softmax_from_logsumexp(input_array, logsumexp, dim, dtype):
		# Ensure logsumexp has same ndim as input_array for broadcasting
		if logsumexp.ndim < input_array.ndim:
			logsumexp = np.expand_dims(logsumexp, axis=dim)
		shifted_input = input_array - logsumexp
		return np.exp(shifted_input).astype(dtype, copy=False)

	def backward(self, grad_output):
		# Type promotion for exp and sum
		input_array = self.context[0] # float64
		logsumexp = self.context[1]   # float64
		dim = self.context[2]

		# Choose a dtype safely (grad_output may be None)
		dtype = grad_output.dtype if grad_output is not None else input_array.dtype

		softmax = LogSumExp._softmax_from_logsumexp(input_array, logsumexp, dim, dtype)

		if grad_output is None:
			return (softmax,)
		else:
			# Ensure grad_output has the correct shape for broadcasting
			if grad_output.ndim < input_array.ndim:
				grad_output = np.expand_dims(grad_output, axis=dim)
			return (softmax * grad_output,)
		
class BinaryCrossEntropy(Operation):

	def backward(self, grad_output):
		logits = self.context[0]
		target = self.context[1]
		dtype = grad_output.dtype if grad_output is not None else logits.dtype
		target = target.astype(dtype, copy=False)
		# Stable sigmoid to avoid overflow for large |logits|
		sigmoid = np.where(
			logits >= 0,
			1 / (1 + np.exp(-logits)),
			np.exp(logits) / (1 + np.exp(logits))
		).astype(dtype, copy=False)
		grad = sigmoid - target
		if grad_output is None:
			return (grad,)
		return (grad * grad_output,)
		
class Copy(Operation):

	def backward(self, grad_output):
		dtype = self.context[0]
		if dtype != grad_output.dtype:
			return (grad_output.astype(dtype, copy=False),)
		else:
			return (grad_output,)