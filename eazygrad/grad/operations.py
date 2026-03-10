import numpy as np
from ..utils import check
# Return always (grad,) because otherwise iterate over multiple of the grad if it is a list

# Primitive operations operating on Tensor. Define a single node in the computation graph

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
	"Tanh",
	"Cos",
	"Sin",
	"Clip",
	"Min",
	"Slice",
	"Reshape",
	"ExpandDims",
	"Squeeze",
	"SwapDims",
]


class Operation:

	# stateful 

	def __init__(self, **kwargs): # context -> args
		self.context = kwargs
		for ctx in self.context.values():
			if isinstance(ctx, np.ndarray):
				# protection against in-place ops
				# prevent modification of saved context
				ctx.flags.writeable=False

	def backward(self, grad_output):
		# return the result of the grad_output-jacobian product
		raise NotImplementedError
	
	def __repr__(self):
		return self.__class__.__name__


class Add(Operation):

	def backward(self, grad_output):
		if len(self.context.keys())==1:
			return (grad_output,)
		return (grad_output, grad_output)

class Sub(Operation):

	def backward(self, grad_output):
		if len(self.context.keys())==1:
			return (grad_output,)
		# tensor - tensor
		return (grad_output, -grad_output)

class Mul(Operation):

	def backward(self, grad_output):
		if "scalar" in self.context:
			scalar = self.context["scalar"]
			return (scalar * grad_output,)
		# tensor * tensor
		arr1 = self.context["arr1"]
		arr2 = self.context["arr2"]
		return (arr2 * grad_output, arr1 * grad_output)

class Div(Operation):

	def backward(self, grad_output):
		if "scalar" in self.context:
			arr = self.context["arr"]
			scalar = self.context["scalar"]
			dtype = arr.dtype
			grad_x = np.array(1 / scalar, dtype=dtype)
			return (grad_x * grad_output,)
		# tensor / tensor
		arr1 = self.context["arr1"]
		arr2 = self.context["arr2"]
		dtype = arr1.dtype
		x_64 = arr1.astype(np.float64, copy=False)
		y_64 = arr2.astype(np.float64, copy=False)
		grad_x = (1 / y_64).astype(dtype, copy=False)
		grad_y = (-x_64 / (y_64 ** 2)).astype(dtype, copy=False)
		return (grad_x * grad_output, grad_y * grad_output)

class RDiv(Operation):

	def backward(self, grad_output):
		# Type promotion for division
		arr = self.context["arr"]
		scalar = self.context["scalar"]
		dtype = arr.dtype
		x_64 = arr.astype(np.float64, copy=False)
		
		# Possibly recast gradients to original dtype
		# if dtype is float64, no op
		grad_x = (- scalar / (x_64 ** 2)).astype(dtype, copy=False)
		return (grad_x * grad_output, None)

class Pow(Operation):

	def backward(self, grad_output):
		arr = self.context["arr"]
		exponent = self.context["exponent"]
		if exponent == 0:
			grad_x = np.zeros_like(arr)
		else:
			grad_x = exponent * (arr ** (exponent - 1))
		return (grad_x * grad_output,)
	
class InnerProduct(Operation):

	def backward(self, grad_output):
		grad_x = self.context["arr2"]
		grad_y = self.context["arr1"]
		if grad_output is None :
			return (grad_x, grad_y)
		else :
			return (grad_output * grad_x, grad_output * grad_y)


class MatMul(Operation):

	def backward(self, grad_output):
		arr1 = self.context["arr1"]
		arr2 = self.context["arr2"]
		if arr1.ndim > 1:
			grad_x = np.swapaxes(arr2, -1, -2)
		else:
			grad_x = np.expand_dims(arr2, axis=0)

		if arr2.ndim > 1:
			grad_y = np.swapaxes(arr1, -1, -2)
		else:
			grad_y = np.expand_dims(arr1, axis=1)

		if grad_output.ndim == 0:
			grad_output = np.expand_dims(grad_output, axis=0)
		return (grad_output @ grad_x, grad_y @ grad_output)


class Sum(Operation):
	def backward(self, grad_output):
		dim = self.context["dim"]
		keepdims = self.context["keepdims"]
		shape = self.context["shape"]
		dtype = self.context["dtype"]
		# Check keepdims
		if not keepdims:
			grad_output = np.expand_dims(grad_output, axis=dim)
		return (grad_output * np.ones(shape, dtype=dtype),)

class Mean(Operation):
	def backward(self, grad_output):
		dim = self.context["dim"]
		keepdims = self.context["keepdims"]
		shape = self.context["shape"]
		div_factor = self.context["div_factor"]
		dtype = self.context["dtype"]
		# Check keepdims
		if not keepdims:
			grad_output = np.expand_dims(grad_output, axis=dim)
		return (grad_output * np.ones(shape, dtype=dtype) * div_factor,)


class Exp(Operation):

	def backward(self, grad_output):
		arr = self.context["arr"]
		if grad_output is None :
			return (np.exp(arr),)
		return (grad_output * np.exp(arr),)
	
class Sigmoid(Operation):

	def backward(self, grad_output):
		sigmoid = self.context["sigmoid"]
		dtype = self.context["dtype"]
		dx = (sigmoid*(1-sigmoid)).astype(dtype, copy=False)
		return (grad_output * dx,)

class Log(Operation):

	def backward(self, grad_output):
		arr = self.context["arr"]
		return (grad_output * 1 / arr,)

class ReLU(Operation):

	def backward(self, grad_output):
		r = self.context["arr"] > 0
		return (r * grad_output,)

class Tanh(Operation):

	def backward(self, grad_output):
		tanh = self.context["tanh"]
		return ((1 - tanh * tanh) * grad_output,)

class Cos(Operation):

	def backward(self, grad_output):
		arr = self.context["arr"]
		return (-np.sin(arr) * grad_output,)

class Sin(Operation):

	def backward(self, grad_output):
		arr = self.context["arr"]
		return (np.cos(arr) * grad_output,)


class Clip(Operation):

	def backward(self, grad_output):
		arr = self.context["arr"]
		low = self.context["low"]
		high = self.context["high"]
		mask = (arr >= low) & (arr <= high)
		return (grad_output * mask.astype(grad_output.dtype),)

class Min(Operation):

	def backward(self, grad_output):
		arr1 = self.context["arr1"]
		arr2 = self.context["arr2"]
		lt_mask = arr1 < arr2
		gt_mask = arr1 > arr2
		eq_mask = ~(lt_mask | gt_mask)
		half = np.array(0.5, dtype=grad_output.dtype)
		grad1 = grad_output * (lt_mask.astype(grad_output.dtype) + half * eq_mask.astype(grad_output.dtype))
		grad2 = grad_output * (gt_mask.astype(grad_output.dtype) + half * eq_mask.astype(grad_output.dtype))
		return (grad1, grad2)

class Slice(Operation):

	def backward(self, grad_output):
		key = self.context["key"]
		shape = self.context["shape"]
		dtype = self.context["dtype"]
		grad = np.zeros(shape, dtype=dtype)

		def _is_advanced_index(index):
			if isinstance(index, (list, np.ndarray)):
				return True
			if isinstance(index, tuple):
				return any(isinstance(i, (list, np.ndarray)) for i in index)
			return False

		if _is_advanced_index(key):
			np.add.at(grad, key, grad_output)
		else:
			grad[key] = grad_output
		return (grad,)

class Reshape(Operation):

	def backward(self, grad_output):
		grad = grad_output.reshape(self.context["shape"])
		return (grad,)

class ExpandDims(Operation):
	# Je crois que c'est juste un squeeze le reverse, pas un sum
	def backward(self, grad_output):
		grad = np.squeeze(grad_output, axis=self.context["dim"])
		return (grad,)

class Squeeze(Operation):

	def backward(self, grad_output):
		grad = np.expand_dims(grad_output, axis=self.context["dim"])
		return (grad,)

class SwapDims(Operation):

	# Je crois que c'est juste un squeeze le reverse, pas un sum
	def backward(self, grad_output):
		grad = np.swapaxes(grad_output, axis1=self.context["dim2"], axis2=self.context["dim1"])
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
		input_array = self.context["arr"]      # float64
		logsumexp = self.context["logsumexp"]  # float64
		dim = self.context["dim"]

		# Choose a dtype safely (grad_output may be None)
		dtype = grad_output.dtype if grad_output is not None else input_array.dtype

		softmax = LogSumExp._softmax_from_logsumexp(input_array, logsumexp, dim, dtype)

		# Ensure grad_output has the correct shape for broadcasting
		if grad_output.ndim < input_array.ndim:
			grad_output = np.expand_dims(grad_output, axis=dim)
		return (softmax * grad_output,)
		
class BinaryCrossEntropy(Operation):

	def backward(self, grad_output):
		logits = self.context["logits"]
		target = self.context["target"]
		dtype = grad_output.dtype if grad_output is not None else logits.dtype
		target = target.astype(dtype, copy=False)
		# Stable sigmoid to avoid overflow for large |logits|
		sigmoid = np.where(
			logits >= 0,
			1 / (1 + np.exp(-logits)),
			np.exp(logits) / (1 + np.exp(logits))
		).astype(dtype, copy=False)
		grad = sigmoid - target
		return (grad * grad_output,)
		
class Copy(Operation):

	def backward(self, grad_output):
		dtype = self.context["dtype"]
		if dtype != grad_output.dtype:
			return (grad_output.astype(dtype, copy=False),)
		else:
			return (grad_output,)
