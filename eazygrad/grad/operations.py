import numpy as np
import time
#from tensor import Tensor
from pytensor.utils import  _cross_correlation_fft, compute_padding_value, dilate_array

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
    "Cos",
    "Sin",
    "Slice",
    "Reshape",
    "ExpandDims",
    "Squeeze",
    "SwapDims",
    "Pad",
    "Conv2d",
    "MaxPool2d",
    "Var",
    "BatchNorm2d",
]

class Operation:

	# stateful 

	def __init__(self, *context): # context -> args
		self.context = []
		for ctx in context:
			if isinstance(ctx, np.ndarray):
				# protection against in-place ops
				ctx.flags.writeable=False
			self.context.append(ctx)

	def backward(self, grad_output):
		# return the result of the grad_output-jacobian product
		raise NotImplementedError
	
	def __repr__(self):
		return self.__class__.__name__

# Bizarre Add et Sub en terme de args
class Add(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (np.ones(self.context[0], dtype=np.float32), np.ones(self.context[1], dtype=np.float32))
		return (grad_output, grad_output)

class Sub(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (np.ones(self.context[0], dtype=np.float32), -np.ones(self.context[1], dtype=np.float32))
		return (grad_output, -grad_output)

class Mul(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (self.context[1] * np.ones_like(self.context[1], dtype=np.float32), self.context[0]* np.ones_like(self.context[0], dtype=np.float32))
		return (self.context[1] * grad_output, self.context[0] * grad_output)

class Div(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (1/self.context[1], - (self.context[0]/self.context[1]**2))
		return (1/self.context[1] * grad_output, - (self.context[0]/self.context[1]**2) * grad_output)

class RDiv(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (-self.context[1]/(self.context[0]**2), None)
		return (-self.context[1]/(self.context[0]**2) * grad_output, None)

class Pow(Operation):

	def backward(self, grad_output):
		if grad_output is None:
			return (self.context[1] * (self.context[0]**(self.context[1]-1)),)	
		return (self.context[1] * (self.context[0]**(self.context[1]-1)) * grad_output,)


class MatMul(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (np.swapaxes(self.context[1],-1,-2), np.swapaxes(self.context[0],-1,-2))
		else :
			return (grad_output@np.swapaxes(self.context[1],-1,-2), np.swapaxes(self.context[0],-1,-2)@grad_output)


class Sum(Operation):
	# TODO : rajouter la prise en charge de keepdims = False car on doit récupérer la dimension squeeze afin de correctement backprop
	def backward(self, grad_output):
		dim, keepdims = self.context[1]
		if grad_output is None :
			return (np.ones(self.context[0], dtype=np.float32),)
		else :
			# Check keepdims
			if not keepdims:
				grad_output = np.expand_dims(grad_output, axis=dim)
			return (grad_output*np.ones(self.context[0], dtype=np.float32),)

class Mean(Operation):
	# TODO : rajouter la prise en charge de keepdims = False car on doit récupérer la dimension squeeze afin de correctement backprop
	def backward(self, grad_output):
		dim, keepdims = self.context[2]
		if grad_output is None :
			return (np.ones(self.context[0], dtype=np.float32)*self.context[1],)
		else :
			# Check keepdims
			if not keepdims:
				grad_output = np.expand_dims(grad_output, axis=dim)
			return (grad_output*np.ones(self.context[0], dtype=np.float32)*self.context[1],)


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
		grad = np.zeros(self.context[0], dtype=np.float32)
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

	def backward(self, grad_output):
		# Type promotion for exp and sum
		input_array = self.context[0] #float64
		logsumexp = self.context[1] #float64
		dim = self.context[2]
		dtype = grad_output.dtype
		if logsumexp.ndim < input_array.ndim:
			# logsumexp was squeezed (keepdims = False)
			logsumexp = np.expand_dims(logsumexp, axis=dim)
		shifted_input = input_array - logsumexp
		softmax = np.exp(shifted_input).astype(dtype, copy=False)  # shape broadcastable to input_array

		if grad_output is None:
			return (softmax,)
		else:
			# Ensure grad_output has the correct shape for broadcasting
			if grad_output.ndim < input_array.ndim: 
				# logsumexp was squeezed (keepdims = False)
				grad_output = np.expand_dims(grad_output, axis=dim)
			return (softmax * grad_output,)
		
class Copy(Operation):

	def backward(self, grad_output):
		dtype = self.context[0]
		if dtype != grad_output.dtype:
			return (grad_output.astype(dtype, copy=False),)
		else:
			return (grad_output,)

class Pad(Operation):

	def backward(self, grad_output):
		if isinstance(self.context[0], int):
			# 1-d padding
			if self.context[1]==0:
				return (grad_output[self.context[0]:None],)
			else:	
				return (grad_output[self.context[0]:-self.context[1]],)

		# n-d padding
		slices = []
		for pad_dims in self.context: 
			if pad_dims[1]==0:
				slices.append(slice(pad_dims[0],None))
			else:	
				slices.append(slice(pad_dims[0],-pad_dims[1]))
		return (grad_output[tuple(slices)],)
	

class Conv2d(Operation):

	def backward(self, grad_output):
		input_ = self.context[0]
		weight = self.context[1]
		padding = self.context[2]
		stride = self.context[3]

		# grad w.r.t weight matrix
		grad_weight = self.backward_cross_correlation_weights(input_, grad_output, stride=stride, output_size=weight.shape[-1], padding=padding)
		grad_input = self.backward_cross_correlation_inputs(grad_output, weight, padding=padding, stride=stride, output_size=input_.shape[-1])
		return (grad_input, grad_weight)	


	def _process_grad_shape(self, grad, padding, stride, weight_shape):
		added_padding=-(2 * padding - weight_shape) % stride
		if stride > 1:
			grad = dilate_array(grad, stride)
		pad = weight_shape-padding-1
		padded_grad = np.pad(grad, ((0,0),(0,0), (pad, pad+added_padding), (pad, pad+added_padding)))
		return padded_grad


	def backward_cross_correlation_weights(self, input_, weight, padding=0, stride=1, output_size=None, out_product=None):
		if stride>1:
			weight = dilate_array(weight, stride)

		if padding >0:
			input_ = np.pad(input_, ((0,0),(0,0), (padding, padding), (padding, padding)))

		B, C, H, W = input_.shape
		out_prod = np.zeros((weight.shape[1], C, H, (W//2)+1), dtype='complex64')
		print("weight : ", out_prod.shape)

		return _cross_correlation_fft(input_=input_, kernel=weight, stride=1, output_size=output_size, out_prod=out_prod, backward_weight=True)
	
	def backward_cross_correlation_inputs(self, input_, weight, padding, stride, output_size=None):
		g = self._process_grad_shape(input_, padding, stride, weight.shape[-1])
		k = np.flip(weight, axis=(-2,-1))  # <- renvoie un array non contigue ???!
		B, C, H, W = g.shape
		out_prod = np.zeros((B, weight.shape[1], H, (W//2)+1), dtype='complex64')
		return _cross_correlation_fft(input_=g, kernel=np.ascontiguousarray(k).transpose(1,0,2,3), stride=1, output_size=output_size, out_prod=out_prod)

class MaxPool2d(Operation):


	def backward(self, grad_output):
		grad_input = np.zeros(self.context[0], dtype='float32')
		max_indices = self.context[1]
		grad_input[max_indices[0], max_indices[1], max_indices[2], max_indices[3]] = grad_output.flatten()
		return (grad_input,)

class Var(Operation):

	@staticmethod
	def d_input(grad_output, x_centered, div, dim):
		d1 = grad_output/div * 2 * x_centered
		d2 = d1.sum(axis=dim, keepdims=True)/div
		grad_input = d1 + d2
		return grad_input


	def backward(self, grad_output):
		x_centered = self.context[0]
		div = self.context[1]
		dim = self.context[2]
		return (VarBackward.d_input(grad_output, x_centered,div, dim),)


class BatchNorm2d(Operation):


	def backward(self, grad_output):
		x_centered = self.context[0]
		var_eps = self.context[1]
		B, C, H, W = x_centered.shape
		div = B * H * W
		dim = (0,2,3)

		a = var_eps**(-1/2)*grad_output # dl/dx
		b = - (var_eps**(-1/2) * grad_output).sum(axis=dim, keepdims=True) * 1/div # 'dl/dmean(x)*dmean(x)/dx'
		c = (-0.5*grad_output*x_centered*var_eps**(-3/2)).sum(axis=dim, keepdims=True) # 'dl/dvar(x)
		d = VarBackward.d_input(c, x_centered, div, dim) # dvar(x)/dx
		return (a+b+d,)
