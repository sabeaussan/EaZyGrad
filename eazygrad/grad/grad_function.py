import numpy as np
import time
#from tensor import Tensor
from pytensor.utils import is_number, _cross_correlation_fft, compute_padding_value, dilate_array

# Return always (grad,) because otherwise iterate over multiple of the grad if it is a list

# Primitive operations operating on Tensor. Define a single node in the computation graph

__all__ = [
    "Operation",
    "AddBackward",
    "SubBackward",
    "MulBackward",
    "DivBackward",
    "RDivBackward",
    "PowBackward",
    "MatMulBackward",
    "SumBackward",
    "MeanBackward",
    "ExpBackward",
    "LogBackward",
    "ReLUBackward",
    "CosBackward",
    "SinBackward",
    "SliceBackward",
    "ReshapeBackward",
    "ExpandDimsBackward",
    "SqueezeBackward",
    "SwapDimsBackward",
    "PadBackward",
    "Conv2dBackward",
    "MaxPool2dBackward",
    "VarBackward",
    "BatchNorm2dBackward",
]

class Operation:

	# stateful 

	def __init__(self, *operands): # operands -> args
		self.operands = []
		for arg in operands:
			self.operands.append(arg)

	def backward(self, grad_output):
		# return the result of the grad_output-jacobian product
		raise NotImplementedError

class Add(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (np.ones_like(self.operands[0], dtype=np.float32), np.ones_like(self.operands[0], dtype=np.float32))
		return (grad_output, grad_output)

class Sub(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (np.ones_like(self.operands[0], dtype=np.float32), -np.ones_like(self.operands[0], dtype=np.float32))
		return (grad_output, -grad_output)

class Mul(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (self.operands[1] * np.ones_like(self.operands[1], dtype=np.float32), self.operands[0]* np.ones_like(self.operands[0], dtype=np.float32))
		#print(np.sum(self.operands[1] * grad_output, axis=(0,2,3)))
		return (self.operands[1] * grad_output, self.operands[0] * grad_output)

class Div(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (1/self.operands[1], - (self.operands[0]/self.operands[1]**2))
		return (1/self.operands[1] * grad_output, - (self.operands[0]/self.operands[1]**2) * grad_output)

class RDiv(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (-self.operands[1]/(self.operands[0]**2), None)
		return (-self.operands[1]/(self.operands[0]**2) * grad_output, None)

class Pow(Operation):

	def backward(self, grad_output):
		if grad_output is None:
			return (self.operands[1] * (self.operands[0]**(self.operands[1]-1)),)	
		return (self.operands[1] * (self.operands[0]**(self.operands[1]-1)) * grad_output,)


class MatMul(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (np.swapaxes(self.operands[1],-1,-2), np.swapaxes(self.operands[0],-1,-2))
		else :

			return (grad_output@np.swapaxes(self.operands[1],-1,-2), np.swapaxes(self.operands[0],-1,-2)@grad_output)


class Sum(Operation):
	# rajouter la prise en charge de keepdims = False car on doit récupérer la dimension squeeze afin de correctement backprop
	def backward(self, grad_output):
		if grad_output is None :
			return (np.ones_like(self.operands[0], dtype=np.float32),)
		else :
			return (grad_output*np.ones_like(self.operands[0], dtype=np.float32),)

class Mean(Operation):
	# rajouter la prise en charge de keepdims = False car on doit récupérer la dimension squeeze afin de correctement backprop
	def backward(self, grad_output):
		if grad_output is None :
			return (np.ones(self.operands[0], dtype=np.float32)*self.operands[1],)
		else :
			return (grad_output*np.ones(self.operands[0], dtype=np.float32)*self.operands[1],)


class Exp(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (np.exp(self.operands[0]),)
		return (grad_output * np.exp(self.operands[0]),)

class Log(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (1/self.operands[0],) 
		return (grad_output * 1/self.operands[0],)

class ReLU(Operation):

	def backward(self, grad_output):
		r = self.operands[0] > 0
		if grad_output is None :
			return (r,)
		return (r * grad_output,)

class Cos(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (-np.sin(self.operands[0]),)
		else :
			return (-np.sin(self.operands[0]) * grad_output,)

class Sin(Operation):

	def backward(self, grad_output):
		if grad_output is None :
			return (np.cos(self.operands[0]),)
		else :
			return (np.cos(self.operands[0]) * grad_output,)

class Slice(Operation):

	def backward(self, grad_output):
		key = self.operands[1]
		grad = np.zeros_like(self.operands[0], dtype=np.float32)
		if grad_output is None :
			grad[key] = 1
		else :
			grad[key] = grad_output
		return (grad,)

class Reshape(Operation):

	def backward(self, grad_output):
		grad = grad_output.reshape(self.operands[0])
		return (grad,)

class ExpandDims(Operation):
	# Je crois que c'est juste un squeeze le reverse, pas un sum
	def backward(self, grad_output):
		grad = np.squeeze(grad_output, axis = self.operands[0])
		return (grad,)

class Squeeze(Operation):

	def backward(self, grad_output):
		grad = np.expand_dims(grad_output, axis = self.operands[0])
		return (grad,)

class SwapDims(Operation):

	# Je crois que c'est juste un squeeze le reverse, pas un sum
	def backward(self, grad_output):
		grad = np.swapaxes(grad_output, axis1 = self.operands[1], axis2 = self.operands[0])
		return (grad,)

class Pad(Operation):

	def backward(self, grad_output):
		if isinstance(self.operands[0], int):
			# 1-d padding
			if self.operands[1]==0:
				return (grad_output[self.operands[0]:None],)
			else:	
				return (grad_output[self.operands[0]:-self.operands[1]],)

		# n-d padding
		slices = []
		for pad_dims in self.operands: 
			if pad_dims[1]==0:
				slices.append(slice(pad_dims[0],None))
			else:	
				slices.append(slice(pad_dims[0],-pad_dims[1]))
		return (grad_output[tuple(slices)],)
	

class Conv2d(Operation):

	def __init__(self, *operands):
		# Call the parent class constructor
		super().__init__(*operands)

		# TODO: inutile car on l'instancie à chaque forward cette objet
		# Initialize grad_input using the shape from operands[0]
		"""self.out_prod_grad_weight = np.zeros((32, 16, 64, 33), dtype='complex64')
		self.out_prod_grad_input = np.zeros((150, 16, 66, 34), dtype='complex64')"""

	def backward(self, grad_output):
		input_ = self.operands[0]
		weight = self.operands[1]
		padding = self.operands[2]
		stride = self.operands[3]

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
		grad_input = np.zeros(self.operands[0], dtype='float32')
		max_indices = self.operands[1]
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
		x_centered = self.operands[0]
		div = self.operands[1]
		dim = self.operands[2]
		return (VarBackward.d_input(grad_output, x_centered,div, dim),)


class BatchNorm2d(Operation):


	def backward(self, grad_output):
		x_centered = self.operands[0]
		var_eps = self.operands[1]
		B, C, H, W = x_centered.shape
		div = B * H * W
		dim = (0,2,3)

		a = var_eps**(-1/2)*grad_output # dl/dx
		b = - (var_eps**(-1/2) * grad_output).sum(axis=dim, keepdims=True) * 1/div # 'dl/dmean(x)*dmean(x)/dx'
		c = (-0.5*grad_output*x_centered*var_eps**(-3/2)).sum(axis=dim, keepdims=True) # 'dl/dvar(x)
		d = VarBackward.d_input(c, x_centered, div, dim) # dvar(x)/dx
		return (a+b+d,)
