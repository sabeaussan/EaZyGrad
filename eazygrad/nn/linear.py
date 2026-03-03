import eazygrad as ez
from .module import Module
import numpy as np
from eazygrad import _Tensor

class Linear(Module):

	def __init__(self, n_in, n_out, bias=True, requires_grad=True):
		super().__init__()
		self.n_in = n_in
		self.n_out = n_out
		self.buffers = [None] * 2
		gain = np.float32(np.sqrt(1/self.n_in))
		self.weights = ez.uniform(low=-gain, high=gain, shape=(n_in, n_out), requires_grad=requires_grad)
		self.register_params(self.weights)

		self.bias = None
		if bias:
			self.bias = ez.uniform(low=-gain, high=gain, shape=(1, n_out), requires_grad=requires_grad)
			self.register_params(self.bias)

	def _get_buffer_shape(self, x):
		return x.shape[:-1]+(self.n_out,)
	
	# TODO : extend buffers instead of reallocating it if the size is not enough
	def forward(self,x):
		if not isinstance(x, _Tensor):
			raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")
		if x.ndim == 1:
			raise ValueError("Input should be a 2D array with shape (batch_size, n_in), got 1D array with shape {}".format(x.shape))
		allocate_buffer = self.buffers[0] is None or self.buffers[0].shape[0] < x.shape[0]
		if allocate_buffer:
			buff_shape = self._get_buffer_shape(x)
			self.buffers[0] = np.empty(buff_shape, dtype=x.dtype)
		y = x.matmul(self.weights, out=self.buffers[0][:x.shape[0]])
		if self.bias:
			if allocate_buffer:
				self.buffers[1] = np.empty(buff_shape, dtype=x.dtype)
			y = y.__add__(self.bias, out=self.buffers[1][:x.shape[0]])
		return y


	def __repr__(self):
		a = super().__repr__()
		return f"------> n_in : {self.n_in}  |  n_out : {self.n_out}"

if __name__ == "__main__":
	a = Linear(10,15)
	x = ez.randn((1,10))
	print(a(x).shape)

