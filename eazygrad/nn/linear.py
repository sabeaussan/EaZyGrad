import eazygrad as ez
from .module import Module
import numpy as np
from eazygrad import _Tensor
import line_profiler

class Linear(Module):

	def __init__(self, n_in, n_out, bias=True, requires_grad=True):
		super().__init__()
		self.n_in = n_in
		self.n_out = n_out
		gain = np.float32(np.sqrt(1/self.n_in))
		self.weights = ez.uniform(low=-gain, high=gain, shape=(n_in, n_out), requires_grad=requires_grad)
		self.register_params(self.weights)

		self.bias = None
		if bias:
			self.bias = ez.uniform(low=-gain, high=gain, shape=(1, n_out), requires_grad=requires_grad)
			self.register_params(self.bias)

	def forward(self,x):
		if not isinstance(x, _Tensor):
			raise TypeError(f"Expected input to be an eazygrad tensor, got {type(input)}")
		if x.ndim == 1:
			raise ValueError("Input should be a 2D array with shape (batch_size, n_in), got 1D array with shape {}".format(x.shape))
		y = x.matmul(self.weights)
		if self.bias:
			y = y + self.bias
		return y


	def __repr__(self):
		a = super().__repr__()
		return f"------> n_in : {self.n_in}  |  n_out : {self.n_out}"

if __name__ == "__main__":
	a = Linear(10,15)
	x = ez.randn((1,10))
	print(a(x).shape)
