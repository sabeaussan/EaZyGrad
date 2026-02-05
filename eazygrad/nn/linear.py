import eazygrad as ez
from .module import Module
import numpy as np
import functools

class Linear(Module):

	def __init__(self, n_in, n_out, bias=True, requires_grad=True):
		self.n_in = n_in
		self.n_out = n_out
		self.out = None
		gain = np.float32(np.sqrt(1/self.n_in))
		self.parameters = [ez.uniform(low=-gain, high=gain, shape=(n_in, n_out), requires_grad=requires_grad)]
		if bias:
			self.parameters.append(ez.uniform(low=-gain, high=gain, shape=(1, n_out), requires_grad=requires_grad))



	# TODO : add a decorator to cache the output array and avoid creating a new one at each forward pass if the shape is the same
	def forward(self,x):
		if self.out is None or self.out.shape[0] < x.shape[0]:
			# TODO : extned out instead of reallocating it if the shape is smaller
			self.out = np.empty((x.shape[0], self.n_out), dtype=np.float32)
		y = x.matmul(self.parameters[0], out=self.out[:x.shape[0]]) #x@self.parameters[0]
		if len(self.parameters)>1:
			y += self.parameters[1]
		return y


	def __repr__(self):
		a = super().__repr__()
		return f"------> n_in : {self.n_in}  |  n_out : {self.n_out}"

if __name__ == "__main__":
	a = Linear(10,15)
	x = ez.randn((1,10))
	print(a(x).shape)

