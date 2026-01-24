import eazygrad as ez
from .module import Module
import numpy as np

class Linear(Module):

	def __init__(self, n_in, n_out, bias=True, requires_grad=True):
		self.n_in = n_in
		self.n_out = n_out
		gain = np.sqrt(1/self.n_in)
		self.parameters = [ez.uniform(low=-gain, high=gain, shape=(n_in, n_out), requires_grad=requires_grad)]
		if bias:
			self.parameters.append(ez.uniform(low=-gain, high=gain, shape=(1, n_out), requires_grad=requires_grad))

	def forward(self,x):
		y = x@self.parameters[0]
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

