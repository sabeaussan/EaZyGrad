import numpy as np
import abc


class Optimizer:

	def __init__(self, parameters, lr = 1e-3):
		self.parameters = parameters
		self.lr = lr


	def zero_grad(self):
		for p in self.parameters:
			p.grad = None

	def set_writeable_flag(self, array):
		# Allow in-place op
		array.flags.writeable=True

	def step(self):
		raise NotImplementedError


class SGD(Optimizer):

	def __init__(self, parameters, lr = 1e-1):
		super().__init__(parameters, lr)

	def step(self):
		for p in self.parameters:
			self.set_writeable_flag(p._array)
			p._array -= self.lr * p.grad 