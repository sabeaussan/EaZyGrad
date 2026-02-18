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
	# TODO : add test for momentum and dampening
	def __init__(self, parameters, lr=1e-3, momentum=0.0, dampening=0.0):
		super().__init__(parameters, lr)
		self.momentum = momentum
		self.dampening_bar = 1-dampening
		if momentum > 0:
			self.buffer = [None] * len(parameters)
			self.first_iter = True

	def _get_grad(self, grad, idx):
		if self.momentum > 0:
			if self.first_iter:
				self.buffer[idx] = grad.copy()
			else:
				self.buffer[idx] *= self.momentum
				self.buffer[idx] += (self.dampening_bar*grad)
			return self.buffer[idx]
		else:
			# simple SGD
			return grad
		

	def step(self):
		for idx, p in enumerate(self.parameters):
			if p.grad is None:
				continue
			self.set_writeable_flag(p._array)
			p._array -= self.lr * self._get_grad(p.grad, idx) 