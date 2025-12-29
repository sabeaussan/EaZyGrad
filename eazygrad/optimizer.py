import numpy as np
import abc


class Optimizer:

	def __init__(self, parameters, lr = 1e-3):
		self.parameters = parameters
		self.lr = lr


	def zero_grad(self):
		for p in self.parameters:
			p.acc_grad = 0

	def step(self):
		raise NotImplementedError


class SGD(Optimizer):

	def __init__(self, parameters, lr = 1e-1):
		super().__init__(parameters, lr)

	def step(self):
		for p in self.parameters:
			p.array -= self.lr * p.acc_grad 