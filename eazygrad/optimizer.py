import numpy as np
from ._tensor import _Tensor

class Optimizer:

	def __init__(self, parameters, lr = 1e-3):
		self.parameters = parameters
		self.lr = lr

	def _check_params(self):
		for p in self.parameters():
			if not isinstance(p, _Tensor):
				raise RuntimeError(f"Parameters should be eazygrad tensors, got {type(p)}.")


	def zero_grad(self):
		for p in self.parameters:
			p.grad = None

	def set_writeable_flag(self, array):
		# Allow in-place op
		array.flags.writeable=True

	def _get_step_size(self, grad, idx):
		raise NotImplementedError

	def step(self):
		for idx, p in enumerate(self.parameters):
			if p.grad is None:
				continue
			self.set_writeable_flag(p._array)
			# Needs to compute step size first
			# because AdamW decay parameters before
			# so in-place modification of p is wrong otherwise
			step_size = self._get_step_size(p.grad, idx) 
			p._array -= self.lr * step_size


class SGD(Optimizer):
	def __init__(self, parameters, lr=1e-3, momentum=0.0, dampening=0.0):
		super().__init__(parameters, lr)
		self.momentum = momentum
		self.dampening_bar = 1-dampening
		if momentum > 0:
			self.buffer = [None] * len(parameters)

	def _get_step_size(self, grad, idx):
		if self.momentum > 0:
			if self.buffer[idx] is None:
				self.buffer[idx] = grad.copy()
			else:
				self.buffer[idx] *= self.momentum
				self.buffer[idx] += (self.dampening_bar*grad)
			grad = self.buffer[idx]
		return grad
		
class Adam(Optimizer):

	def __init__(self, parameters, lr = 1e-3, betas=(0.9, 0.99), eps=1e-8):
		super().__init__(parameters, lr)
		self.betas = betas
		self.eps = eps
		self.running_mean = [np.float32(0.0)] * len(parameters)
		self.running_var = [np.float32(0.0)] * len(parameters)
		self.t_steps = [1] * len(parameters)

	def _get_step_size(self, grad, idx):
		self.running_mean[idx] = self.betas[0]*self.running_mean[idx] + (1-self.betas[0])*grad
		self.running_var[idx] = self.betas[1]*self.running_var[idx] + (1-self.betas[1])*(grad**2)
		# bias correction
		corrected_running_mean = self.running_mean[idx]/(1-self.betas[0]**self.t_steps[idx])
		corrected_running_var = self.running_var[idx]/(1-self.betas[1]**self.t_steps[idx])
		self.t_steps[idx] += 1
		return corrected_running_mean/(np.sqrt(corrected_running_var)+self.eps)


class AdamW(Adam):

	def __init__(self, parameters, lr = 1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01):
		super().__init__(parameters, lr, betas, eps)
		self.weight_decay = weight_decay

	def _get_step_size(self, grad, idx):
		# decay step
		self.parameters[idx]._array = self.parameters[idx]._array - self.lr*self.weight_decay*self.parameters[idx]._array
		return super()._get_step_size(grad, idx)


	