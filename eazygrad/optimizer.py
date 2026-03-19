from __future__ import annotations

from typing import Sequence

import numpy as np
from ._tensor import _Tensor
import sys

class Optimizer:

	def __init__(self, parameters: Sequence[_Tensor], lr: float = 1e-3) -> None:
		# print(len(parameters))
		# raise
		self.parameters = parameters
		self._check_params()
		self.lr = lr


	def _check_params(self) -> None:
		for p in self.parameters:
			if not isinstance(p, _Tensor):
				raise RuntimeError(f"Parameters should be eazygrad tensors, got {type(p)}.")


	def zero_grad(self) -> None:
		for p in self.parameters:
			p.grad = None

	def set_writeable_flag(self, array: np.ndarray) -> None:
		# Allow in-place op
		array.flags.writeable=True

	def _get_step_size(self, grad: np.ndarray, idx: int) -> np.ndarray:
		raise NotImplementedError

	def step(self) -> None:
		for idx, p in enumerate(self.parameters):
			if p.grad is None:
				continue
			# Saved forward tensors can become read-only; optimizers need to undo
			# that before applying an in-place parameter update.
			self.set_writeable_flag(p._array)
			# Needs to compute step size first
			# because AdamW decay parameters before
			# so in-place modification of p is wrong otherwise
			step_size = self._get_step_size(p.grad, idx) 
			p._array -= self.lr * step_size


class SGD(Optimizer):
	"""
	Stochastic gradient descent optimizer.

	Parameters
	----------
	parameters : sequence of _Tensor
		Iterable of tensors to optimize.
	lr : float, default=1e-3
		Learning rate.
	momentum : float, default=0.0
		Momentum factor.
	dampening : float, default=0.0
		Dampening applied to the momentum update.

	Notes
	-----
	If ``momentum`` is zero, the optimizer reduces to plain stochastic gradient
	descent. Otherwise, it maintains one momentum buffer per parameter.

	See Also
	--------
	`torch.optim.SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_
	"""
	def __init__(
		self,
		parameters: Sequence[_Tensor],
		lr: float = 1e-3,
		momentum: float = 0.0,
		dampening: float = 0.0,
	) -> None:
		super().__init__(parameters, lr)
		self.momentum = momentum
		self.dampening_bar = 1-dampening
		if momentum > 0:
			self.buffer = [None] * len(parameters)

	def _get_step_size(self, grad: np.ndarray, idx: int) -> np.ndarray:
		if self.momentum > 0:
			if self.buffer[idx] is None:
				# Match PyTorch's behavior: initialize momentum from the first grad.
				self.buffer[idx] = grad.copy()
			else:
				self.buffer[idx] *= self.momentum
				self.buffer[idx] += (self.dampening_bar*grad)
			grad = self.buffer[idx]
		return grad
		
class Adam(Optimizer):
	"""
	Adam optimizer.

	Parameters
	----------
	parameters : sequence of _Tensor
		Iterable of tensors to optimize.
	lr : float, default=1e-3
		Learning rate.
	betas : tuple of float, default=(0.9, 0.99)
		Coefficients used for the running averages of the gradient and squared
		gradient.
	eps : float, default=1e-8
		Small value added for numerical stability.

	Notes
	-----
	The optimizer maintains per-parameter first and second moment estimates and
	uses bias correction during the update step.

	See Also
	--------
	`torch.optim.Adam <https://pytorch.org/docs/stable/generated/torch.optim.Adam.html>`_
	"""

	def __init__(
		self,
		parameters: Sequence[_Tensor],
		lr: float = 1e-3,
		betas: tuple[float, float] = (0.9, 0.99),
		eps: float = 1e-8,
	) -> None:
		super().__init__(parameters, lr)
		self.betas = betas
		self.eps = eps
		self.running_mean = [np.float32(0.0)] * len(parameters)
		self.running_var = [np.float32(0.0)] * len(parameters)
		self.t_steps = [1] * len(parameters)

	def _get_step_size(self, grad: np.ndarray, idx: int) -> np.ndarray:
		# Track first and second moments independently for each parameter tensor.
		self.running_mean[idx] = self.betas[0]*self.running_mean[idx] + (1-self.betas[0])*grad
		self.running_var[idx] = self.betas[1]*self.running_var[idx] + (1-self.betas[1])*(grad**2)
		# bias correction
		corrected_running_mean = self.running_mean[idx]/(1-self.betas[0]**self.t_steps[idx])
		corrected_running_var = self.running_var[idx]/(1-self.betas[1]**self.t_steps[idx])
		self.t_steps[idx] += 1
		return corrected_running_mean/(np.sqrt(corrected_running_var)+self.eps)


class AdamW(Adam):
	"""
	AdamW optimizer with decoupled weight decay.

	Parameters
	----------
	parameters : sequence of _Tensor
		Iterable of tensors to optimize.
	lr : float, default=1e-3
		Learning rate.
	betas : tuple of float, default=(0.9, 0.99)
		Coefficients used for the running averages of the gradient and squared
		gradient.
	eps : float, default=1e-8
		Small value added for numerical stability.
	weight_decay : float, default=0.01
		Decoupled weight decay coefficient applied before the Adam update.

	See Also
	--------
	`torch.optim.AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_
	"""

	def __init__(
		self,
		parameters: Sequence[_Tensor],
		lr: float = 1e-3,
		betas: tuple[float, float] = (0.9, 0.99),
		eps: float = 1e-8,
		weight_decay: float = 0.01,
	) -> None:
		super().__init__(parameters, lr, betas, eps)
		self.weight_decay = weight_decay

	def _get_step_size(self, grad: np.ndarray, idx: int) -> np.ndarray:
		# AdamW applies weight decay outside of the adaptive moment update.
		self.parameters[idx]._array = self.parameters[idx]._array - self.lr*self.weight_decay*self.parameters[idx]._array
		return super()._get_step_size(grad, idx)


	
