from __future__ import annotations

import eazygrad as ez
from .module import Module
import numpy as np
from eazygrad import _Tensor

class Linear(Module):
	"""
	Fully connected linear layer.

	Parameters
	----------
	n_in : int
		Number of input features.
	n_out : int
		Number of output features.
	bias : bool, default=True
		Whether to include a learnable bias term.
	requires_grad : bool, default=True
		Whether the layer parameters should participate in automatic
		differentiation.

	Notes
	-----
	The layer stores weights with shape ``(n_in, n_out)`` and applies the
	transformation ``x @ weights + bias`` to batched 2D inputs.
	"""

	def __init__(self, n_in: int, n_out: int, bias: bool = True, requires_grad: bool = True) -> None:
		super().__init__()
		self.n_in = n_in
		self.n_out = n_out
		gain = np.float32(np.sqrt(1/self.n_in))
		self.weights = ez.uniform(n_in, n_out, low=-gain, high=gain, requires_grad=requires_grad)
		self.register_params(self.weights)

		self.bias = None
		if bias:
			self.bias = ez.uniform(1, n_out, low=-gain, high=gain,requires_grad=requires_grad)
			self.register_params(self.bias)

	def forward(self, x: _Tensor) -> _Tensor:
		if not isinstance(x, _Tensor):
			raise TypeError(f"Expected input to be an eazygrad tensor, got {type(x)}")
		if x.ndim == 1:
			raise ValueError("Input should be a 2D array with shape (batch_size, n_in), got 1D array with shape {}".format(x.shape))
		y = x.matmul(self.weights)
		if self.bias:
			y = y + self.bias
		return y


	def __repr__(self) -> str:
		a = super().__repr__()
		return f"------> n_in : {self.n_in}  |  n_out : {self.n_out}"
