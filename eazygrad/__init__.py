from __future__ import annotations

from typing import Any, Callable

from .functions import *
from .tensor_factories import *
from .utils import check
from ._tensor import _Tensor
from . import nn
from . import data
from .optimizer import SGD, Adam, AdamW

# TODO : add more robust matmul tests

class no_grad_ctx:
	"""
	Context manager that temporarily disables gradient tracking.

	Examples
	--------
	>>> with eazygrad.no_grad_ctx():
	...     y = x + 1.0
	"""
	def __enter__(self) -> None:
		self.prev_state = dag.grad_enable
		dag.grad_enable = False

	def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
		dag.grad_enable = self.prev_state

def no_grad(func: Callable[..., Any]) -> Callable[..., Any]:
	"""
	Decorator that disables gradient tracking inside a function.

	Parameters
	----------
	func : callable
		Function to execute with gradient tracking disabled.

	Returns
	-------
	callable
		Wrapped function.
	"""
	def wrapper(*args, **kwargs):
		with no_grad_ctx():
			return func(*args, **kwargs)
	return wrapper
