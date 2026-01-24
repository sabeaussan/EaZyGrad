from .functions import *
from .tensor_factories import *
from .utils import check
from ._tensor import _Tensor
import nn
# /!\ TODO : add equal func

# TODO : add ez no grad
# TODO : prevent type promotion
# TODO : add profiling decorator

# TODO : add more robust matmul tests

class no_grad_ctx:
	"""
	Context manager to disable gradient tracking.
	"""
	def __enter__(self):
		self.prev_state = dag.grad_enable
		dag.grad_enable = False

	def __exit__(self, exc_type, exc_value, traceback):
		dag.grad_enable = self.prev_state

def no_grad(func):
	"""
	Decorator to disable gradient tracking for a specific function.
	"""
	def wrapper(*args, **kwargs):
		with no_grad_ctx():
			return func(*args, **kwargs)
	return wrapper