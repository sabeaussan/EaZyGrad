from .functions import *
from .tensor_factories import *

# /!\ TODO : add equal func

# TODO : add ez no grad
# TODO : prevent type promotion
# TODO : add profiling decorator

# TODO : add more robust matmul tests

class no_grad:
	"""
	Context manager to disable gradient tracking.
	"""
	def __enter__(self):
		self.prev_state = dag.enable_grad
		dag.enable_grad = False

	def __exit__(self, exc_type, exc_value, traceback):
		dag.enable_grad = self.prev_state