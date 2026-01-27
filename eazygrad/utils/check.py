import numpy as np
import numbers
import fractions

VALID_DTYPES = frozenset({
    np.float32, np.float64, 
	np.int32, np.int64, np.uint8
})

def input_array_type(array, dtype):
	dtype = np.dtype(dtype).type if dtype is not None else None
	if dtype is not None and (dtype not in VALID_DTYPES):
		raise TypeError(f"Specified dtype not supported : {dtype}. List of supported dtypes : {VALID_DTYPES}")
	if isinstance(array, np.ndarray):
		if 0 in array.shape:
			raise ValueError("Array with at least one of the dimension empty is not supported")
		if dtype is None:
			dtype = array.dtype
		if array.dtype != dtype:
			array = array.astype(dtype)
		
	elif isinstance(array, list):
		if len(array) == 0:
			raise ValueError("Empty list as input is not supported")
		if dtype is None:
			dtype = np.float32
		array = np.array(array, dtype=dtype)
	elif is_scalar(array):
		try:
			dtype = array.dtype
		except AttributeError:
			dtype = None
		# Default dtype is float32
		if dtype is None:
			dtype = np.float32
		array = np.array(array, dtype=dtype)
	else:
		raise TypeError(f"The array field should be either a numpy array or a list with a homogenous shape, not a {type(array)}.")
	return array

	
def is_scalar(a):
	return isinstance(a, numbers.Real) and not isinstance(a, (bool, fractions.Fraction))

def check_tensors_type(func):
	def wrapper(*args, **kwargs):
		if is_scalar(input):
			raise TypeError("Expected a tensor")

def broadcasted_shape(grad, tensor):
	"""
		Sum along expanded axis
		Used to correct for broadcasted shape during backpropagation
	"""
	
	# Check expanded dims
	num_expanded_dims = grad.ndim - tensor._array.ndim

	if num_expanded_dims > 0:
		# float64 promotion for reduction
		grad = grad.astype(np.float64, copy=False)
		# Sum over expanded dims
		grad = np.sum(grad, tuple(np.arange(num_expanded_dims)))

	# grad and tensor have now the same amount of dims
	# Dims where the original tensor had size 1 (broadcasted)
	broadcasted_dims = [i for i, (gs, ts) in enumerate(zip(grad.shape, tensor.shape)) if ts == 1 and gs != 1]
	if broadcasted_dims:
		# float64 promotion for reduction
		grad = grad.astype(np.float64, copy=False)
		grad = np.sum(grad, tuple(broadcasted_dims), keepdims = True)
	return grad.astype(np.float32)