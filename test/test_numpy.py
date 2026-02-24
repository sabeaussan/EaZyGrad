import pytest
from test_utils import make_tensor
import numpy as np
import eazygrad as ez

@pytest.mark.parametrize("input, dtype", [
    (np.array(10.0), None),
    (np.array([0.0]), np.float32),
    (np.random.randn(10,20), np.int64),
    ([-5,4,3], np.float64), 
    (10, None),             
])
def test_array_interface(input, dtype):
	t = ez.tensor(input)
	input = np.array(input, dtype=dtype)
	
	np.testing.assert_allclose(np.asarray(t, dtype=dtype), input, rtol=1e-4, atol=1e-4)