import pytest
from hypothesis import given, strategies as st
import eazygrad
import numpy as np
import decimal
import fractions

@pytest.mark.parametrize("value, expected", [
    (5, True),                   # int
    (3.14, True),                # float
    (np.int32(7), True),         # numpy int
    (np.float64(2.1), True),     # numpy float
    (True, False),               # bool (excluded)
    (3 + 4j, False),             # complex (excluded)
    (decimal.Decimal("2.1"), False),   # decimal.Decimal excluded by numbers.Real
    (fractions.Fraction(3, 4), False), # fractions.Fraction excluded by numbers.Real
    ("42", False),               # string
    (None, False),               # None
    ([1, 2, 3], False),          # list
])
def test_is_scalar(value, expected):
    assert eazygrad.check.is_scalar(value) == expected

@pytest.mark.parametrize("array, requires_grad", [
    (3.14, False),
    ([5], True),
    ([3.14, 1, 5], False), 
    (np.random.randn(10), True), 
    (np.zeros(50), False),  
    ([3.14, 1, -5, 0], True), 
])
def test_tensor_valid(array, requires_grad):
	assert eazygrad.tensor(array, requires_grad).dtype in (np.float32, np.float64)
	assert eazygrad.tensor(array, requires_grad).requires_grad == requires_grad

@pytest.mark.parametrize("array, requires_grad", [ 
    ([], False), 
    (np.arange(0), True),
    ((3.14, 32), False), 
    (["string", True], True),
    ([np.random.randn(10), np.random.randn(5)], True), 
    ({"key" : np.arange(10)}, True), 
])
def test_tensor_invalid(array, requires_grad):
    with pytest.raises(Exception) as excinfo:
        eazygrad.tensor(array, requires_grad)
    assert isinstance(excinfo.value, (TypeError, ValueError))
	    

@pytest.mark.parametrize("shape, requires_grad", [
    ([5], True),
    ((3.14, 32), True),
    ((3, -32), True),
    ((3, 0), True),
    ([3.14, 1, 5], False),
    (3.14, False), 
    (np.arange(10), True),
    (["string", True], True),
    ([np.random.randn(10), np.random.randn(5)], True), 
    ({"key" : np.arange(10)}, True), 
])
def test_tensor_factories_invalid(shape, requires_grad):
    for factory in [eazygrad.randn, eazygrad.ones, eazygrad.zeros]:
        with pytest.raises(Exception) as excinfo:
            factory(shape, requires_grad)
        assert isinstance(excinfo.value, (TypeError, ValueError))

@pytest.mark.parametrize("shape, requires_grad", [
    ((3, 32), True),
    ((3,), False),
    ((1,), False), 
])
def test_tensor_factories_valid(shape, requires_grad):
    # test randn, empty, ones, zeros
    for factory in [eazygrad.randn, eazygrad.ones, eazygrad.zeros]:
        assert factory(shape, requires_grad)._array.shape == shape

def test_uniform_valid():
    shape = (3, 4)
    low = -6.0
    high = 5.2
    ez = eazygrad.uniform(low, high, shape, requires_grad=True)
    assert isinstance(ez, eazygrad._Tensor)
    assert ez._array.shape == shape
    assert ez.requires_grad is True
    assert np.all(ez._array >= low)
    assert np.all(ez._array <= high)

@pytest.mark.parametrize("low, high", [
    ("a", 1.0),
    (0.0, "b"),
    (None, 1.0),
    ([], {}),
])
def test_uniform_invalid_bounds(low, high):
    with pytest.raises(TypeError):
        eazygrad.uniform(low, high, (2, 2))

# ------------------------
# Valid input test cases
# ------------------------

@pytest.mark.parametrize("factory", [eazygrad.randn, eazygrad.ones, eazygrad.zeros])
@pytest.mark.parametrize("shape, requires_grad", [
    ((3,), False),
    ((2, 2), True),
    ((1, 3, 5), False)
])
def test_factory_valid_inputs(factory, shape, requires_grad):
    ez = factory(shape, requires_grad=requires_grad)
    assert ez.shape == shape
    assert ez.requires_grad == requires_grad
    assert isinstance(ez, eazygrad._Tensor)

# ------------------------
# TypeError: shape is not a tuple
# ------------------------

@pytest.mark.parametrize("factory", [eazygrad.randn, eazygrad.ones, eazygrad.zeros])
@pytest.mark.parametrize("bad_shape", [
    [3],          # list
    3,            # int
    "3,3",        # string
    None
])
def test_factory_type_error(factory, bad_shape):
    with pytest.raises(TypeError):
        factory(bad_shape)

# ------------------------
# ValueError: shape contains a 0
# ------------------------

@pytest.mark.parametrize("factory", [eazygrad.randn, eazygrad.ones, eazygrad.zeros])
def test_factory_zero_dim(factory):
    with pytest.raises(ValueError):
        factory((3, 0, 2))

# =============== Property tests ===============
@given(st.integers() | st.floats(allow_nan=False, allow_infinity=False))
def test_is_number_with_numbers(x):
    assert eazygrad.check.is_scalar(x) is True

@given(st.text() | st.lists(st.integers()) | st.dictionaries(st.text(), st.integers()))
def test_is_number_with_non_numbers(x):
    assert eazygrad.check.is_scalar(x) is False

@given(st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3))
def test_ones_factory_shapes(shape_list):
    shape = tuple(shape_list)
    ez = eazygrad.ones(shape)
    assert ez.shape == shape
    assert np.all(ez._array == 1.0)
    assert ez._array.dtype in (np.float32, np.float64)

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=10))
def test_tensor_from_list_matches_numpy(values):
    ez = eazygrad.tensor(values, requires_grad=True)
    expected = np.array(values, dtype=np.float32)
    np.testing.assert_array_equal(ez._array, expected)
    assert ez.requires_grad is True