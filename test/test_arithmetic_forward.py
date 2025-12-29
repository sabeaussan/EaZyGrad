import numpy as np
import pytest
from hypothesis import given, strategies as st
import test_utils


# ---- ADDITION ----
@given(array=test_utils.array_strategy, scalar=test_utils.scalar_strategy)
def test_add_scalar(array, scalar):
    a = test_utils.make_tensor(array)
    b = a + scalar
    np.testing.assert_array_equal(b._array, array + scalar)


@given(arrays=test_utils.array_pair_broadcast_compat_strategy)
def test_add_tensor(arrays):
    a, b = map(test_utils.make_tensor, arrays)
    c = a + b
    np.testing.assert_array_equal(c._array, arrays[0] + arrays[1])


# ---- SUBTRACTION ----
@given(array=test_utils.array_strategy, scalar=test_utils.scalar_strategy)
def test_sub_scalar(array, scalar):
    a = test_utils.make_tensor(array)
    b = a - scalar
    np.testing.assert_array_equal(b._array, array - scalar)


@given(arrays=test_utils.array_pair_broadcast_compat_strategy)
def test_sub_tensor(arrays):
    a, b = map(test_utils.make_tensor, arrays)
    c = a - b
    np.testing.assert_array_equal(c._array, arrays[0] - arrays[1])


@given(array=test_utils.array_strategy, scalar=test_utils.scalar_strategy)
def test_rsub_scalar(array, scalar):
    a = test_utils.make_tensor(array)
    b = scalar - a
    np.testing.assert_array_equal(b._array, scalar - array)


# ---- MULTIPLICATION ----
@given(array=test_utils.array_strategy, scalar=test_utils.scalar_strategy)
def test_mul_scalar(array, scalar):
    a = test_utils.make_tensor(array)
    b = a * scalar
    np.testing.assert_array_equal(b._array, array * scalar)


@given(arrays=test_utils.array_pair_broadcast_compat_strategy)
def test_mul_tensor(arrays):
    a, b = map(test_utils.make_tensor, arrays)
    c = a * b
    np.testing.assert_array_equal(c._array, a._array * b._array)


@given(array=test_utils.array_strategy, scalar=test_utils.scalar_strategy)
def test_rmul_scalar(array, scalar):
    a = test_utils.make_tensor(array)
    b = scalar * a
    np.testing.assert_array_equal(b._array, scalar * array)


# ---- DIVISION ----
@given(array=test_utils.array_strategy, scalar=test_utils.scalar_strategy.filter(lambda x: x != 0))
def test_truediv_scalar(array, scalar):
    a = test_utils.make_tensor(array)
    b = a / scalar
    np.testing.assert_array_equal(b._array, array / scalar)


@given(arrays=test_utils.array_pair_broadcast_compat_strategy)
def test_truediv_tensor(arrays):
    # avoid division by zero
    arr1 = arrays[0]
    arr2 = np.where(arrays[1] == 0, 1, arrays[0])
    a = test_utils.make_tensor(arr1)
    b = test_utils.make_tensor(arr2)
    c = a / b
    np.testing.assert_array_equal(c._array, arr1 / arr2)


@given(array=test_utils.array_strategy)
def test_rtruediv_scalar(array):
    a = test_utils.make_tensor(np.where(array == 0, 1, array))  # avoid zero
    scalar = 8
    b = scalar / a
    np.testing.assert_array_equal(b._array, scalar / a._array)


# ---- POWER ----
@given(array=test_utils.array_strategy, integer=test_utils.int_strategy)
def test_pow_scalar(array, integer):
    # negative scalars with fractionnal power will produce nan
    a = test_utils.make_tensor(array)
    b = a ** integer
    np.testing.assert_array_equal(b._array, array ** integer)


@given(arrays=test_utils.array_pair_broadcast_compat_strategy)
def test_pow_tensor_not_supported(arrays):
    a, b = map(test_utils.make_tensor, arrays)
    with pytest.raises(NotImplementedError):
        _ = a ** b

# ---- MATMUL ----
@given(arrays=test_utils.array_pair_matmul_compat_strategy)
def test_matmul(arrays):
    a, b = map(test_utils.make_tensor, arrays)
    r = a@b
    r_ = a.numpy()@b.numpy()
    np.testing.assert_array_equal(r.numpy(), r_)

@pytest.mark.parametrize("array1, array2", [
    (np.array(10.0), np.array([10.0])),
    (np.array([10.0]), np.array(10.0)),
    (np.random.randn(5,5), np.array(10.0)),
    (np.array(10.0), np.random.randn(5,5)),
])
def test_matmul_invalid_input(array1, array2):
    a, b = map(test_utils.make_tensor, [array1, array2])
    with pytest.raises(RuntimeError):
        a@b


# ---- MEAN ----
@given(
    array=test_utils.array_strategy,
    keepdims=st.booleans(),
    data=st.data(),
)
def test_mean_forward(array, keepdims, data):
    dim_arg = _random_axes(array, data)
    tensor = test_utils.make_tensor(array)
    result = tensor.mean(dim=dim_arg, keepdims=keepdims)

    axes = dim_arg
    if axes is None:
        axes = tuple(range(array.ndim))
    elif isinstance(axes, int):
        axes = (axes,)

    expected = (
        array.astype(np.float64, copy=False)
        .mean(axis=axes, keepdims=keepdims)
        .astype(np.float32)
    )
    np.testing.assert_allclose(result._array, expected, rtol=1e-5, atol=1e-6)


# ---- SUM ----
@given(
    array=test_utils.array_strategy,
    keepdims=st.booleans(),
    data=st.data(),
)
def test_sum_forward(array, keepdims, data):
    dim_arg = _random_axes(array, data)
    tensor = test_utils.make_tensor(array)
    result = tensor.sum(dim=dim_arg, keepdims=keepdims)

    axes = dim_arg
    if axes is None:
        axes = tuple(range(array.ndim))
    elif isinstance(axes, int):
        axes = (axes,)

    expected = (
        array.astype(np.float64, copy=False)
        .sum(axis=axes, keepdims=keepdims)
        .astype(np.float32)
    )
    np.testing.assert_allclose(result._array, expected, rtol=1e-5, atol=1e-6)