import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
import test_utils
import torch

@settings(deadline=None)
@given(arrays=test_utils.array_pair_broadcast_compat_strategy)
def test_add_backward(arrays):
	a_ez, b_ez = map(test_utils.make_tensor, arrays)
	a_t = torch.tensor(arrays[0], requires_grad=True)
	b_t = torch.tensor(arrays[1], requires_grad=True)
	r_ez = a_ez + b_ez
	r_t = a_t + b_t

	grad_output = test_utils.random_grad(r_ez._array.shape)
	r_ez.backward(grad_output)
	r_t.backward(torch.tensor(grad_output))
	np.testing.assert_allclose(a_ez.grad, a_t.grad.numpy(), rtol=5e-5, atol=5e-5)
	np.testing.assert_allclose(b_ez.grad, b_t.grad.numpy(), rtol=5e-5, atol=5e-5)


@given(arrays=test_utils.array_pair_broadcast_compat_strategy)
def test_sub_backward(arrays):
    a_ez, b_ez = map(test_utils.make_tensor, arrays)
    a_t = torch.tensor(arrays[0], requires_grad=True)
    b_t = torch.tensor(arrays[1], requires_grad=True)
    r_ez = a_ez - b_ez
    r_t = a_t - b_t

    grad_output = test_utils.random_grad(r_ez._array.shape)
    r_ez.backward(grad_output)
    r_t.backward(torch.tensor(grad_output))
    np.testing.assert_allclose(a_ez.grad, a_t.grad.numpy(), rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(b_ez.grad, b_t.grad.numpy(), rtol=5e-5, atol=5e-5)


@given(arrays=test_utils.array_pair_broadcast_compat_strategy)
def test_mul_backward(arrays):
    a_ez, b_ez = map(test_utils.make_tensor, arrays)
    a_t = torch.tensor(arrays[0], requires_grad=True)
    b_t = torch.tensor(arrays[1], requires_grad=True)
    r_ez = a_ez * b_ez
    r_t = a_t * b_t

    grad_output = test_utils.random_grad(r_ez._array.shape)
    r_ez.backward(grad_output)
    r_t.backward(torch.tensor(grad_output))
    np.testing.assert_allclose(a_ez.grad, a_t.grad.numpy(), rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(b_ez.grad, b_t.grad.numpy(), rtol=5e-5, atol=5e-5)


@given(array=test_utils.array_strategy)
def test_neg_backward(array):
    a_ez = test_utils.make_tensor(array)
    a_t = torch.tensor(array, requires_grad=True)
    r_ez = -a_ez
    r_t = -a_t

    grad_output = test_utils.random_grad(r_ez._array.shape)
    r_ez.backward(grad_output)
    r_t.backward(torch.tensor(grad_output))
    np.testing.assert_allclose(a_ez.grad, a_t.grad.numpy(), rtol=5e-5, atol=5e-5)


@given(arrays=test_utils.array_pair_broadcast_compat_strategy)
def test_truediv_backward(arrays):
    arr1 = arrays[0]
    arr2 = np.where(arrays[1] == 0, 1, arrays[1])
    a_ez = test_utils.make_tensor(arr1)
    b_ez = test_utils.make_tensor(arr2)
    a_t = torch.tensor(arr1, requires_grad=True)
    b_t = torch.tensor(arr2, requires_grad=True)
    r_ez = a_ez / b_ez
    r_t = a_t / b_t

    grad_output = test_utils.random_grad(r_ez._array.shape)
    r_ez.backward(grad_output)
    r_t.backward(torch.tensor(grad_output))
    np.testing.assert_allclose(a_ez.grad, a_t.grad.numpy(), rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(b_ez.grad, b_t.grad.numpy(), rtol=5e-5, atol=5e-5)


@given(array=test_utils.array_strategy, exponent=st.integers(0, 5))
def test_pow_backward(array, exponent):
    a_ez = test_utils.make_tensor(array)
    a_t = torch.tensor(array, requires_grad=True)
    r_ez = a_ez ** exponent
    r_t = a_t ** exponent

    grad_output = test_utils.random_grad(r_ez._array.shape)
    r_ez.backward(grad_output)
    r_t.backward(torch.tensor(grad_output))
    np.testing.assert_allclose(a_ez.grad, a_t.grad.numpy(), rtol=5e-5, atol=5e-5)


@given(arrays=test_utils.array_pair_matmul_compat_strategy)
def test_matmul_backward(arrays):
    a_ez, b_ez = map(test_utils.make_tensor, arrays)
    a_t = torch.tensor(arrays[0], requires_grad=True)
    b_t = torch.tensor(arrays[1], requires_grad=True)
    r_ez = a_ez @ b_ez
    r_t = a_t @ b_t

    grad_output = test_utils.random_grad(r_ez._array.shape)
    r_ez.backward(grad_output)
    r_t.backward(torch.tensor(grad_output))
    np.testing.assert_allclose(a_ez.grad, a_t.grad.numpy(), rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(b_ez.grad, b_t.grad.numpy(), rtol=5e-5, atol=5e-5)


@given(
    array=test_utils.array_strategy,
    keepdims=st.booleans(),
    data=st.data(),
)
def test_mean_backward(array, keepdims, data):
    dim_arg = test_utils.random_axes(array, data)
    a_ez = test_utils.make_tensor(array)
    a_t = torch.tensor(array, requires_grad=True)

    r_ez = a_ez.mean(dim=dim_arg, keepdim=keepdims)
    axes = dim_arg
    if axes is None:
        axes = tuple(range(a_t.ndim))
    r_t = a_t.mean(dim=axes, keepdim=keepdims)

    grad_output = test_utils.random_grad(r_ez._array.shape)
    r_ez.backward(grad_output)
    r_t.backward(torch.tensor(grad_output))
    np.testing.assert_allclose(a_ez.grad, a_t.grad.numpy(), rtol=5e-5, atol=5e-5)


@given(
    array=test_utils.array_strategy,
    keepdims=st.booleans(),
    data=st.data(),
)
def test_sum_backward(array, keepdims, data):
    dim_arg = test_utils.random_axes(array, data)
    a_ez = test_utils.make_tensor(array)
    a_t = torch.tensor(array, requires_grad=True)

    r_ez = a_ez.sum(dim=dim_arg, keepdims=keepdims)
    axes = dim_arg
    if axes is None:
        axes = tuple(range(a_t.ndim))
    r_t = a_t.sum(dim=axes, keepdim=keepdims)

    grad_output = test_utils.random_grad(r_ez._array.shape)
    r_ez.backward(grad_output)
    r_t.backward(torch.tensor(grad_output))
    np.testing.assert_allclose(a_ez.grad, a_t.grad.numpy(), rtol=5e-5, atol=5e-5)
