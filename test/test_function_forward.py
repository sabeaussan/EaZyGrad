import numpy as np
import pytest
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp
import pytensor
import torch
import random
from pytensor import operations, dag

# Add a check for softmax/log_sum_exp dim arg
# Define scaffold for tests (codex ?)

# TODO : check type promotion
# TODO : consolidate existing code, refactor, function name, numerical precision, useless copies etc...

# -----------------------------
# Property-based tests
# -----------------------------

float_strategy = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, allow_subnormal=False, width=32)

# N-D arrays
array_strategy = hnp.arrays(
    dtype=np.float32,
    shape=hnp.array_shapes(min_dims=0, max_dims=4, min_side=1, max_side=5),
    elements=float_strategy,
)

tensor_module = pytensor.tensor_module
_Tensor = tensor_module._Tensor

def make_tensor(arr, requires_grad=True):
    return pytensor.tensor(arr, requires_grad)

# ------------------ Log sum exp ------------------

# Test invalid arguments for log_sum_exp
@pytest.mark.parametrize("invalid_dim", [
    ([5]),
    ((3.14, 32)),
    ((3, -32)),
    ((3, 0)),
    ([3.14, 1, 5]),
    (3.14), 
])
def test_logsumexp_dim_arg_invalid(invalid_dim):
    a = np.random.randn(3,3).astype(np.float32)
    t = make_tensor(a)
    with pytest.raises(ValueError):
        pytensor.logsumexp(t, dim=invalid_dim)

@pytest.mark.parametrize("dim", [-100, 100, 999])
def test_logsumexp_dim_out_of_range(dim):
    a = np.random.randn(3,3).astype(np.float32)
    t = make_tensor(a)
    with pytest.raises(ValueError):
        pytensor.logsumexp(t, dim=dim)


@given(array=array_strategy)
def test_logsumexp_forward(array):
    # TODO : Work with scalar ?
    ez = make_tensor(array, requires_grad=False)
    t = torch.tensor(array)
    if len(t.shape)==0:
        dim=0
    else:
        dim = random.randint(0, len(array.shape)-1)
    keepdims = random.random()>0.5
    result = pytensor.logsumexp(ez, dim=dim, keepdims=keepdims)
    expected = torch.logsumexp(t, dim=dim, keepdims=keepdims)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)

@given(array=array_strategy)
def test_logsumexp_backward(array):
    # TODO : rajouter le test de keep dims
    ez = make_tensor(array, requires_grad=True)
    t = torch.tensor(array, requires_grad=True)
    if len(t.shape)==0:
        dim=0
    else:
        dim = random.randint(0, len(array.shape)-1)
    keepdims = random.random()>0.5
    expected = torch.logsumexp(t, dim=dim, keepdims=keepdims).sum()
    result = pytensor.logsumexp(ez, dim=dim, keepdims=keepdims).sum()
    expected.backward()
    result.backward()
    np.testing.assert_allclose(ez.grad, t.grad.numpy(), rtol=5e-5, atol=5e-5)

# ------------------ Softmax ------------------

@pytest.mark.parametrize("invalid_dim", [
    ([5]),
    ((3.14, 32)),
    ("hello"),
    (None),
    ([3.14, 1, 5]),
    (3.14),
])
def test_softmax_dim_arg_invalid(invalid_dim):
    a = np.random.randn(3,3).astype(np.float32)
    t = make_tensor(a)
    with pytest.raises((TypeError, ValueError)):
        pytensor.softmax(t, dim=invalid_dim)


@pytest.mark.parametrize("dim", [-100, 100, 999])
def test_softmax_dim_out_of_range(dim):
    a = np.random.randn(4,4).astype(np.float32)
    t = make_tensor(a)
    with pytest.raises(ValueError):
        pytensor.softmax(t, dim=dim)


@given(array=array_strategy)
def test_softmax_forward(array):
    ez = make_tensor(array, requires_grad=False)
    t = torch.tensor(array)

    if t.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, t.ndim - 1)

    result = pytensor.softmax(ez, dim=dim).numpy()
    expected = torch.softmax(t, dim=dim).numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

@given(array=array_strategy)
def test_softmax_normalization(array):
    ez = make_tensor(array, requires_grad=False)

    if ez.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, ez.ndim - 1)

    result = pytensor.softmax(ez, dim=dim).numpy()

    assert np.allclose(
        result.sum(axis=dim), 
        np.ones_like(result.sum(axis=dim)),
        atol=1e-5
    )

@given(array=array_strategy)
def test_softmax_non_negative(array):
    ez = make_tensor(array)

    if ez.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, ez.ndim - 1)

    result = pytensor.softmax(ez, dim=dim).numpy()
    assert np.all(result >= 0)

@given(array=array_strategy)
def test_softmax_preserves_shape(array):
    ez = make_tensor(array)

    if ez.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, ez.ndim - 1)

    result = pytensor.softmax(ez, dim=dim).numpy()
    assert result.shape == ez.shape


@given(array=array_strategy)
def test_softmax_backward_autograd(array):
    x = make_tensor(array, requires_grad=True)

    if x.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, x.ndim - 1)

    y = pytensor.softmax(x, dim=dim)
    y_sum = y.sum()
    y_sum.backward()

    t = torch.tensor(array, requires_grad=True)
    y_torch = torch.softmax(t, dim=dim).sum()
    y_torch.backward()

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5
    )

# ============================================
#                  EXP
# ============================================

@given(array=array_strategy)
def test_exp_forward(array):
    t = make_tensor(array)
    result = pytensor.exp(t).numpy()
    expected = torch.exp(torch.tensor(array)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)



@given(array=array_strategy)
def test_exp_backward_autograd(array):
    x = make_tensor(array, requires_grad=True)
    y = pytensor.exp(x)
    y_sum = y.sum()
    y_sum.backward()

    t = torch.tensor(array, requires_grad=True)
    y_torch = torch.exp(t).sum()
    y_torch.backward()

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5
    )

# ============================================
#                  LOG
# ============================================


@given(array=array_strategy)
def test_log_forward(array):
    # ensure positivity to avoid nan
    array = np.abs(array) + 1e-3
    t = make_tensor(array)
    result = pytensor.log(t).numpy()
    expected = torch.log(torch.tensor(array)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(array=array_strategy)
def test_log_backward_autograd(array):
    array = np.abs(array) + 1e-3
    x = make_tensor(array, requires_grad=True)
    y = pytensor.log(x)
    y_sum = y.sum()
    y_sum.backward()

    t = torch.tensor(array, requires_grad=True)
    torch_y = torch.log(t).sum()
    torch_y.backward()

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5
    )

# ============================================
#                  RELU
# ============================================


@given(array=array_strategy)
def test_relu_forward(array):
    t = make_tensor(array)
    result = pytensor.relu(t).numpy()
    expected = torch.relu(torch.tensor(array)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(array=array_strategy)
def test_relu_backward_autograd(array):
    x = make_tensor(array, requires_grad=True)
    y = pytensor.relu(x).sum()
    y.backward()

    t = torch.tensor(array, requires_grad=True)
    torch_y = torch.relu(t).sum()
    torch_y.backward()

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5
    )

# ============================================
#                  SIGMOID
# ============================================

@given(array=array_strategy)
def test_sigmoid_forward(array):
    t = make_tensor(array)
    result = pytensor.sigmoid(t).numpy()
    expected = torch.sigmoid(torch.tensor(array)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

@given(array=array_strategy)
def test_sigmoid_backward_autograd(array):
    x = make_tensor(array, requires_grad=True)
    y = pytensor.sigmoid(x).sum()
    y.backward()

    t = torch.tensor(array, requires_grad=True)
    torch_y = torch.sigmoid(t).sum()
    torch_y.backward()

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5
    )

a = torch.arange(10, requires_grad=True, dtype=float).reshape(2,5)
b = a**2
c = b.sum()
#b[0]=-10
a.retain_grad()
c.backward()
print(a.grad)