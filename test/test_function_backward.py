import numpy as np
from hypothesis import given
import pytensor
import torch
import random
import test_utils

# ------------------ Log sum exp ------------------

@given(array=test_utils.array_or_scalar_strategy)
def test_logsumexp_backward(array):
    # TODO : rajouter le test de keep dims
    ez = test_utils.make_tensor(array, requires_grad=True)
    t = torch.tensor(array, requires_grad=True)
    if t.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, t.ndim - 1)
    keepdims = random.random() > 0.5
    expected = torch.logsumexp(t, dim=dim, keepdims=keepdims).sum()
    result = pytensor.logsumexp(ez, dim=dim, keepdims=keepdims).sum()
    expected.backward()
    result.backward()
    np.testing.assert_allclose(ez.grad, t.grad.numpy(), rtol=5e-5, atol=5e-5)


# ------------------ Softmax ------------------

@given(array=test_utils.array_or_scalar_strategy)
def test_softmax_backward(array):
    x = test_utils.make_tensor(array, requires_grad=True)

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
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#                  EXP
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_exp_backward(array):
    x = test_utils.make_tensor(array, requires_grad=True)
    y = pytensor.exp(x)
    y_sum = y.sum()
    y_sum.backward()

    t = torch.tensor(array, requires_grad=True)
    y_torch = torch.exp(t).sum()
    y_torch.backward()

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#                  LOG
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_log_backward(array):
    array = np.abs(array) + 1e-3
    x = test_utils.make_tensor(array, requires_grad=True)
    y = pytensor.log(x)
    y_sum = y.sum()
    y_sum.backward()

    t = torch.tensor(array, requires_grad=True)
    torch_y = torch.log(t).sum()
    torch_y.backward()

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#                  RELU
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_relu_backward(array):
    x = test_utils.make_tensor(array, requires_grad=True)
    y = pytensor.relu(x).sum()
    y.backward()

    t = torch.tensor(array, requires_grad=True)
    torch_y = torch.relu(t).sum()
    torch_y.backward()

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#                  SIGMOID
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_sigmoid_backward(array):
    x = test_utils.make_tensor(array, requires_grad=True)
    y = pytensor.sigmoid(x).sum()
    y.backward()

    t = torch.tensor(array, requires_grad=True)
    torch_y = torch.sigmoid(t).sum()
    torch_y.backward()

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )
