import numpy as np
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as hnp
import eazygrad
import torch
import random
import test_utils


# ------------------ Log sum exp ------------------
@settings(deadline=1000)
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
    r_ez = eazygrad.logsumexp(ez, dim=dim, keepdims=keepdims)
    r_t = torch.logsumexp(t, dim=dim, keepdims=keepdims)
    grad_output = test_utils._random_grad(r_ez._array.shape)
    r_ez.backward(grad_output)
    r_t.backward(torch.tensor(grad_output))
    np.testing.assert_allclose(ez.grad, t.grad.numpy(), rtol=5e-5, atol=5e-5)


# ------------------ Softmax ------------------

@given(array=test_utils.array_or_scalar_strategy)
def test_softmax_backward(array):
    x = test_utils.make_tensor(array, requires_grad=True)

    if x.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, x.ndim - 1)

    y = eazygrad.softmax(x, dim=dim)

    t = torch.tensor(array, requires_grad=True)
    y_torch = torch.softmax(t, dim=dim)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    y_torch.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ------------------ Log softmax ------------------

@given(array=test_utils.array_or_scalar_strategy)
def test_log_softmax_backward_autograd(array):
    x = test_utils.make_tensor(array, requires_grad=True)

    if x.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, x.ndim - 1)

    y = eazygrad.log_softmax(x, dim=dim)

    t = torch.tensor(array, requires_grad=True)
    y_torch = torch.log_softmax(t, dim=dim)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    y_torch.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#                  EXP
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_exp_backward_autograd(array):
    x = test_utils.make_tensor(array, requires_grad=True)
    y = eazygrad.exp(x)

    t = torch.tensor(array, requires_grad=True)
    y_torch = torch.exp(t)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    y_torch.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#                  LOG
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_log_backward_autograd(array):
    array = np.abs(array) + 1e-3
    x = test_utils.make_tensor(array, requires_grad=True)
    y = eazygrad.log(x)

    t = torch.tensor(array, requires_grad=True)
    torch_y = torch.log(t)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    torch_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#                  COS
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_cos_backward_autograd(array):
    x = test_utils.make_tensor(array, requires_grad=True)
    y = eazygrad.cos(x)

    t = torch.tensor(array, requires_grad=True)
    torch_y = torch.cos(t)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    torch_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#                  SIN
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_sin_backward_autograd(array):
    x = test_utils.make_tensor(array, requires_grad=True)
    y = eazygrad.sin(x)

    t = torch.tensor(array, requires_grad=True)
    torch_y = torch.sin(t)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    torch_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#                  RELU
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_relu_backward_autograd(array):
    x = test_utils.make_tensor(array, requires_grad=True)
    y = eazygrad.relu(x)

    t = torch.tensor(array, requires_grad=True)
    torch_y = torch.relu(t)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    torch_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#                  SIGMOID
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_sigmoid_backward_autograd(array):
    x = test_utils.make_tensor(array, requires_grad=True)
    y = eazygrad.sigmoid(x)

    t = torch.tensor(array, requires_grad=True)
    torch_y = torch.sigmoid(t)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    torch_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        x.grad, t.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#           BCE WITH LOGITS LOSS
# ============================================


@given(arrays=test_utils.array_pair_same_compat_strategy)
def test_bce_with_logits_backward_autograd(arrays):
    logits_array, target_array = arrays
    logits = test_utils.make_tensor(logits_array, requires_grad=True)
    target = test_utils.make_tensor(target_array, requires_grad=False)
    y = eazygrad.bce_with_logits_loss(logits, target)

    t_logits = torch.tensor(logits_array, requires_grad=True)
    t_target = torch.tensor(target_array)
    y_torch = torch.nn.functional.binary_cross_entropy_with_logits(t_logits, t_target)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    y_torch.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        logits.grad, t_logits.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#             CROSS ENTROPY LOSS
# ============================================


@given(data=st.data())
def test_cross_entropy_loss_backward_class_indices(data):
    logits_array, target_array = test_utils.logits_and_class_targets(data)

    logits = test_utils.make_tensor(logits_array, requires_grad=True)
    target = eazygrad.tensor(target_array, requires_grad=False, dtype=np.int64)
    y = eazygrad.cross_entropy_loss(logits, target)

    t_logits = torch.tensor(logits_array, requires_grad=True)
    t_target = torch.tensor(target_array)
    y_torch = torch.nn.functional.cross_entropy(t_logits, t_target)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    y_torch.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        logits.grad, t_logits.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


@given(data=st.data())
def test_cross_entropy_loss_backward_probs(data):
    shape = data.draw(hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5))
    logits_array = data.draw(
        hnp.arrays(dtype=np.float32, shape=shape, elements=test_utils.float_strategy)
    )
    target_raw = data.draw(
        hnp.arrays(
            dtype=np.float32,
            shape=shape,
            elements=st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32,
            ),
        )
    )
    target_probs = target_raw + np.float32(1e-6)
    target_probs = target_probs / target_probs.sum(axis=1, keepdims=True)

    logits = test_utils.make_tensor(logits_array, requires_grad=True)
    target = test_utils.make_tensor(target_probs, requires_grad=False)
    y = eazygrad.cross_entropy_loss(logits, target)

    t_logits = torch.tensor(logits_array, requires_grad=True)
    t_target = torch.tensor(target_probs)
    y_torch = torch.nn.functional.cross_entropy(t_logits, t_target)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    y_torch.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        logits.grad, t_logits.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )


# ============================================
#                 NLL LOSS
# ============================================


@given(data=st.data())
def test_nll_loss_backward(data):
    logits_array, target_array = test_utils.logits_and_class_targets(data)

    log_probs = torch.log_softmax(torch.tensor(logits_array), dim=1).numpy()
    predicted = test_utils.make_tensor(log_probs, requires_grad=True)
    target = eazygrad.tensor(target_array, requires_grad=False, dtype=np.int64)
    y = eazygrad.nll_loss(predicted, target)

    t_predicted = torch.tensor(log_probs, requires_grad=True)
    t_target = torch.tensor(target_array)
    y_torch = torch.nn.functional.nll_loss(t_predicted, t_target)

    grad_output = test_utils._random_grad(y._array.shape)
    y.backward(grad_output)
    y_torch.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(
        predicted.grad, t_predicted.grad.numpy(),
        atol=1e-5, rtol=1e-5,
    )
