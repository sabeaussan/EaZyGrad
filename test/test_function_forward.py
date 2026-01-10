import numpy as np
import pytest
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp
import eazygrad
import torch
import random
import test_utils


# TODO : check type promotion
# TODO : consolidate existing code, refactor, function name, numerical precision, useless copies etc...

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
    a = np.random.randn(3, 3).astype(np.float32)
    t = test_utils.make_tensor(a)
    with pytest.raises(ValueError):
        eazygrad.logsumexp(t, dim=invalid_dim)


@pytest.mark.parametrize("dim", [-100, 100, 999])
def test_logsumexp_dim_out_of_range(dim):
    a = np.random.randn(3, 3).astype(np.float32)
    t = test_utils.make_tensor(a)
    with pytest.raises(ValueError):
        eazygrad.logsumexp(t, dim=dim)


@given(array=test_utils.array_or_scalar_strategy)
def test_logsumexp_forward(array):
    # TODO : Work with scalar ?
    ez = test_utils.make_tensor(array, requires_grad=False)
    t = torch.tensor(array)
    if t.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, t.ndim - 1)
    keepdims = random.random() > 0.5
    result = eazygrad.logsumexp(ez, dim=dim, keepdims=keepdims)
    expected = torch.logsumexp(t, dim=dim, keepdims=keepdims)
    np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)

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
    a = np.random.randn(3, 3).astype(np.float32)
    t = test_utils.make_tensor(a)
    with pytest.raises((TypeError, ValueError)):
        eazygrad.softmax(t, dim=invalid_dim)


@pytest.mark.parametrize("dim", [-100, 100, 999])
def test_softmax_dim_out_of_range(dim):
    a = np.random.randn(4, 4).astype(np.float32)
    t = test_utils.make_tensor(a)
    with pytest.raises(ValueError):
        eazygrad.softmax(t, dim=dim)


@given(array=test_utils.array_or_scalar_strategy)
def test_softmax_forward(array):
    ez = test_utils.make_tensor(array, requires_grad=False)
    t = torch.tensor(array)

    if t.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, t.ndim - 1)

    result = eazygrad.softmax(ez, dim=dim).numpy()
    expected = torch.softmax(t, dim=dim).numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(array=test_utils.array_or_scalar_strategy)
def test_softmax_normalization(array):
    ez = test_utils.make_tensor(array, requires_grad=False)

    if ez.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, ez.ndim - 1)

    result = eazygrad.softmax(ez, dim=dim).numpy()

    assert np.allclose(
        result.sum(axis=dim),
        np.ones_like(result.sum(axis=dim)),
        atol=1e-5,
    )


@given(array=test_utils.array_or_scalar_strategy)
def test_softmax_non_negative(array):
    ez = test_utils.make_tensor(array)

    if ez.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, ez.ndim - 1)

    result = eazygrad.softmax(ez, dim=dim).numpy()
    assert np.all(result >= 0)


@given(array=test_utils.array_or_scalar_strategy)
def test_softmax_preserves_shape(array):
    ez = test_utils.make_tensor(array)

    if ez.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, ez.ndim - 1)

    result = eazygrad.softmax(ez, dim=dim).numpy()
    assert result.shape == ez.shape


# ------------------ Log softmax ------------------

@given(array=test_utils.array_or_scalar_strategy)
def test_log_softmax_forward(array):
    ez = test_utils.make_tensor(array, requires_grad=False)
    t = torch.tensor(array)

    if t.ndim == 0:
        dim = 0
    else:
        dim = random.randint(0, t.ndim - 1)

    result = eazygrad.log_softmax(ez, dim=dim).numpy()
    expected = torch.log_softmax(t, dim=dim).numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


# ============================================
#                  EXP
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_exp_forward(array):
    t = test_utils.make_tensor(array)
    result = eazygrad.exp(t).numpy()
    expected = torch.exp(torch.tensor(array)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


# ============================================
#                  LOG
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_log_forward(array):
    # ensure positivity to avoid nan
    array = np.abs(array) + 1e-3
    t = test_utils.make_tensor(array)
    result = eazygrad.log(t).numpy()
    expected = torch.log(torch.tensor(array)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)



# ============================================
#                  COS
# ============================================


@given(array=test_utils.array_or_scalar_strategy)
def test_cos_forward(array):
    t = test_utils.make_tensor(array)
    result = eazygrad.cos(t).numpy()
    expected = torch.cos(torch.tensor(array)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


# ============================================
#                  SIN
# ============================================
@given(array=test_utils.array_or_scalar_strategy)
def test_sin_forward(array):
    t = test_utils.make_tensor(array)
    result = eazygrad.sin(t).numpy()
    expected = torch.sin(torch.tensor(array)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

# ============================================
#                  RELU
# ============================================
@given(array=test_utils.array_or_scalar_strategy)
def test_relu_forward(array):
    t = test_utils.make_tensor(array)
    result = eazygrad.relu(t).numpy()
    expected = torch.relu(torch.tensor(array)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


# ============================================
#                  SIGMOID
# ============================================
@given(array=test_utils.array_or_scalar_strategy)
def test_sigmoid_forward(array):
    t = test_utils.make_tensor(array)
    result = eazygrad.sigmoid(t).numpy()
    expected = torch.sigmoid(torch.tensor(array)).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

# ============================================
#                  MSE LOSS
# ============================================
@given(arrays=test_utils.array_pair_same_compat_strategy)
def test_mse_loss(arrays):
    predicted, target = map(test_utils.make_tensor, arrays)
    result = eazygrad.mse_loss(predicted, target).numpy()
    expected = torch.nn.functional.mse_loss(
        torch.tensor(arrays[0]), torch.tensor(arrays[1])
    ).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

# ============================================
#                  BCE LOSS
# ============================================
@given(arrays=test_utils.array_pair_same_compat_strategy)
def test_bce_loss_with_logits(arrays):
    logits, target = map(test_utils.make_tensor, arrays)
    result = eazygrad.bce_with_logits_loss(logits, target).numpy()
    
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(arrays[0]), torch.tensor(arrays[1])
    ).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

# @given(arrays=test_utils.array_pair_same_compat_strategy)
# def test_bce_loss(arrays):
#     predicted = test_utils.make_tensor(np.abs(arrays[0]))
#     target = test_utils.make_tensor(np.abs(arrays[1]))
#     result = eazygrad.bce_loss(predicted, target).numpy()
    
#     expected = torch.nn.functional.binary_cross_entropy(
#         torch.tensor(np.abs(arrays[0])), torch.tensor(np.abs(arrays[1]))
#     ).numpy()
#     np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

# ============================================
#             CROSS ENTROPY LOSS
# ============================================


@given(data=st.data())
def test_cross_entropy_loss_forward_class_indices(data):
    shape = data.draw(hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5))
    batch_size, num_classes = shape
    logits_array = data.draw(
        hnp.arrays(dtype=np.float32, shape=shape, elements=test_utils.float_strategy)
    )
    target_array = data.draw(
        hnp.arrays(
            dtype=np.int64,
            shape=(batch_size,),
            elements=st.integers(min_value=0, max_value=num_classes - 1),
        )
    )

    logits = test_utils.make_tensor(logits_array, requires_grad=False)
    target = eazygrad.tensor(target_array, requires_grad=False, dtype=np.int64)
    result = eazygrad.cross_entropy_loss(logits, target).numpy()

    expected = torch.nn.functional.cross_entropy(
        torch.tensor(logits_array), torch.tensor(target_array)
    ).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(data=st.data())
def test_cross_entropy_loss_forward_probs(data):
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

    logits = test_utils.make_tensor(logits_array, requires_grad=False)
    target = test_utils.make_tensor(target_probs, requires_grad=False)
    result = eazygrad.cross_entropy_loss(logits, target).numpy()

    expected = torch.nn.functional.cross_entropy(
        torch.tensor(logits_array), torch.tensor(target_probs)
    ).numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
