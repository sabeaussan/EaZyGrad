import numpy as np
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp
import torch
import test_utils
from eazygrad.nn.linear import Linear


def _linear_case(data, bias):
    batch = data.draw(st.integers(min_value=1, max_value=5))
    n_in = data.draw(st.integers(min_value=1, max_value=5))
    n_out = data.draw(st.integers(min_value=1, max_value=5))
    x_array = data.draw(
        hnp.arrays(dtype=np.float32, shape=(batch, n_in), elements=test_utils.float_strategy)
    )
    w_array = data.draw(
        hnp.arrays(dtype=np.float32, shape=(n_in, n_out), elements=test_utils.float_strategy)
    )
    b_array = None
    if bias:
        b_array = data.draw(
            hnp.arrays(dtype=np.float32, shape=(1, n_out), elements=test_utils.float_strategy)
        )
    return x_array, w_array, b_array


@given(data=st.data())
def test_linear_forward_with_bias(data):
    x_array, w_array, b_array = _linear_case(data, bias=True)
    x = test_utils.make_tensor(x_array, requires_grad=False)
    linear = Linear(x_array.shape[1], w_array.shape[1], bias=True, requires_grad=True)
    linear.parameters[0] = test_utils.make_tensor(w_array, requires_grad=True)
    linear.parameters[1] = test_utils.make_tensor(b_array, requires_grad=True)
    result = linear(x).numpy()

    t_linear = torch.nn.Linear(x_array.shape[1], w_array.shape[1], bias=True)
    with torch.no_grad():
        t_linear.weight.copy_(torch.tensor(w_array).T)
        t_linear.bias.copy_(torch.tensor(b_array).reshape(-1))
    expected = t_linear(torch.tensor(x_array)).detach().numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(data=st.data())
def test_linear_forward_no_bias(data):
    x_array, w_array, _ = _linear_case(data, bias=False)
    x = test_utils.make_tensor(x_array, requires_grad=False)
    linear = Linear(x_array.shape[1], w_array.shape[1], bias=False, requires_grad=True)
    linear.parameters[0] = test_utils.make_tensor(w_array, requires_grad=True)
    result = linear(x).numpy()

    t_linear = torch.nn.Linear(x_array.shape[1], w_array.shape[1], bias=False)
    with torch.no_grad():
        t_linear.weight.copy_(torch.tensor(w_array).T)
    expected = t_linear(torch.tensor(x_array)).detach().numpy()
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(data=st.data())
def test_linear_backward_with_bias(data):
    x_array, w_array, b_array = _linear_case(data, bias=True)
    x = test_utils.make_tensor(x_array, requires_grad=True)
    linear = Linear(x_array.shape[1], w_array.shape[1], bias=True, requires_grad=True)
    linear.parameters[0] = test_utils.make_tensor(w_array, requires_grad=True)
    linear.parameters[1] = test_utils.make_tensor(b_array, requires_grad=True)
    y = linear(x)

    t_x = torch.tensor(x_array, requires_grad=True)
    t_linear = torch.nn.Linear(x_array.shape[1], w_array.shape[1], bias=True)
    with torch.no_grad():
        t_linear.weight.copy_(torch.tensor(w_array).T)
        t_linear.bias.copy_(torch.tensor(b_array).reshape(-1))
    t_y = t_linear(t_x)

    grad_output = test_utils.random_grad(y._array.shape)
    y.backward(grad_output)
    t_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(x.grad, t_x.grad.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        linear.parameters[0].grad, t_linear.weight.grad.numpy().T, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        linear.parameters[1].grad, t_linear.bias.grad.numpy().reshape(1, -1), rtol=1e-5, atol=1e-5
    )


@given(data=st.data())
def test_linear_backward_no_bias(data):
    x_array, w_array, _ = _linear_case(data, bias=False)
    x = test_utils.make_tensor(x_array, requires_grad=True)
    linear = Linear(x_array.shape[1], w_array.shape[1], bias=False, requires_grad=True)
    linear.parameters[0] = test_utils.make_tensor(w_array, requires_grad=True)
    y = linear(x)

    t_x = torch.tensor(x_array, requires_grad=True)
    t_linear = torch.nn.Linear(x_array.shape[1], w_array.shape[1], bias=False)
    with torch.no_grad():
        t_linear.weight.copy_(torch.tensor(w_array).T)
    t_y = t_linear(t_x)

    grad_output = test_utils.random_grad(y._array.shape)
    y.backward(grad_output)
    t_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(x.grad, t_x.grad.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        linear.parameters[0].grad, t_linear.weight.grad.numpy().T, rtol=1e-5, atol=1e-5
    )
