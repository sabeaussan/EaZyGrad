import numpy as np
from hypothesis import given,settings, strategies as st
import hypothesis.extra.numpy as hnp
import torch
import test_utils
from eazygrad.optimizer import SGD


def _params_and_grads(data):
    count = data.draw(st.integers(min_value=1, max_value=4))
    params = []
    grads = []
    for _ in range(count):
        shape = data.draw(hnp.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=5))
        param = data.draw(
            hnp.arrays(dtype=np.float32, shape=shape, elements=test_utils.float_strategy)
        )
        grad = data.draw(
            hnp.arrays(dtype=np.float32, shape=shape, elements=test_utils.float_strategy)
        )
        params.append(param)
        grads.append(grad)
    return params, grads

@settings(deadline=None)
@given(data=st.data(), lr=st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_sgd_step_matches_torch(data, lr):
    params, grads = _params_and_grads(data)
    ez_params = [test_utils.make_tensor(p, requires_grad=True) for p in params]
    for p, g in zip(ez_params, grads):
        p.grad = g

    t_params = [torch.tensor(p, requires_grad=True) for p in params]
    for p, g in zip(t_params, grads):
        p.grad = torch.tensor(g)

    ez_opt = SGD(ez_params, lr=lr)
    t_opt = torch.optim.SGD(t_params, lr=lr)

    ez_opt.step()
    t_opt.step()

    for ez_p, t_p in zip(ez_params, t_params):
        np.testing.assert_allclose(ez_p._array, t_p.detach().numpy(), rtol=1e-5, atol=1e-5)


@given(data=st.data())
def test_sgd_zero_grad(data):
    params, grads = _params_and_grads(data)
    ez_params = [test_utils.make_tensor(p, requires_grad=True) for p in params]
    for p, g in zip(ez_params, grads):
        p.grad = g

    opt = SGD(ez_params, lr=0.1)
    opt.zero_grad()

    for p in ez_params:
        assert p.grad == None
