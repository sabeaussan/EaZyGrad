import numpy as np
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as hnp
import torch
import test_utils
from eazygrad.optimizer import SGD, Adam, AdamW


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


def _run_steps_eazygrad(params, grads_per_step, optimizer_cls, **opt_kwargs):
    ez_params = [test_utils.make_tensor(p, requires_grad=True) for p in params]
    optimizer = optimizer_cls(ez_params, **opt_kwargs)
    for grads in grads_per_step:
        for p, g in zip(ez_params, grads):
            p.grad = None if g is None else g.copy()
        optimizer.step()
    return [p.numpy() for p in ez_params]


def _run_steps_torch(params, grads_per_step, optimizer_cls, **opt_kwargs):
    t_params = [torch.tensor(p, requires_grad=True) for p in params]
    optimizer = optimizer_cls(t_params, **opt_kwargs)
    for grads in grads_per_step:
        for p, g in zip(t_params, grads):
            p.grad = None if g is None else torch.tensor(g)
        optimizer.step()
    return [p.detach().numpy().copy() for p in t_params]


@settings(deadline=None)
@given(
    data=st.data(),
    lr=st.floats(min_value=1e-4, max_value=5e-1, allow_nan=False, allow_infinity=False),
    momentum=st.floats(min_value=0.1, max_value=0.95, allow_nan=False, allow_infinity=False),
    dampening=st.floats(min_value=0.0, max_value=0.8, allow_nan=False, allow_infinity=False),
)
def test_sgd_momentum_matches_torch_multi_step(data, lr, momentum, dampening):
    params, grads = _params_and_grads(data)

    # Reuse the same gradient for a few steps to stress optimizer internal state.
    grads_per_step = [grads, grads, grads]

    ez_final = _run_steps_eazygrad(
        params, grads_per_step, SGD, lr=lr, momentum=momentum, dampening=dampening
    )
    torch_final = _run_steps_torch(
        params, grads_per_step, torch.optim.SGD, lr=lr, momentum=momentum, dampening=dampening
    )

    for ez_p, t_p in zip(ez_final, torch_final):
        np.testing.assert_allclose(ez_p, t_p, rtol=1e-5, atol=1e-5)


@settings(deadline=None)
@given(
    data=st.data(),
    lr=st.floats(min_value=1e-5, max_value=5e-2, allow_nan=False, allow_infinity=False),
)
def test_adam_matches_torch_multi_step(data, lr):
    params, grads = _params_and_grads(data)
    # Three steps with varying gradients to exercise running stats + bias correction.
    grads_step2 = [g * np.float32(0.3) for g in grads]
    grads_step3 = [g * np.float32(-0.2) for g in grads]
    grads_per_step = [grads, grads_step2, grads_step3]
    betas = (0.9, 0.99)
    eps = 1e-8
    print("lr : ",lr)
    print("params : ",params)
    print("grads : ",grads)

    ez_final = _run_steps_eazygrad(
        params, grads_per_step, Adam, lr=lr, betas=betas, eps=eps
    )
    torch_final = _run_steps_torch(
        params, grads_per_step, torch.optim.Adam, lr=lr, betas=betas, eps=eps
    )
    print("ez_final : ",ez_final)
    print("torch_final : ",torch_final)
    print("="*50)

    for ez_p, t_p in zip(ez_final, torch_final):
        np.testing.assert_allclose(ez_p, t_p, rtol=5e-5, atol=5e-5)


@settings(deadline=None)
@given(
    data=st.data(),
    lr=st.floats(min_value=1e-5, max_value=5e-2, allow_nan=False, allow_infinity=False),
    weight_decay=st.floats(min_value=0.0, max_value=1e-1, allow_nan=False, allow_infinity=False),
)
def test_adamw_matches_torch_multi_step(data, lr, weight_decay):
    params, grads = _params_and_grads(data)
    grads_step2 = [g * np.float32(0.5) for g in grads]
    grads_step3 = [g * np.float32(-0.1) for g in grads]
    grads_per_step = [grads, grads_step2, grads_step3]
    betas = (0.9, 0.99)
    eps = 1e-8

    ez_final = _run_steps_eazygrad(
        params,
        grads_per_step,
        AdamW,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    torch_final = _run_steps_torch(
        params,
        grads_per_step,
        torch.optim.AdamW,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    for ez_p, t_p in zip(ez_final, torch_final):
        np.testing.assert_allclose(ez_p, t_p, rtol=5e-5, atol=5e-5)


@given(data=st.data())
def test_adam_and_adamw_skip_none_grads(data):
    params, grads = _params_and_grads(data)
    params = [p.copy() for p in params]
    grads_with_none = []
    for i, g in enumerate(grads):
        grads_with_none.append(None if i % 2 == 0 else g)
    grads_per_step = [grads_with_none]

    ez_adam = _run_steps_eazygrad(params, grads_per_step, Adam, lr=1e-3, betas=(0.9, 0.99), eps=1e-8)
    torch_adam = _run_steps_torch(
        params, grads_per_step, torch.optim.Adam, lr=1e-3, betas=(0.9, 0.99), eps=1e-8
    )
    ez_adamw = _run_steps_eazygrad(
        params, grads_per_step, AdamW, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01
    )
    torch_adamw = _run_steps_torch(
        params,
        grads_per_step,
        torch.optim.AdamW,
        lr=1e-3,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01,
    )

    for ez_p, t_p in zip(ez_adam, torch_adam):
        np.testing.assert_allclose(ez_p, t_p, rtol=1e-6, atol=1e-6)
    for ez_p, t_p in zip(ez_adamw, torch_adamw):
        np.testing.assert_allclose(ez_p, t_p, rtol=1e-6, atol=1e-6)
