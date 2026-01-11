import numpy as np
import pytest
import eazygrad
from eazygrad.grad import dag


def _reset_dag_state(prev_state):
    dag.grad_enable = prev_state
    dag.clear()


def test_no_grad_disables_tracking_for_new_tensors():
    prev_state = dag.grad_enable
    dag.grad_enable = True
    dag.clear()
    try:
        node_count_before = dag.node_count
        with eazygrad.no_grad_ctx():
            t = eazygrad.tensor([1.0, 2.0], requires_grad=True)
            assert t.requires_grad is False
            assert t.node_id is None
            assert dag.node_count == node_count_before
        t2 = eazygrad.tensor([3.0], requires_grad=True)
        assert t2.requires_grad is True
        assert t2.node_id is not None
    finally:
        _reset_dag_state(prev_state)


def test_no_grad_disables_tracking_for_operations():
    prev_state = dag.grad_enable
    dag.grad_enable = True
    dag.clear()
    try:
        x = eazygrad.tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        y = eazygrad.tensor(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)
        node_count_before = dag.node_count
        with eazygrad.no_grad_ctx():
            z = x + y
            assert z.requires_grad is False
            assert z.node_id is None
            assert dag.node_count == node_count_before
        assert x.requires_grad is True
        assert y.requires_grad is True
    finally:
        _reset_dag_state(prev_state)


def test_no_grad_restores_state_on_exception():
    prev_state = dag.grad_enable
    dag.grad_enable = True
    dag.clear()
    try:
        with pytest.raises(RuntimeError):
            with eazygrad.no_grad_ctx():
                assert dag.grad_enable is False
                raise RuntimeError("boom")
        assert dag.grad_enable is True
    finally:
        _reset_dag_state(prev_state)


def test_no_grad_decorator_disables_tracking_for_new_tensors():
    prev_state = dag.grad_enable
    dag.grad_enable = True
    dag.clear()
    try:
        node_count_before = dag.node_count

        @eazygrad.no_grad
        def build_tensor():
            return eazygrad.tensor([1.0, 2.0], requires_grad=True)

        t = build_tensor()
        assert t.requires_grad is False
        assert t.node_id is None
        assert dag.node_count == node_count_before
    finally:
        _reset_dag_state(prev_state)


def test_no_grad_decorator_disables_tracking_for_operations():
    prev_state = dag.grad_enable
    dag.grad_enable = True
    dag.clear()
    try:
        x = eazygrad.tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
        y = eazygrad.tensor(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)
        node_count_before = dag.node_count

        @eazygrad.no_grad
        def add_tensors(a, b):
            return a + b

        z = add_tensors(x, y)
        assert z.requires_grad is False
        assert z.node_id is None
        assert dag.node_count == node_count_before
        assert x.requires_grad is True
        assert y.requires_grad is True
    finally:
        _reset_dag_state(prev_state)


def test_no_grad_decorator_restores_state_on_exception():
    prev_state = dag.grad_enable
    dag.grad_enable = True
    dag.clear()
    try:
        @eazygrad.no_grad
        def fail():
            assert dag.grad_enable is False
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            fail()
        assert dag.grad_enable is True
    finally:
        _reset_dag_state(prev_state)
