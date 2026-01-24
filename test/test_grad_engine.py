import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import HealthCheck
import eazygrad
from eazygrad import dag
import torch
import operator
import random
import test_utils


# ============================= Tests for grad engine ============================= #

dag_torch = {}
node_torch_id = 0 

ARRAY_BOUND = 2

# Operation for the computation graph

def get_rng():
    seed = min(len(dag.node_map.keys()), len(dag_torch.keys()))
    # Avoid global state
    return random.Random(seed), seed

def mean_op(tensor):
    rng, seed = get_rng()
    keep_dims = rng.random() > 0.5
    dims = test_utils.generate_dim_from_shape(tensor.shape, rng)
    return tensor.mean(dim=dims, keepdim=keep_dims)

def sum_op(tensor):
    rng, seed = get_rng()
    keep_dims = rng.random() > 0.5
    dims = test_utils.generate_dim_from_shape(tensor.shape, rng)
    return tensor.sum(dim=dims, keepdims=keep_dims)

def safe_divide(tensor1, tensor2):
    arr = tensor2.detach().numpy()
    if np.any(arr<=0.00001):
        # Add a small epsilon to avoid division by close to 0
        print("Safe divide triggered")
        if isinstance(tensor2, torch.Tensor):
            tensor2 = apply_torch_op(operator.add, [tensor2, 0.1])
        else:
            tensor2 += 0.1
    return operator.truediv(tensor1, tensor2)


def reshape_op(tensor):
    if len(tensor.shape)==0:
        # no op
        return tensor.reshape(-1)
    rng, seed = get_rng()
    shape = test_utils.random_factorization(np.prod(tensor.shape), rng)
    if isinstance(shape, int):
        new_t = tensor.reshape(shape)
    else:
        new_t = tensor.reshape(*shape)
    return new_t

def getitem_op(tensor):
    if len(tensor.shape)==0:
        # no op
        return tensor.reshape(-1)
    rng, seed = get_rng()
    dim = rng.randint(0,len(tensor.shape)-1)
    idx = rng.randint(0,tensor.shape[dim]-1)

    # build slice
    slice_ = tuple(
        idx if d == dim else slice(None)
        for d in range(len(tensor.shape))
    )
    return tensor[slice_]


test_ops = [
    operator.add, 
    operator.sub,
    safe_divide, 
    operator.mul,  
    reshape_op,
    sum_op,
    getitem_op,
    mean_op,
]
    
# TODO : test computation graph backwarding
# Generate a random simplegrad computation graph using basic operations (add, sub, mul, matmul)
# Generate a the same pytorch computation graph
# Compute root node with mean, sum or random compatible vector
# Make all intermediate variables retain_grad
# Compare all computed grads
# Edge cases : broadcasted inputs
# Check obsolete nodes
# Check for disconnected computation graph
 

def pad_to_longest_shape(array1_shape, array2_shape):
    diff = len(array1_shape)-len(array2_shape)
    if diff > 0:
        # pad array2_shape
        array2_shape = np.pad(array2_shape, (diff,0), constant_values=1)
    else:
        # pad array1_shape
        array1_shape = np.pad(array1_shape, (-diff,0), constant_values=1)
    return array1_shape, array2_shape

def _sample_array_stable(shape, eps = 0.1):
    # Exclude too small values to avoid numerical instability
    array = np.random.choice([-1, 1], shape) * np.random.uniform(low=eps, high=ARRAY_BOUND, size=shape)
    check = np.sum(array==0)
    if check:
        print(check)
        raise
    return array.astype(np.float32)

def generate_broadcast_compat_array(array_shape):
    # TODO : padd to max len
    shape = np.random.randint(low=1, high=5, size=4)
    array_shape, shape = pad_to_longest_shape(array_shape, shape)
    new_shape = []
    for d, s in zip(shape, array_shape):
        if s!=1 and d!=s:
            d=s
        new_shape.append(d)
    array = _sample_array_stable(new_shape)
    return array

def is_broadcastable(array1_shape, array2_shape):
    array1_shape, array2_shape = pad_to_longest_shape(array1_shape, array2_shape)
    for d1,d2 in zip(array1_shape, array2_shape):
        if d1!=d2 and d1!=1 and d2!=1:
            return False
    return True
    

def find_broadcast_compat_array(array_shape):
    current_node_id = max(dag.node_map.keys())
    valid_candidates = []
    for node_id, node in dag.node_map.items():
        if node_id==current_node_id:
            continue
        shape = node.result.shape
        if is_broadcastable(array_shape, shape):
            valid_candidates.append(node_id)
        
    try:
        id_ = random.choice(valid_candidates)
    except IndexError:
        return None
    return id_

def sample_tensors(input_array_shape, pnew):
    new = np.random.rand() < pnew
    if new:
        other_array = generate_broadcast_compat_array(input_array_shape)
        return test_utils.make_tensor(other_array), make_torch_tensor(other_array)
    else:
        id_ = find_broadcast_compat_array(input_array_shape)
        if id_ is None:
            other_array = generate_broadcast_compat_array(input_array_shape)
            return test_utils.make_tensor(other_array), make_torch_tensor(other_array)
        else:
            return dag.node_map[id_].result, dag_torch[id_]

def register_torch_tensor(tensor):
    # node_torch_id is re-bound (re-assigned as node_torch_id++) so
    # we need to declare it as global else compile error : UnboundLocal
    global node_torch_id
    dag_torch[node_torch_id] = tensor
    node_torch_id += 1

def apply_torch_op(op, operands):
    # apply op
    # apply retain_grad 
    # register new tensor
    out = op(*operands)
    out.retain_grad()
    register_torch_tensor(out)
    return out

def make_torch_tensor(array):
    tensor = torch.tensor(array.copy(), requires_grad=True)
    register_torch_tensor(tensor)
    return tensor


def sample_root(output_ez, output_torch):
    root_ez = output_ez.mean()
    root_torch = output_torch.mean()
    root_torch.retain_grad()
    return root_ez, root_torch

def sample_node(input_ez, input_torch, pnew=0.6):
    # sample a new node for the computation graph
    # If the op requires a 2nd tensor, generate/reuse one
    op=random.choice(test_ops)
    torch_tensors = [input_torch]
    ez_tensors = [input_ez]
    num_args = test_utils.get_args_count(op) 
    if num_args > 1:
        ez_t, torch_t = sample_tensors(input_ez.shape, pnew)
        ez_tensors.append(ez_t)
        torch_tensors.append(torch_t)
    
    out_ez = op(*ez_tensors)
    out_torch = apply_torch_op(op, torch_tensors)
    np.testing.assert_allclose(out_ez.numpy(), out_torch.detach().numpy(), rtol=1e-2, atol=5e-4, err_msg=f"{op}")
    return out_ez, out_torch 

def sample_computation_graph(seed):
    # Define basic properties of the graph
    np.random.seed(seed)
    random.seed(seed)
    num_nodes = np.random.randint(low=10, high=15)
    p_new = np.random.uniform(low=0.2, high=0.3)
    cnt_node = 0

    # Generate a first array
    shape = np.random.randint(low=1, high=5, size=4)
    input_array = _sample_array_stable(shape)
    in_ez, in_torch = test_utils.make_tensor(input_array), make_torch_tensor(input_array)
    out_ez, out_torch = sample_node(input_ez=in_ez, input_torch=in_torch, pnew=p_new)
    while cnt_node < num_nodes:
        out_ez, out_torch = sample_node(input_ez=out_ez, input_torch=out_torch, pnew=p_new)
        cnt_node += 1

    # Compute and return root nodes
    return sample_root(output_ez=out_ez, output_torch=out_torch)

def assert_allclose_with_index(x, y, rtol=1e-2, atol=5e-4):
    x = np.asarray(x)
    y = np.asarray(y)

    abs_diff = np.abs(x - y)
    tol = atol + rtol * np.abs(y)

    fail = abs_diff > tol
    if not np.any(fail):
        return

    i = np.unravel_index(np.argmax(abs_diff - tol), x.shape)

    dag.plot()

    raise AssertionError(
        f"Allclose failed at index {i}\n"
        f"x[{i}] = {x[i]}\n"
        f"y[{i}] = {y[i]}\n"
        f"|x-y| = {abs_diff[i]}\n"
        f"allowed tol = {tol[i]}"
    )

def check_grads():
    rtol, atol = 1e-2, 1e-3
    for i in reversed(dag_torch.keys()):
        torch_grad = dag_torch[i].grad
        ez_grad = dag.node_map[i].result.grad
        
        assert_allclose_with_index(torch_grad.numpy(), ez_grad, rtol=rtol, atol=atol)


def test_computation_graph():
    dag.clear()
    torch.set_num_threads(1)
    global node_torch_id
    global dag_torch
    for seed in range(500):
        print(f"============ {seed} ============")
        rg, rt = sample_computation_graph(seed)
        np.testing.assert_allclose(rg._array, rt.detach().numpy(), rtol=1e-2, atol=5e-4)
        rg.backward(retain_graph=True)
        rt.backward()
        check_grads()
        node_torch_id = 0
        dag_torch = {}
        dag.clear()

# ============================= Tests for no_grad context manager and decorator ============================= #


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


# ============================= Tests for grad broadcast ============================= #

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(arrays=test_utils.array_pair_broadcast_compat_strategy)
def test_check_broadcasted_shape(mock_dag, arrays):
    # dag is mocked to avoid side effects
    a, b = map(test_utils.make_tensor, arrays)
    c = a + b
    grad_a = eazygrad.check.broadcasted_shape(
        grad=c._array,
        tensor=a
	)
    grad_b = eazygrad.check.broadcasted_shape(
        grad=c._array,
        tensor=b
	)
    assert grad_a.shape==a.shape, "Grad shape does not match for tensor a"
    assert grad_b.shape==b.shape, "Grad shape does not match for tensor b"

@pytest.mark.parametrize("vector, array", [ 
    ([], np.random.randn(10)), 
    ((3.14, 32), np.random.randn(10)), 
    (["string", True], np.random.randn(10)),
    ([np.random.randn(10), np.random.randn(5)], np.random.randn(10)), 
    ({"key" : np.arange(10)}, np.random.randn(10)), 
    (np.random.randn(5), np.random.randn(10)),
])
def test_backward_invalid_input(mock_dag, vector, array):
    # dag is mocked to avoid side effects
    a = test_utils.make_tensor(array)
    with pytest.raises(Exception) as excinfo:
        a.backward(vector)
    print(excinfo)
    assert isinstance(excinfo.value, (RuntimeError, TypeError, AttributeError))

