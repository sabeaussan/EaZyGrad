import numpy as np
import pytest
from unittest.mock import MagicMock
from hypothesis import given, settings, strategies as st
from hypothesis import HealthCheck
import hypothesis.extra.numpy as hnp
import pytensor
from pytensor import dag
import torch
import operator
import random
from utils import generate_dim_from_shape, random_factorization, get_args_count

tensor_module = pytensor.tensor_module
_Tensor = tensor_module._Tensor

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
    dims = generate_dim_from_shape(tensor.shape, rng)
    return tensor.mean(dim=dims, keepdims=keep_dims)

def sum_op(tensor):
    rng, seed = get_rng()
    keep_dims = rng.random() > 0.5
    dims = generate_dim_from_shape(tensor.shape, rng)
    return tensor.sum(dim=dims, keepdims=keep_dims)


def reshape_op(tensor):
    if len(tensor.shape)==0:
        # no op
        return tensor.reshape(-1)
    rng, seed = get_rng()
    shape = random_factorization(np.prod(tensor.shape), rng)
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
    # operator.mul,  
    reshape_op,
    sum_op,
    getitem_op,
    mean_op,
]





float_strategy = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, allow_subnormal=False, width=32)

# Scalars: int or float
scalar_strategy = float_strategy | st.integers(-1e3, 1e3)

# N-D arrays
array_strategy = hnp.arrays(
    dtype=np.float32,
    shape=hnp.array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=5),
    elements=float_strategy,
)

def broadcastable_shapes(min_dims=1, max_dims=4, min_side=1, max_side=5):
    """Generate a pair of broadcast-compatible shapes."""
    
    # Pick a "base" shape
    base_shape = hnp.array_shapes(
        min_dims=min_dims, max_dims=max_dims,
        min_side=min_side, max_side=max_side
    )

    def make_compatible(shape):
        # Rule: prepend extra dims or use size=1 in some dims
        return st.lists(
            st.one_of(st.just(1), st.integers(1, max_side)),
            min_size=len(shape), max_size=len(shape)
        ).map(lambda dims: tuple(d if s == 1 else s for d, s in zip(dims, shape)))

    return base_shape.flatmap(
        lambda shape: st.tuples(
            st.just(shape), make_compatible(shape)
        )
    )

array_pair_strategy = broadcastable_shapes().flatmap(
    lambda shapes: st.tuples(
        hnp.arrays(
            dtype=np.float32, shape=shapes[0],
            elements=float_strategy
        ),
        hnp.arrays(
            dtype=np.float32, shape=shapes[1],
            elements=float_strategy
        )
    )
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(arrays=array_pair_strategy)
def test_check_broadcasted_shape(mock_dag, arrays):
    a, b = map(make_tensor, arrays)
    c = a + b
    grad_a = pytensor.check_broadcasted_shape(
        grad=c._array,
        tensor=a
	)
    grad_b = pytensor.check_broadcasted_shape(
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
    a = make_tensor(array)
    with pytest.raises(Exception) as excinfo:
        a.backward(vector)
    print(excinfo)
    assert isinstance(excinfo.value, (RuntimeError, TypeError, AttributeError))
    
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


def generate_broadcast_compat_array(array_shape):
    # TODO : padd to max len
    shape = np.random.randint(low=1, high=5, size=4)
    array_shape, shape = pad_to_longest_shape(array_shape, shape)
    new_shape = []
    for d, s in zip(shape, array_shape):
        if s!=1 and d!=s:
            d=s
        new_shape.append(d)
    array = np.random.uniform(low=-ARRAY_BOUND, high=ARRAY_BOUND, size=new_shape).astype(np.float32)
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
        return make_tensor(other_array), make_torch_tensor(other_array)
    else:
        id_ = find_broadcast_compat_array(input_array_shape)
        if id_ is None:
            other_array = generate_broadcast_compat_array(input_array_shape)
            return make_tensor(other_array), make_torch_tensor(other_array)
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
    tensor = torch.tensor(array, requires_grad=True)
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
    num_args = get_args_count(op) 
    if num_args > 1:
        ez_t, torch_t = sample_tensors(input_ez.shape, pnew)
        ez_tensors.append(ez_t)
        torch_tensors.append(torch_t)
    
    out_ez = op(*ez_tensors)
    out_torch = apply_torch_op(op, torch_tensors)
    np.testing.assert_allclose(out_ez.numpy(), out_torch.detach().numpy(), rtol=1e-2, atol=5e-4, err_msg=f"{ez_tensors} vs {torch_tensors}")
    return out_ez, out_torch 

def sample_computation_graph(seed):
    # Define basic properties of the graph
    np.random.seed(seed)
    random.seed(seed)
    num_nodes = np.random.randint(low=15, high=25)
    p_new = np.random.uniform(low=0.2, high=0.3)
    cnt_node = 0

    # Generate a first array
    shape = np.random.randint(low=1, high=5, size=4)
    input_array = np.random.uniform(low=-ARRAY_BOUND, high=ARRAY_BOUND, size=shape).astype(np.float32)
    in_ez, in_torch = make_tensor(input_array), make_torch_tensor(input_array)
    out_ez, out_torch = sample_node(input_ez=in_ez, input_torch=in_torch, pnew=p_new)
    while cnt_node < num_nodes:
        out_ez, out_torch = sample_node(input_ez=out_ez, input_torch=out_torch, pnew=p_new)
        cnt_node += 1

    # Compute and return root nodes
    return sample_root(output_ez=out_ez, output_torch=out_torch)

def check_grads():
    for i in reversed(dag_torch.keys()):
        torch_grad = dag_torch[i].grad.numpy()
        ez_grad = dag.node_map[i].result.grad
        np.testing.assert_allclose(torch_grad, ez_grad, rtol=1e-2, atol=5e-4, err_msg=f"node {i}")


def test_computation_graph():
    torch.set_num_threads(1)
    global node_torch_id
    global dag_torch
    for seed in range(500):
        print(f"============ {seed} ============")
        rg, rt = sample_computation_graph(seed)
        np.testing.assert_allclose(rg._array, rt.detach().numpy(), rtol=1e-2, atol=5e-4)
        # print("Forward pass outputs match!")
        # dag.plot()  
        rg.backward(retain_graph=True)
        rt.backward()
        check_grads()
        # print("Backward pass completed!")
        node_torch_id = 0
        dag_torch = {}
        dag.clear()

# test_computation_graph()