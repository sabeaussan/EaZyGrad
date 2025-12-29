import numpy as np
import pytest
from unittest.mock import MagicMock
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp
import pytensor

# TODO : test mean, sum, matmul

tensor_module = pytensor.tensor_module
_Tensor = tensor_module._Tensor

float_strategy = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, allow_subnormal=False, width=32)
bool_strategy = st.booleans()

# N-D arrays
array_strategy = hnp.arrays(
    dtype=np.float32,
    shape=hnp.array_shapes(min_dims=0, max_dims=4, min_side=1, max_side=5),
    elements=float_strategy,
)


@st.composite
def build_shape_matmul_array(draw, min_dims=1, max_dims=4, min_side=1, max_side=5):
    # TODO : finish it

    # Pick a "base" shape
    base_shape_st = hnp.array_shapes(
        min_dims=min_dims, max_dims=max_dims,
        min_side=min_side, max_side=max_side
    )

    base_shape = draw(base_shape_st)


    def make_compatible(shape):
        # Rule: prepend extra dims or use size=1 in some dims
        return st.lists(
            st.integers(1, max_side),
            min_size=1, max_size=max_dims
        ).map(lambda dims: d if s == 1 else s for dim, s in enumerate(shape))
    
    return base_shape



@pytest.fixture(autouse=True)
def mock_dag_and_ops(mocker):
    """Patch dag.create_node and operations.* so we can check calls."""
    mocker.patch.object(tensor_module, "dag")
    mocker.patch.object(tensor_module, "operations")
    tensor_module.dag.create_node = MagicMock(return_value="fake_node_id")
    return tensor_module


def make_tensor(arr, requires_grad=True):
    t = _Tensor(arr, requires_grad=requires_grad)
    t.node_id = "base_id"
    return t

def generate_dim_from_shape(shape):
    if np.random.rand() < 0.5:
        return None
    n_dims = np.random.randint(0, len(shape)+1)
    if n_dims == 0:
        return None
    dims = np.random.choice(len(shape), size=n_dims, replace=False)
    return tuple(dims)


# ---- MEAN ----
@given(array=array_strategy, keepdims=bool_strategy)
def test_mean(array, keepdims):
    a = make_tensor(array)
    dims = generate_dim_from_shape(a.shape)
    np.testing.assert_allclose(
        a.mean(dim=dims, keepdims=keepdims)._array, 
        array.mean(axis=dims, keepdims=keepdims),
        rtol=1e-2, 
        atol=5e-4
    )

# ---- SUM ----
@given(array=array_strategy, keepdims=bool_strategy)
def test_sum(array, keepdims):
    a = make_tensor(array)
    dims = generate_dim_from_shape(a.shape)
    np.testing.assert_allclose(
        a.sum(dim=dims, keepdims=keepdims)._array, 
        array.sum(axis=dims, keepdims=keepdims),
        rtol=1e-2, 
        atol=5e-4
    )

@given(build_shape_matmul_array())
def test_shape(shape):
    pass
