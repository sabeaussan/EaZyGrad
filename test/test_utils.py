import random
import numpy as np
from math import isqrt
import inspect
import eazygrad
from eazygrad._tensor import _Tensor
from hypothesis import strategies as st
import hypothesis.extra.numpy as hnp
import functools

# -----------------------------
# Property-based tests
# -----------------------------

def _random_axes(array, data):
    dims = data.draw(
        st.one_of(
            st.none(),
            st.lists(
                st.integers(0, array.ndim - 1),
                min_size=1,
                max_size=array.ndim,
                unique=True,
            ).map(tuple),
        )
    )
    if dims is not None and len(dims) == 1 and data.draw(st.booleans()):
        dims = dims[0]
    return dims

RANGE_BOUND = 5

float_strategy = st.floats(
    min_value=-RANGE_BOUND, 
    max_value=RANGE_BOUND, 
    allow_nan=False, 
    allow_infinity=False, 
    allow_subnormal=False, 
    width=32
)

int_strategy = st.integers(-RANGE_BOUND, RANGE_BOUND)

# Scalars: int or float
scalar_strategy = float_strategy | int_strategy

# N-D arrays
array_strategy = hnp.arrays(
    dtype=np.float32,
    shape=hnp.array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=5),
    elements=float_strategy,
)

# N-D arrays or scalars
array_or_scalar_strategy = hnp.arrays(
    dtype=np.float32,
    shape=hnp.array_shapes(min_dims=0, max_dims=4, min_side=1, max_side=5),
    elements=float_strategy,
)

def _random_grad(shape):
    if shape == ():
        return np.array(np.random.rand(), dtype=np.float32)
    return np.random.rand(*shape).astype(np.float32)


def transform_same(dims, shape):
    return shape

def transform_broadcastable(dims, shape):
    return tuple(d if s == 1 else s for d, s in zip(dims, shape))

# TODO: add more matmul configurations
def transform_matmul(dims, shape):
    N = len(shape)
    # handle 1-D case
    if N==1:
        return shape
    
    dims[-2]=shape[-1]
    if N==2:
        return dims
    
    for i in range(N-2):
        if dims[i]!=shape[i] and shape[i]!=1:
            dims[i]=shape[i]
    return dims

# Build broadcast compatible arrays
def compatible_shapes(transform, min_dims=1, max_dims=4, min_side=1, max_side=5):
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
        ).map(functools.partial(transform, shape=shape))

    return base_shape.flatmap(
        lambda shape: st.tuples(
            st.just(shape), make_compatible(shape)
        )
    )

array_pair_broadcast_compat_strategy = compatible_shapes(transform_broadcastable).flatmap(
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

array_pair_same_compat_strategy = compatible_shapes(transform_same).flatmap(
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

array_pair_matmul_compat_strategy = compatible_shapes(transform_matmul).flatmap(
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

def make_tensor(arr, requires_grad=True):
    t = _Tensor(arr.copy(), requires_grad=requires_grad)
    if requires_grad:
        # dag must come from _tensor to be mocked properly
        t.node_id = eazygrad._tensor.dag.create_node(parents_id=[None], operation=None, result=t, is_leaf=True)
    return t

def generate_dim_from_shape(shape, rng):
    n_dims = rng.randint(0, len(shape))
    if n_dims == 0:
        return None
    dims = rng.sample(range(len(shape)), k=n_dims)
    # print("generate_dim_from_shape : ", n_dims, dims)
    return tuple(dims)

# def generate_idxs_from_shape(shape):
#     n_dims = np.random.randint(1, len(shape)+1)
#     if n_dims == 0:
#         return None
#     dims = np.random.choice(len(shape), size=n_dims, replace=False)
    
#     return tuple(dims)

def random_split(n, rng):
    """Return a random nontrivial split (a, b) of n, or None if no such split exists."""
    divisors = []
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            divisors.append(d)
            if d != n // d:
                divisors.append(n // d)
    if not divisors:  # n is prime
        return None
    d = rng.choice(divisors)
    return d, n // d


def random_factorization(n, rng, p_trivial=0.33, p_stop=0.35):
    """
    Return a random factorization of n into arbitrarily many factors.

    p_trivial : probability of returning -1 (i.e., trivial n*1)
    p_stop    : probability of not splitting a composite factor further
    """
    # (1) trivial full stop
    if rng.random() < p_trivial:
        return -1

    result = [n]
    i = 0

    while i < len(result):
        split = random_split(result[i], rng)
        if split is not None:
            # (2) decide whether to stop early at a *non-prime composite*
            if rng.random() < p_stop or len(result)==4:
                i += 1  # keep it unsplit
                continue

            # (3) actually split
            a, b = split
            result.pop(i)
            result.extend([a, b])
        else:
            i += 1

    return result

def get_args_count(func):
    sig = inspect.signature(func)
    return len(sig.parameters)