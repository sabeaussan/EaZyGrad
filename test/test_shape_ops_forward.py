import numpy as np
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp
import eazygrad
import test_utils


@given(data=st.data())
def test_reshape_forward(data):
    array = data.draw(test_utils.array_strategy)
    size = int(np.prod(array.shape))
    shapes = [(-1,), (size,)]
    if size > 1:
        shapes.append((1, size))
    new_shape = data.draw(st.sampled_from(shapes))
    t = test_utils.make_tensor(array, requires_grad=False)
    result = t.reshape(*new_shape).numpy()
    expected = array.reshape(new_shape)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(data=st.data())
def test_unsqueeze_forward_single_axis(data):
    array = data.draw(test_utils.array_or_scalar_strategy)
    ndim = array.ndim
    axis = data.draw(st.integers(-ndim - 1, ndim))
    t = test_utils.make_tensor(array, requires_grad=False)
    result = t.unsqueeze(axis).numpy()
    expected = np.expand_dims(array, axis=axis)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(data=st.data())
def test_unsqueeze_forward_multi_axis(data):
    array = data.draw(test_utils.array_or_scalar_strategy)
    ndim = array.ndim
    num_axes = data.draw(st.integers(2, 3))
    out_ndim = ndim + num_axes
    axes = data.draw(
        st.lists(
            st.integers(0, out_ndim - 1),
            min_size=num_axes,
            max_size=num_axes,
            unique=True,
        ).map(tuple)
    )
    print(axes)
    t = test_utils.make_tensor(array, requires_grad=False)
    result = t.unsqueeze(*axes).numpy()
    expected = np.expand_dims(array, axis=axes)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(array=test_utils.array_or_scalar_strategy)
def test_squeeze_forward_all(array):
    t = test_utils.make_tensor(array, requires_grad=False)
    result = t.squeeze().numpy()
    expected = np.squeeze(array)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(data=st.data())
def test_squeeze_forward_axes(data):
    ndim = data.draw(st.integers(min_value=1, max_value=4))
    shape = data.draw(
        st.lists(
            st.integers(min_value=1, max_value=4),
            min_size=ndim,
            max_size=ndim,
        )
    )
    one_idx = data.draw(st.integers(min_value=0, max_value=ndim - 1))
    shape[one_idx] = 1
    array = data.draw(
        hnp.arrays(dtype=np.float32, shape=tuple(shape), elements=test_utils.float_strategy)
    )
    squeeze_axes = [i for i, s in enumerate(shape) if s == 1]
    axes = data.draw(
        st.lists(
            st.sampled_from(squeeze_axes),
            min_size=1,
            max_size=len(squeeze_axes),
            unique=True,
        )
    )
    axes = tuple(
        a - ndim if data.draw(st.booleans()) else a
        for a in axes
    )
    t = test_utils.make_tensor(array, requires_grad=False)
    result = t.squeeze(*axes).numpy()
    expected = np.squeeze(array, axis=axes)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(data=st.data())
def test_swapdims_forward(data):
    array = data.draw(
        hnp.arrays(
            dtype=np.float32,
            shape=hnp.array_shapes(min_dims=2, max_dims=4, min_side=1, max_side=5),
            elements=test_utils.float_strategy,
        )
    )
    ndim = array.ndim
    dim1 = data.draw(st.integers(-ndim, ndim - 1))
    dim2 = data.draw(st.integers(-ndim, ndim - 1))
    t = test_utils.make_tensor(array, requires_grad=False)
    result = t.swapdims(dim1, dim2).numpy()
    expected = np.swapaxes(array, dim1, dim2)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@given(data=st.data())
def test_transpose_forward(data):
    array = data.draw(
        hnp.arrays(
            dtype=np.float32,
            shape=hnp.array_shapes(min_dims=2, max_dims=4, min_side=1, max_side=5),
            elements=test_utils.float_strategy,
        )
    )
    t = test_utils.make_tensor(array, requires_grad=False)
    result = t.T().numpy()
    expected = np.swapaxes(array, -1, -2)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
