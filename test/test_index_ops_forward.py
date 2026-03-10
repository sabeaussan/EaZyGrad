import numpy as np
import torch

import eazygrad
import test_utils


def test_getitem_forward_basic_slice():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    ez_x = test_utils.make_tensor(array, requires_grad=False)
    torch_x = torch.tensor(array)

    result = ez_x[1:, :2].numpy()
    expected = torch_x[1:, :2].numpy()

    np.testing.assert_allclose(result, expected)


def test_getitem_forward_negative_index():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    ez_x = test_utils.make_tensor(array, requires_grad=False)
    torch_x = torch.tensor(array)

    result = ez_x[-1].numpy()
    expected = torch_x[-1].numpy()

    np.testing.assert_allclose(result, expected)


def test_getitem_forward_advanced_index_1d():
    array = np.array([1.0, 4.0, 2.0, 7.0], dtype=np.float32)
    index = np.array([3, 1, 1, 0])
    ez_x = test_utils.make_tensor(array, requires_grad=False)
    torch_x = torch.tensor(array)

    result = ez_x[index].numpy()
    expected = torch_x[torch.tensor(index)].numpy()

    np.testing.assert_allclose(result, expected)


def test_getitem_forward_advanced_index_2d_gather():
    array = np.arange(12, dtype=np.float32).reshape(4, 3)
    row_index = np.arange(4)
    col_index = np.array([2, 0, 1, 2])
    ez_x = test_utils.make_tensor(array, requires_grad=False)
    torch_x = torch.tensor(array)

    result = ez_x[row_index, col_index].numpy()
    expected = torch_x[torch.tensor(row_index), torch.tensor(col_index)].numpy()

    np.testing.assert_allclose(result, expected)


def test_getitem_forward_boolean_mask():
    array = np.arange(6, dtype=np.float32)
    mask = np.array([True, False, True, False, True, False])
    ez_x = test_utils.make_tensor(array, requires_grad=False)
    torch_x = torch.tensor(array)

    result = ez_x[mask].numpy()
    expected = torch_x[torch.tensor(mask)].numpy()

    np.testing.assert_allclose(result, expected)
