import numpy as np
import torch

import eazygrad
import test_utils


def test_getitem_backward_basic_slice(dag_setup):
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    ez_x = test_utils.make_tensor(array, requires_grad=True)
    torch_x = torch.tensor(array, requires_grad=True)

    ez_y = ez_x[1:, :2]
    torch_y = torch_x[1:, :2]

    grad_output = test_utils.random_grad(ez_y.shape)
    ez_y.backward(grad_output)
    torch_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(ez_x.grad, torch_x.grad.numpy(), atol=1e-6, rtol=1e-6)


def test_getitem_backward_advanced_index_with_repeats_1d(dag_setup):
    array = np.array([1.0, 4.0, 2.0, 7.0], dtype=np.float32)
    index = np.array([3, 1, 1, 0])
    ez_x = test_utils.make_tensor(array, requires_grad=True)
    torch_x = torch.tensor(array, requires_grad=True)

    ez_y = ez_x[index]
    torch_y = torch_x[torch.tensor(index)]

    grad_output = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    ez_y.backward(grad_output)
    torch_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(ez_x.grad, torch_x.grad.numpy(), atol=1e-6, rtol=1e-6)


def test_getitem_backward_advanced_index_2d_gather(dag_setup):
    array = np.arange(12, dtype=np.float32).reshape(4, 3)
    row_index = np.arange(4)
    col_index = np.array([2, 0, 1, 2])
    ez_x = test_utils.make_tensor(array, requires_grad=True)
    torch_x = torch.tensor(array, requires_grad=True)

    ez_y = ez_x[row_index, col_index]
    torch_y = torch_x[torch.tensor(row_index), torch.tensor(col_index)]

    grad_output = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    ez_y.backward(grad_output)
    torch_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(ez_x.grad, torch_x.grad.numpy(), atol=1e-6, rtol=1e-6)


def test_getitem_backward_advanced_index_2d_with_repeated_coordinates(dag_setup):
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    row_index = np.array([0, 0, 1, 1])
    col_index = np.array([1, 1, 2, 2])
    ez_x = test_utils.make_tensor(array, requires_grad=True)
    torch_x = torch.tensor(array, requires_grad=True)

    ez_y = ez_x[row_index, col_index]
    torch_y = torch_x[torch.tensor(row_index), torch.tensor(col_index)]

    grad_output = np.array([0.5, 0.25, 0.75, 0.1], dtype=np.float32)
    ez_y.backward(grad_output)
    torch_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(ez_x.grad, torch_x.grad.numpy(), atol=1e-6, rtol=1e-6)


def test_getitem_backward_boolean_mask(dag_setup):
    array = np.arange(6, dtype=np.float32)
    mask = np.array([True, False, True, False, True, False])
    ez_x = test_utils.make_tensor(array, requires_grad=True)
    torch_x = torch.tensor(array, requires_grad=True)

    ez_y = ez_x[mask]
    torch_y = torch_x[torch.tensor(mask)]

    grad_output = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    ez_y.backward(grad_output)
    torch_y.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(ez_x.grad, torch_x.grad.numpy(), atol=1e-6, rtol=1e-6)
