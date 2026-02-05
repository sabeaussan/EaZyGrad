import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F

import eazygrad as ez
import eazygrad.nn as nn

SEED = 100
BATCH_SIZE = 128
N_EPOCH = 1000
LR = 1e-3
INPUT_DIM = 28 * 28
HIDDEN_DIM = 128
OUTPUT_DIM = 10
N_LAYER = 2


class EzModel(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, n_layer=2):
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(n_in=in_dim, n_out=h_dim))
        for _ in range(n_layer - 1):
            self.net.append(nn.Linear(n_in=h_dim, n_out=h_dim))
        self.net.append(nn.Linear(n_in=h_dim, n_out=out_dim))

    def forward(self, x):
        y = x
        for i in range(len(self.net) - 1):
            y = ez.relu(self.net[i](y))
        return self.net[-1](y)


class TorchModel(tnn.Module):
    def __init__(self, in_dim, out_dim, h_dim, n_layer=2):
        super().__init__()
        layers = [tnn.Linear(in_dim, h_dim)]
        for _ in range(n_layer - 1):
            layers.append(tnn.Linear(h_dim, h_dim))
        layers.append(tnn.Linear(h_dim, out_dim))
        self.net = tnn.ModuleList(layers)

    def forward(self, x):
        y = x
        for i in range(len(self.net) - 1):
            y = F.relu(self.net[i](y))
        return self.net[-1](y)


def _copy_eazygrad_weights_to_torch(ez_model, torch_model):
    with torch.no_grad():
        for ez_layer, torch_layer in zip(ez_model.net, torch_model.net):
            ez_w = np.asarray(ez_layer.parameters[0]._array, dtype=np.float32)
            ez_b = np.asarray(ez_layer.parameters[1]._array, dtype=np.float32).reshape(-1)
            torch_layer.weight.copy_(torch.from_numpy(ez_w.T.copy()))
            torch_layer.bias.copy_(torch.from_numpy(ez_b.copy()))


def _max_param_diff(ez_model, torch_model):
    max_diff = 0.0
    for ez_layer, torch_layer in zip(ez_model.net, torch_model.net):
        ez_w = ez_layer.parameters[0]._array.astype(np.float32, copy=False)
        ez_b = ez_layer.parameters[1]._array.astype(np.float32, copy=False).reshape(-1)
        tw = torch_layer.weight.detach().cpu().numpy().T
        tb = torch_layer.bias.detach().cpu().numpy()
        max_diff = max(max_diff, float(np.max(np.abs(ez_w - tw))))
        max_diff = max(max_diff, float(np.max(np.abs(ez_b - tb))))
    return max_diff


def _make_batch(dataset, start_idx, batch_size):
    x_np = dataset.data[start_idx : start_idx + batch_size].reshape(batch_size, -1).astype(np.float32)
    y_np = dataset.targets[start_idx : start_idx + batch_size].astype(np.int64)
    return ez.from_numpy(x_np), ez.from_numpy(y_np), torch.from_numpy(x_np), torch.from_numpy(y_np)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = ez.data.MNISTDataset()

    ez_model = EzModel(in_dim=INPUT_DIM, out_dim=OUTPUT_DIM, h_dim=HIDDEN_DIM, n_layer=N_LAYER)
    torch_model = TorchModel(in_dim=INPUT_DIM, out_dim=OUTPUT_DIM, h_dim=HIDDEN_DIM, n_layer=N_LAYER)
    _copy_eazygrad_weights_to_torch(ez_model, torch_model)

    ez_optimizer = ez.SGD(ez_model.net.parameters(), lr=LR)
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=LR)

    print("Training eazygrad and PyTorch with shared init, batches, and hyperparameters")
    for epoch in range(N_EPOCH):
        start_idx = np.random.randint(len(dataset.data) - BATCH_SIZE)
        x_ez, y_ez, x_torch, y_torch = _make_batch(dataset, start_idx, BATCH_SIZE)

        ez_optimizer.zero_grad()
        torch_optimizer.zero_grad()

        ez_logits = ez_model(x_ez)
        torch_logits = torch_model(x_torch)

        ez_loss = ez.cross_entropy_loss(ez_logits, y_ez)
        torch_loss = F.cross_entropy(torch_logits, y_torch)

        ez_loss.backward()
        torch_loss.backward()

        ez_optimizer.step()
        torch_optimizer.step()

        if epoch % 50 == 0:
            diff = _max_param_diff(ez_model, torch_model)
            print(
                f"epoch={epoch:04d} "
                f"ez_loss={float(ez_loss.numpy()):.6f} "
                f"torch_loss={torch_loss.item():.6f} "
                f"max_param_diff={diff:.6e}"
            )
            assert diff < 1e-5, f"Parameters diverged too much at epoch {epoch}"


if __name__ == "__main__":
    main()
