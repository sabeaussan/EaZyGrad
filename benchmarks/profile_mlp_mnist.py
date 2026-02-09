import argparse
import time

import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F

import eazygrad as ez
import eazygrad.nn as nn

INPUT_DIM = 28 * 28
OUTPUT_DIM = 10


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


def _extract_ez_params(ez_model):
    params = []
    for layer in ez_model.net:
        w = np.asarray(layer.parameters[0]._array, dtype=np.float32).copy()
        b = np.asarray(layer.parameters[1]._array, dtype=np.float32).reshape(-1).copy()
        params.append((w, b))
    return params


def _load_params_into_ez(ez_model, params):
    for layer, (w, b) in zip(ez_model.net, params):
        layer.parameters[0]._array[...] = w
        layer.parameters[1]._array[...] = b.reshape(1, -1)


def _load_params_into_torch(torch_model, params):
    with torch.no_grad():
        for layer, (w, b) in zip(torch_model.net, params):
            layer.weight.copy_(torch.from_numpy(w.T.copy()))
            layer.bias.copy_(torch.from_numpy(b.copy()))


def _prepare_batches(dataset, batch_size, steps, seed):
    rng = np.random.default_rng(seed)
    max_start = len(dataset.data) - batch_size
    starts = rng.integers(0, max_start + 1, size=steps)

    x_batches = []
    y_batches = []
    for start in starts:
        x_np = dataset.data[start : start + batch_size].reshape(batch_size, -1).astype(np.float32)
        y_np = dataset.targets[start : start + batch_size].astype(np.int64)
        x_batches.append(x_np)
        y_batches.append(y_np)
    return x_batches, y_batches


def benchmark_eazygrad(params, x_batches, y_batches, h_dim, n_layer, lr, warmup):
    model = EzModel(INPUT_DIM, OUTPUT_DIM, h_dim, n_layer)
    _load_params_into_ez(model, params)
    optimizer = ez.SGD(model.net.parameters(), lr=lr)

    for i in range(min(warmup, len(x_batches))):
        x_ez = ez.from_numpy(x_batches[i], requires_grad=False)
        y_ez = ez.from_numpy(y_batches[i], requires_grad=False)
        optimizer.zero_grad()
        loss = ez.cross_entropy_loss(model(x_ez), y_ez)
        loss.backward()
        optimizer.step()

    total_forward = 0.0
    total_backward = 0.0
    total_step = 0.0
    start = time.perf_counter()
    n_steps = 0
    for x_np, y_np in zip(x_batches[warmup:], y_batches[warmup:]):
        x_ez = ez.from_numpy(x_np, requires_grad=False)
        y_ez = ez.from_numpy(y_np, requires_grad=False)
        optimizer.zero_grad()
        t0 = time.perf_counter()
        loss = ez.cross_entropy_loss(model(x_ez), y_ez)
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        optimizer.step()
        t3 = time.perf_counter()
        total_forward += t1 - t0
        total_backward += t2 - t1
        total_step += t3 - t2
        n_steps += 1
    elapsed = time.perf_counter() - start
    return {
        "elapsed": elapsed,
        "n_steps": n_steps,
        "forward": total_forward,
        "backward": total_backward,
        "step": total_step,
    }


def benchmark_torch(params, x_batches, y_batches, h_dim, n_layer, lr, warmup):
    model = TorchModel(INPUT_DIM, OUTPUT_DIM, h_dim, n_layer)
    _load_params_into_torch(model, params)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for i in range(min(warmup, len(x_batches))):
        x_t = torch.from_numpy(x_batches[i])
        y_t = torch.from_numpy(y_batches[i])
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_t), y_t)
        loss.backward()
        optimizer.step()

    total_forward = 0.0
    total_backward = 0.0
    total_step = 0.0
    start = time.perf_counter()
    n_steps = 0
    for x_np, y_np in zip(x_batches[warmup:], y_batches[warmup:]):
        x_t = torch.from_numpy(x_np)
        y_t = torch.from_numpy(y_np)
        optimizer.zero_grad()
        t0 = time.perf_counter()
        loss = F.cross_entropy(model(x_t), y_t)
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        optimizer.step()
        t3 = time.perf_counter()
        total_forward += t1 - t0
        total_backward += t2 - t1
        total_step += t3 - t2
        n_steps += 1
    elapsed = time.perf_counter() - start
    return {
        "elapsed": elapsed,
        "n_steps": n_steps,
        "forward": total_forward,
        "backward": total_backward,
        "step": total_step,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile MLP+MNIST runtime: EazyGrad vs PyTorch")
    parser.add_argument("--steps", type=int, default=200, help="Total train steps per framework")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup steps per framework (excluded from timing)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--h-dim", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.warmup >= args.steps:
        raise ValueError("--warmup must be smaller than --steps")

    dataset = ez.data.MNISTDataset()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    init_ez = EzModel(INPUT_DIM, OUTPUT_DIM, args.h_dim, args.n_layer)
    shared_params = _extract_ez_params(init_ez)

    x_batches, y_batches = _prepare_batches(dataset, args.batch_size, args.steps, args.seed)

    ez_stats = benchmark_eazygrad(
        shared_params, x_batches, y_batches, args.h_dim, args.n_layer, args.lr, args.warmup
    )
    torch_stats = benchmark_torch(
        shared_params, x_batches, y_batches, args.h_dim, args.n_layer, args.lr, args.warmup
    )

    profiled_steps = args.steps - args.warmup
    profiled_samples = profiled_steps * args.batch_size

    ez_sps = profiled_samples / ez_stats["elapsed"]
    torch_sps = profiled_samples / torch_stats["elapsed"]

    print("\nMLP+MNIST Runtime Comparison (forward+backward+SGD step)")
    print(f"steps={args.steps} warmup={args.warmup} profiled_steps={profiled_steps} batch_size={args.batch_size}")
    print(f"model: input={INPUT_DIM} hidden={args.h_dim} n_layer={args.n_layer} output={OUTPUT_DIM} lr={args.lr}")
    print("-" * 72)
    print(
        f"EazyGrad : {ez_stats['elapsed']:9.4f} s total | "
        f"{1e3 * ez_stats['elapsed'] / ez_stats['n_steps']:8.3f} ms/step | "
        f"{ez_sps:10.2f} samples/s"
    )
    print(
        f"  forward={ez_stats['forward']:8.4f}s ({1e3 * ez_stats['forward'] / ez_stats['n_steps']:7.3f} ms/step) | "
        f"backward={ez_stats['backward']:8.4f}s ({1e3 * ez_stats['backward'] / ez_stats['n_steps']:7.3f} ms/step) | "
        f"sgd_step={ez_stats['step']:8.4f}s ({1e3 * ez_stats['step'] / ez_stats['n_steps']:7.3f} ms/step)"
    )
    print(
        f"PyTorch  : {torch_stats['elapsed']:9.4f} s total | "
        f"{1e3 * torch_stats['elapsed'] / torch_stats['n_steps']:8.3f} ms/step | "
        f"{torch_sps:10.2f} samples/s"
    )
    print(
        f"  forward={torch_stats['forward']:8.4f}s ({1e3 * torch_stats['forward'] / torch_stats['n_steps']:7.3f} ms/step) | "
        f"backward={torch_stats['backward']:8.4f}s ({1e3 * torch_stats['backward'] / torch_stats['n_steps']:7.3f} ms/step) | "
        f"sgd_step={torch_stats['step']:8.4f}s ({1e3 * torch_stats['step'] / torch_stats['n_steps']:7.3f} ms/step)"
    )
    print("-" * 72)
    print(f"Speedup (PyTorch/EazyGrad): {ez_stats['elapsed'] / torch_stats['elapsed']:8.3f}x")


if __name__ == "__main__":
    main()
