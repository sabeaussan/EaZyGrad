import argparse
import cProfile
import pstats
import time

import numpy as np
import torch

import eazygrad as ez
from eazygrad.functions.specials import logsumexp as ez_logsumexp


def run_eazygrad(x_np, dim, keepdims, steps):
    x = ez.from_numpy(x_np, requires_grad=False)
    start = time.perf_counter()
    for _ in range(steps):
        _ = ez_logsumexp(x, dim=dim, keepdims=keepdims)
    return time.perf_counter() - start


def run_torch(x_np, dim, keepdims, steps):
    x = torch.from_numpy(x_np)
    start = time.perf_counter()
    for _ in range(steps):
        _ = torch.logsumexp(x, dim=dim, keepdim=keepdims)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Benchmark eazygrad logsumexp vs PyTorch")
    parser.add_argument("--shape", type=int, nargs="+", default=[128, 784])
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--keepdims", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sort", type=str, default="tottime", help="pstats sort key")
    parser.add_argument("--top", type=int, default=25, help="Show top N entries")
    args = parser.parse_args()

    np.random.seed(args.seed)
    x_np = np.random.randn(*args.shape).astype(np.float32)

    profiler = cProfile.Profile()
    profiler.enable()
    ez_time = run_eazygrad(x_np, args.dim, args.keepdims, args.steps)
    profiler.disable()

    print("EazyGrad cProfile")
    pstats.Stats(profiler).sort_stats(args.sort).print_stats(args.top)

    profiler = cProfile.Profile()
    profiler.enable()
    torch_time = run_torch(x_np, args.dim, args.keepdims, args.steps)
    profiler.disable()

    print("PyTorch cProfile")
    pstats.Stats(profiler).sort_stats(args.sort).print_stats(args.top)

    print("LogSumExp benchmark")
    print(f"shape={tuple(args.shape)} dim={args.dim} keepdims={args.keepdims} steps={args.steps}")
    print(f"EazyGrad: {ez_time:.6f}s total | {1e6 * ez_time / args.steps:.3f} us/step")
    print(f"PyTorch : {torch_time:.6f}s total | {1e6 * torch_time / args.steps:.3f} us/step")
    if torch_time > 0:
        print(f"Speedup (PyTorch/EazyGrad): {ez_time / torch_time:.3f}x")


if __name__ == "__main__":
    main()
