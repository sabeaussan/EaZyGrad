import argparse
import cProfile
import pstats
import time

import numpy as np
import torch
import torch.nn as tnn

import eazygrad as ez
from eazygrad.nn.linear import Linear

np.show_config()
print(torch.__config__.show())

def run_eazygrad(batch_size, in_dim, out_dim, steps, backward, seed):
    np.random.seed(seed)

    layer = Linear(n_in=in_dim, n_out=out_dim)
    x = ez.randn((batch_size, in_dim), requires_grad=backward)

    start = time.perf_counter()
    for _ in range(steps):
        x+=1
        y = layer(x)
        if backward:
            loss = y.sum()
            loss.backward()
    elapsed = time.perf_counter() - start
    return elapsed


def run_torch(batch_size, in_dim, out_dim, steps, backward, seed):
    torch.manual_seed(seed)

    layer = tnn.Linear(in_dim, out_dim)
    x = torch.randn(batch_size, in_dim, requires_grad=backward)

    start = time.perf_counter()
    for _ in range(steps):
        x+=1 # Avoids some optimization in PyTorch that makes the benchmark less accurate
        y = layer(x)
        if backward:
            loss = y.sum()
            loss.backward()
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="cProfile benchmark for eazygrad.nn.Linear")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--in-dim", type=int, default=784)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--backward", action="store_true", help="Include backward pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sort", type=str, default="tottime", help="pstats sort key")
    parser.add_argument("--top", type=int, default=25, help="Show top N entries")
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    ez_elapsed = run_eazygrad(
        batch_size=args.batch_size,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        steps=args.steps,
        backward=args.backward,
        seed=args.seed,
    )
    profiler.disable()

    print(f"EazyGrad Linear: steps={args.steps} backward={args.backward} elapsed={ez_elapsed:.4f}s")

    stats = pstats.Stats(profiler)
    stats.sort_stats(args.sort).print_stats(args.top)

    profiler = cProfile.Profile()
    profiler.enable()
    torch_elapsed = run_torch(
        batch_size=args.batch_size,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        steps=args.steps,
        backward=args.backward,
        seed=args.seed,
    )
    profiler.disable()

    print(f"PyTorch Linear: steps={args.steps} backward={args.backward} elapsed={torch_elapsed:.4f}s")

    stats = pstats.Stats(profiler)
    stats.sort_stats(args.sort).print_stats(args.top)


if __name__ == "__main__":
    main()
