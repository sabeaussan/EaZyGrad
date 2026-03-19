<p align="center">
  <img src="docs/source/_static/_logo.png" alt="EaZyGrad Logo" width="380"/>
</p>

# EaZyGrad

EaZyGrad is a small educational python-only deep learning library built to make automatic differentiation easy to read, inspect, and extend. It only uses numpy as the building block, with a sprinkle of numba compilation for speed. This is similar in spirit to karphathy's micrograd but it operates on Tensor instead of scalar values. George Hotz's tinygrad is also an inspiration but EaZyGrad will focus on readibility and simplicity. The goal is not to compete with the army of extremely talented engineers who built PyTorch but to help understand the pieces behind libraries like PyTorch:

- tensor objects,
- dynamic computation graphs,
- backward propagation,
- neural network modules,
- optimizers.

The project first started as me trying to figure out how Hessian-vector product are computed in practice. We aren't there yet but it still implement Jacobian-vector product to compute backpropagation (reverse-mode autodiff). The current API is as robust and reliable as an overfitted model on production so expect bugs and weird behaviors. Make sure to checkout the examples to see the versatility of the autograd engine. See the [project page](https://eazygrad.readthedocs.io/en/latest/index.html) for the API and other hopefully interesting stuff.

## Installation

Install from PyPI:

```bash
pip install eazygrad
```

For development:

```bash
git clone https://github.com/sabeaussan/EaZyGrad.git
cd EaZyGrad
pip install -e ".[dev]"
```

## System Requirements

- Python `>=3.10, <3.14`
- NumPy `>=2.3.0`
- Numba `>=0.63.1`
- Graphviz
- Pillow
- Matplotlib
- tqdm

Development and validation tools:

- PyTorch
- pytest
- hypothesis
- Sphinx
- Furo

## What EaZyGrad Looks Like

### 1. Basic autograd

```python
import eazygrad as ez

x = ez.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).mean()
y.backward()

print("y =", y.numpy())
print("grad =", x.grad)
```

This example shows the core workflow:

1. create tensors,
2. build a computation graph through normal tensor operations,
3. call `backward()`,
4. inspect `grad`.

### 2. Small neural network

```python
import eazygrad as ez
import eazygrad.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(ez.relu(self.fc1(x)))

model = MLP()
optimizer = ez.Adam(model.parameters(), lr=1e-3)
```

This is intentionally close to the way you would write the same code in PyTorch.

## How The Library Is Organized

The easiest way to navigate the project is to follow the same path as a forward and backward pass:

### Core tensor and graph

- [`eazygrad/_tensor.py`](eazygrad/_tensor.py): the `_Tensor` implementation and tensor methods
- [`eazygrad/grad/computation_graph.py`](eazygrad/grad/computation_graph.py): the global dynamic computation graph
- [`eazygrad/grad/operations.py`](eazygrad/grad/operations.py): local backward rules for each primitive operation

### User-facing API

- [`eazygrad/tensor_factories.py`](eazygrad/tensor_factories.py): `tensor`, `from_numpy`, `randn`, `ones`, `zeros`, `uniform`
- [`eazygrad/functions/`](eazygrad/functions): activations, math ops, reductions, special functions, losses
- [`eazygrad/optimizer.py`](eazygrad/optimizer.py): `SGD`, `Adam`, `AdamW`
- [`eazygrad/nn/`](eazygrad/nn): `Module`, `ModuleList`, `Linear`

### Data and utilities

- [`eazygrad/data/`](eazygrad/data): simple dataset and dataloader utilities
- [`eazygrad/utils/check.py`](eazygrad/utils/check.py): shape, dtype, and utility checks

## How To Read The Code

If you are new to autograd internals, this order works well:

1. Read [`eazygrad/tensor_factories.py`](eazygrad/tensor_factories.py) to see how tensors enter the system.
2. Read [`eazygrad/_tensor.py`](eazygrad/_tensor.py) to understand the public tensor API.
3. Read [`eazygrad/grad/computation_graph.py`](eazygrad/grad/computation_graph.py) to see how graph nodes are stored and traversed.
4. Read [`eazygrad/grad/operations.py`](eazygrad/grad/operations.py) to see how each operation implements its backward rule.
5. Read [`eazygrad/nn/linear.py`](eazygrad/nn/linear.py) and [`eazygrad/optimizer.py`](eazygrad/optimizer.py) to connect autograd to learning.
6. Run one of the examples and compare the training loop to a PyTorch equivalent.

## Examples Included In The Repository

### Supervised learning

- [`examples/supervised_learning/classif/main.py`](examples/supervised_learning/classif/main.py)
  - MNIST classification with an MLP
  - tracks loss and accuracy
  - saves visualizations of training curves and predictions

### Unsupervised learning

- [`examples/unsupervised_learning/GAN/main.py`](examples/unsupervised_learning/GAN/main.py)
  - simple GAN trained on MNIST
  - demonstrates alternating generator/discriminator updates
  - saves generated sample grids during training

### Reinforcement learning

- [`examples/reinforcement_learning/main.py`](examples/reinforcement_learning/main.py)
  - PPO on CartPole
  - includes rollout collection, GAE, PPO losses, and optimizer updates
  - saves reward curves and a rendered rollout of the best policy

## Current Scope And Limitations

EaZyGrad is intentionally small and educational. That also means there are limitations:

- the API surface is still compact,
- some production-grade features are not implemented yet (gpu support),
- performance is not the primary objective,
- a few internals are still evolving.

Known limitation:

- detached graph memory management is not fully solved yet, so `detach()` is intentionally not exposed as a working feature.

## Roadmap

Planned next steps:

- [ ] Implement `Conv2d`
- [ ] Implement batch normalization
- [ ] Implement attention layers
- [ ] Add GPU support (Triton perhaps)
- [ ] Add higher-order grad computation
- [ ] Improve graph lifetime management and memory behavior
- [ ] Expand module coverage and tests
- [ ] Add more tutorial-style examples

## Contributing

Issues, design discussions, documentation improvements, and small teaching-oriented features are all welcome.

If you contribute, prioritize:

- readability,
- correctness,
- small focused abstractions,
- examples and tests that help explain the behavior.
