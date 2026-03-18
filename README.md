<p align="center">
  <img src="logo.png" alt="EaZyGrad Logo" width="250"/>
</p>

# EaZyGrad

Auto-diff made eazy

## TODO :

* Add support for multi-dimensional inputs (e.g. image segmentation).
* Increase test coverage and edge cases.

## Know issues

* Detached graph memory leak. If a tensor is detached from the global dag during forward and is not traversed during backprop, it will not be freed and will grow and lead to RAM exhaustion. Happend for the discriminator loss computation. Workaround : use no_grad context.
