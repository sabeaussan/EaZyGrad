"What I cannot create, I do not understand", Richard Feynman. I am starting this blog post with a Feynman quote, first and foremost to sound clever, but also because it summarizes well the purpose of EaZyGrad : understanding automatic differentiation at a fundamental level by rebuilding it. Not just a vague idea such as "its just the chain rule bro". But a useful and reliable mental model of what pytorch does under the hood when the all mighty ".backward()" is invoked on a tensor. 

We'll use a simple example to illustrate everything:
```python
import eazygrad as ez

x = ez.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = ez.tensor([0.5, -1.0, 4.0], requires_grad=True)

z = x * y + x
loss = z.mean()

print(loss)
loss.backward()

print(x.grad)
print(y.grad)
```

This tiny program already contains most of what makes autograd interesting:
- elementwise multiplication,
- elementwise addition,
- a reduction to a scalar loss,
- and finally a backward pass that propagates gradients back to the leaf tensors `x` and `y`.

The important part is that nothing special happens when you write `z = x * y + x`. EaZyGrad executes the operations eagerly, but while doing so it also records enough information to differentiate the result later.

## The Computation Graph

The first important concept is the computation graph. At its core, a computation graph is simply a directed graph that records how a value was produced. Each node represents a value (typically a tensor), and each edge represents an operation that transforms one or more inputs into an output. Instead of thinking in terms of “running code line by line,” you can think of the program as building a graph of dependencies between values. Every time you apply an operation (addition, multiplication, exponentiation, etc.), EaZyGrad creates a new node behind the scenes.

Each node:
- stores the resulting value,
- keeps references to the parent nodes (its inputs),
- and remembers how it was computed (the operation).

You can see that directly in the tensor operators. For example, the `__add__` implementation creates the result tensor and, if gradients are required, asks the global graph to create a node:

```python
def __add__(self, other: _Tensor | float | int) -> _Tensor:
    ...
    result = _Tensor(result_arr, requires_grad=requires_grad)
    if requires_grad:
        result.node_id = dag.create_node(
            parents_id=[self.node_id, other.node_id],
            operation=operations.Add(),
            result=result,
        )
    return result
```

The graph itself stores that information in a compact node structure:

```python
class Node:
    def __init__(self, parents_id, operation, result, is_leaf=False):
        self.parents_id = parents_id
        self.operation = operation
        self.result = result
        self.is_leaf = is_leaf

def create_node(self, parents_id, operation, result, is_leaf=False):
    self.node_count += 1
    node = Node(parents_id, operation, result, is_leaf)
    self.node_map[self.node_count] = node
    self._register_node(self.node_count, parents_id)
    return self.node_count
```

There are two key ideas here.

First, each intermediate tensor gets a `node_id` that points to the operation that produced it. In other words, the tensor carries a pointer into the computation graph.

Second, the graph does not store Python source code or symbolic expressions. It stores only what is needed for differentiation:
- who the parents were,
- which backward rule to apply,
- and the resulting tensor whose gradient will later be accumulated.

Leaf tensors such as `x` and `y` are also registered, but with `is_leaf=True` and no associated operation. They are the endpoints where gradients are accumulated for the user.

By the end of the forward pass, you’ve essentially built a fully traceable history of the computation, like a recording tape. The dag can be plotted for inspection in eazygrad:

```python
loss.plot_dag()
```

This renders the subgraph rooted at `loss`. For our example, the graph contains leaf nodes for `x` and `y`, then nodes for multiplication, addition, and finally the mean reduction. Visualizing the graph is useful because it makes reverse-mode differentiation feel much less magical: `.backward()` is simply a traversal of that recorded structure.


## Playing the tape backward

The recording tape analogy was actually a strategic move on my side to transition to the backward pass. The autograd engine of pytorch implement what is called a tape-based reverse mode auto-differentiation. Operation recorded on the tape during the forward can be replayed on the reverse order when computing the gradient, starting with the root node, and traversing the graph backward in topological order (i.e each node v is visited only after all its dependencies are visited). Calling ".backward" on the root tensor will fire up this process

```python
def backward(self, vector: np.ndarray | None = None, retain_graph: bool = False) -> None:
    if vector is None:
        self.acc_grad = np.float32(1.0)
    else:
        self.acc_grad = vector
    dag.backward(self.node_id, retain_graph=retain_graph)
```

That method on the tensor is intentionally very small. Its job is only to seed the gradient at the root and delegate the real work to the computation graph. If the output is a scalar loss, the seed gradient is `1.0`, because:

$$
\frac{\partial L}{\partial L} = 1
$$

The graph then walks backward:

```python
def backward(self, root_node_id: int, retain_graph: bool = False) -> None:
    pending_nodes = []
    heapq.heappush(pending_nodes, -root_node_id)
    while pending_nodes:
        current_node_id = -heapq.heappop(pending_nodes)
        current_node = self.node_map[current_node_id]
        if not current_node.is_leaf:
            grads_inputs = current_node.operation.backward(current_node.result.acc_grad)
            ...
            for parent_id, grad in zip(parent_nodes_id, grads_inputs):
                parent = self.node_map[parent_id]
                grad = ez.check.broadcasted_shape(grad, parent.result)
                parent.result.grad += grad
                parent.result.acc_grad += grad
```

Conceptually, each step does three things:

1. take the gradient that has reached the current node,
2. apply the local backward rule of the recorded operation,
3. send the resulting gradients to the parent nodes.

This is why the operation object is stored during the forward pass. It knows how to transform an incoming gradient at the output into outgoing gradients for each input.

For example, if the current node corresponds to `u = a * b`, then the local backward rule is:

$$
\frac{\partial u}{\partial a} = b
\qquad
\frac{\partial u}{\partial b} = a
$$

So when an upstream gradient arrives, the multiplication node can immediately produce the two input gradients and push them one step further backward.

## Computing gradients along the way

This post assumes that you know what a gradient is. Lets talk about how it is efficiently computed thanks to vector-Jacobian products.

Let us go back to the example:

$$
z = x \odot y + x
\qquad\text{and}\qquad
L = \mathrm{mean}(z)
$$

where $\odot$ denotes elementwise multiplication. Written component-wise:

$$
z_i = x_i y_i + x_i
$$

and

$$
L = \frac{1}{n} \sum_{i=1}^{n} z_i
$$

The chain rule tells us that to differentiate the loss with respect to one input component, we compose derivatives along the path from the loss back to that input. For $x_i$, this gives:

$$
\frac{\partial L}{\partial x_i}
=
\frac{\partial L}{\partial z_i}
\frac{\partial z_i}{\partial x_i}
=
\frac{1}{n}(y_i + 1)
$$

Similarly, for $y_i$:

$$
\frac{\partial L}{\partial y_i}
=
\frac{\partial L}{\partial z_i}
\frac{\partial z_i}{\partial y_i}
=
\frac{1}{n}x_i
$$

For our concrete values

$$
x = [1, 2, 3], \qquad y = [0.5, -1, 4]
$$

we obtain:

$$
\nabla_x L = \frac{1}{3}[1.5, 0, 5] = [0.5, 0, 1.\overline{6}]
$$

and

$$
\nabla_y L = \frac{1}{3}[1, 2, 3] = \left[\frac{1}{3}, \frac{2}{3}, 1\right]
$$

That is the math. The implementation question is: how do we compute these gradients efficiently for large tensor programs?

The naive answer would be to build the full Jacobian matrix of every operation. But that is almost always wasteful. If an operation maps $\mathbb{R}^m$ to $\mathbb{R}^n$, its Jacobian has shape $n \times m$. In deep learning, those matrices are usually enormous, and most of the time we do not need the matrix itself. We only need its action on an incoming gradient.

This is why reverse-mode autodiff is phrased in terms of vector-Jacobian products (VJPs). If an operation is

$$
f : \mathbb{R}^m \to \mathbb{R}^n
$$

with Jacobian $J_f \in \mathbb{R}^{n \times m}$, and if an upstream gradient

$$
v = \frac{\partial L}{\partial f}
$$

arrives from later in the graph, then what we need is:

$$
v J_f
$$

This product gives the gradient with respect to the inputs of `f`, without ever materializing the full Jacobian.

That is exactly what each `operation.backward(...)` method in EaZyGrad computes. It takes the upstream gradient stored in `result.acc_grad` and returns the VJP for each parent. The engine then accumulates those gradients and continues walking backward.

This is efficient for two reasons:

- the backward pass visits each recorded operation once,
- and each operation computes only the gradient information that is actually needed by its parents.

For scalar losses, reverse mode is especially attractive because no matter how many parameters the model has, we still start from a single seed value:

$$
\frac{\partial L}{\partial L} = 1
$$

and propagate that information backward through the graph. That is why backpropagation scales so well for machine learning workloads: one forward pass builds the tape, one reverse pass applies local VJPs, and the gradients of all parameters fall out as a consequence.



## Eager mode

Eazygrad uses an "eager" execution model which apply ops and build the graph on the fly. In other words the graph is dynamic and follow the flow of the program. This is flexible, conceptually simple and very useful when debugging because it makes the extremely effective "print('here')" strategy works. But it is also a significant waste of computing ressources :  
- It requires to rebuild the graph each time 
- Ops are executed as they occur so no optimization can be done by reordering or fusing ops
- and last but not least it introduces a large python overhead.


Disclaimer : EaZyGrad is tape-based i.e giant list, not like pytorch


