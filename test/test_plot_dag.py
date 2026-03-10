import numpy as np

import eazygrad as ez


def main():
    x = ez.tensor(np.array([[1.0, -2.0], [3.0, 0.5]], dtype=np.float32), requires_grad=True)
    w = ez.tensor(np.array([[0.2], [-0.4]], dtype=np.float32), requires_grad=True)
    b = ez.tensor(np.array([[0.1]], dtype=np.float32), requires_grad=True)

    logits = ez.relu(x @ w + b)
    logits.plot_dag()
    loss = logits.mean()

    print(f"loss = {loss.numpy()}")
    loss.plot_dag(full_graph=False)


if __name__ == "__main__":
    main()
