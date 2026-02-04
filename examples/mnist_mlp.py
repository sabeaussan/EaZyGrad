import random

import numpy as np

import eazygrad as ez
import eazygrad.nn as nn
from eazygrad.data.dataloader import Dataloader
from eazygrad.grad import dag

SEED = 100
BATCH_SIZE = 128
N_EPOCH = 100
LR = 1e-3
INPUT_DIM = 28 * 28
HIDDEN_DIM = 128
OUTPUT_DIM = 10
N_LAYER = 2


class Model(nn.Module):
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


def evaluate(model, loader):
    prev_grad_state = dag.grad_enable
    dag.grad_enable = False
    try:
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0
        for x_np, y_np in loader:
            x = ez.from_numpy(x_np.astype(np.float32), requires_grad=False).reshape(-1, INPUT_DIM)
            y = ez.from_numpy(y_np.astype(np.int64), requires_grad=False)
            logits = model(x)
            loss = ez.cross_entropy_loss(logits, y)
            preds = np.argmax(logits.numpy(), axis=1)
            correct += int(np.sum(preds == y_np))
            total += y_np.shape[0]
            total_loss += float(loss.numpy())
            n_batches += 1
        mean_loss = total_loss / n_batches if n_batches > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return mean_loss, acc
    finally:
        dag.grad_enable = prev_grad_state


def main():
    np.random.seed(SEED)
    random.seed(SEED)

    train_dataset = ez.data.MNISTDataset(train=True)
    test_dataset = ez.data.MNISTDataset(train=False)
    train_loader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = Dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = Model(in_dim=INPUT_DIM, out_dim=OUTPUT_DIM, h_dim=HIDDEN_DIM, n_layer=N_LAYER)
    optimizer = ez.SGD(model.net.parameters(), lr=LR)

    print("Training EaZyGrad MLP on MNIST")
    for epoch in range(N_EPOCH):
        epoch_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0
        for x_np, y_np in train_loader:
            x = ez.from_numpy(x_np.astype(np.float32)).reshape(-1, INPUT_DIM)
            y = ez.from_numpy(y_np.astype(np.int64), requires_grad=False)

            optimizer.zero_grad()
            logits = model(x)
            loss = ez.cross_entropy_loss(logits, y)
            loss.backward()
            optimizer.step()

            preds = np.argmax(logits.numpy(), axis=1)
            correct += int(np.sum(preds == y_np))
            total += y_np.shape[0]
            epoch_loss += float(loss.numpy())
            n_batches += 1

        train_acc = correct / total if total > 0 else 0.0
        test_loss, test_acc = evaluate(model, test_loader)
        print(
            f"epoch={epoch:04d} "
            f"train_loss={epoch_loss / n_batches:.6f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.6f} test_acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
