import random
from pathlib import Path

import numpy as np

import eazygrad as ez
import eazygrad.nn as nn
from eazygrad.data.dataloader import Dataloader
from eazygrad.grad import dag
from plot import capture_model_state, restore_model_state, save_prediction_grid, save_training_curves

SEED = 100
BATCH_SIZE = 256
N_EPOCH = 10
LR = 1e-3
INPUT_DIM = 28 * 28
HIDDEN_DIM = 128
OUTPUT_DIM = 10
N_LAYER = 2


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, n_layer=2):
        super().__init__()
        self.net = nn.ModuleList()
        # The output layer is stored first so the forward pass can mirror the
        # original hand-written architecture from the inside out.
        self.net.append(nn.Linear(n_in=h_dim, n_out=out_dim))
        for _ in range(n_layer - 1):
            self.net.append(nn.Linear(n_in=h_dim, n_out=h_dim))
        self.net.append(nn.Linear(n_in=in_dim, n_out=h_dim))

    def forward(self, x):
        y = ez.relu(self.net[-1](x))
        for i in range(1, len(self.net) - 1):
            y = ez.relu(self.net[i](y))
        return self.net[0](y)


def evaluate(model, loader):
    prev_grad_state = dag.grad_enable
    # Evaluation only needs forward values, so skip graph construction entirely.
    dag.grad_enable = False
    try:
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0
        for x_np, y_np in loader:
            # MNIST arrives as uint8 images; flatten and cast before the MLP.
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


def visualize_dag(model):
    x = ez.randn(1,784)
    y = model(x)
    y.plot_dag()
    


def main():
    np.random.seed(SEED)
    random.seed(SEED)

    train_dataset = ez.data.MNISTDataset(train=True)
    test_dataset = ez.data.MNISTDataset(train=False)
    train_loader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = Dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    figures_dir = Path(__file__).resolve().parent / "figures"

    model = Model(in_dim=INPUT_DIM, out_dim=OUTPUT_DIM, h_dim=HIDDEN_DIM, n_layer=N_LAYER)
    # visualize_dag(model)
    optimizer = ez.SGD(model.parameters(), lr=LR)
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    best_test_acc = -np.inf
    best_state = capture_model_state(model)

    print("Training EaZyGrad MLP on MNIST")
    for epoch in range(N_EPOCH):
        epoch_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0
        for x_np, y_np in train_loader:
            # Training uses the same flattening path as evaluation to keep the
            # model interface purely 2D: [batch, features].
            x = ez.from_numpy(x_np.astype(np.float32)).reshape(-1, INPUT_DIM)
            y = ez.from_numpy(y_np.astype(np.int64), requires_grad=False)

            optimizer.zero_grad()
            logits = model(x)
            loss = ez.cross_entropy_loss(logits, y)
            # print(loss)
            loss.backward()
            optimizer.step()

            preds = np.argmax(logits.numpy(), axis=1)
            correct += int(np.sum(preds == y_np))
            total += y_np.shape[0]
            epoch_loss += float(loss.numpy())
            n_batches += 1

        train_acc = correct / total if total > 0 else 0.0
        train_loss = epoch_loss / n_batches
        test_loss, test_acc = evaluate(model, test_loader)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = capture_model_state(model)
        print(
            f"epoch={epoch:04d} "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.6f} test_acc={test_acc:.4f}"
        )

    restore_model_state(model, best_state)
    curves_path = save_training_curves(history, figures_dir)
    preds_path = save_prediction_grid(
        model=model,
        test_images=test_dataset.data,
        test_labels=test_dataset.targets,
        input_dim=INPUT_DIM,
        output_dir=figures_dir,
    )
    print(f"Saved training curves to {curves_path}")
    print(f"Saved test predictions to {preds_path}")


if __name__ == "__main__":
    main()
