from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import eazygrad as ez
from eazygrad.grad import dag


def capture_model_state(model):
    return [param.numpy() for param in model.parameters()]


def restore_model_state(model, state):
    for param, saved in zip(model.parameters(), state):
        param._array.flags.writeable = True
        param._array[...] = saved


def _predict(model, images, input_dim):
    prev_grad_state = dag.grad_enable
    dag.grad_enable = False
    try:
        x = ez.from_numpy(images.astype(np.float32), requires_grad=False).reshape(-1, input_dim)
        return model(x).numpy()
    finally:
        dag.grad_enable = prev_grad_state


def save_training_curves(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    epochs = history["epoch"]

    axes[0].plot(epochs, history["train_loss"], label="train", linewidth=2)
    axes[0].plot(epochs, history["test_loss"], label="test", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, history["train_acc"], label="train", linewidth=2)
    axes[1].plot(epochs, history["test_acc"], label="test", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    best_idx = int(np.argmax(history["test_acc"]))
    fig.suptitle(
        f"MNIST training summary | best epoch={history['epoch'][best_idx]} "
        f"| test acc={history['test_acc'][best_idx]:.4f}"
    )

    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_prediction_grid(model, test_images, test_labels, input_dim, output_dir, n_samples=12):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logits = _predict(model, test_images, input_dim)
    preds = np.argmax(logits, axis=1)

    # Show a representative sample of the test set instead of over-sampling
    # mistakes, otherwise a high-accuracy model can still look visually poor.
    rng = np.random.default_rng(0)
    selected = rng.choice(len(test_labels), size=min(n_samples, len(test_labels)), replace=False)

    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for ax, idx in zip(axes.flat, selected):
        pred = int(preds[idx])
        label = int(test_labels[idx])
        correct = pred == label
        ax.imshow(test_images[idx], cmap="gray")
        ax.set_title(
            f"pred={pred} | label={label}",
            color=("tab:green" if correct else "tab:red"),
            fontsize=10,
        )
        ax.axis("off")

    accuracy = float(np.mean(preds == test_labels))
    fig.suptitle(f"Best-model predictions on MNIST test digits | sample accuracy={accuracy:.4f}")

    path = output_dir / "test_predictions.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
