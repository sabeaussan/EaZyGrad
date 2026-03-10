from pathlib import Path

import numpy as np
from PIL import Image

import eazygrad as ez
import eazygrad.nn as nn
from eazygrad.data.dataloader import Dataloader
from eazygrad.data.datasets import MNISTDataset

LATENT_DIM = 64
GEN_HIDDEN_DIM = 256
DISC_HIDDEN_DIM = 256
BATCH_SIZE = 128
N_EPOCHS = 100
G_LR = 2e-4
D_LR = 1e-4
LOG_INTERVAL = 100
SAMPLE_INTERVAL = 5
SAMPLE_COUNT = 16
SEED = 0
REAL_LABEL = 0.9


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_in=input_dim, n_out=hidden_dim))
        for _ in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(n_in=hidden_dim, n_out=hidden_dim))
        self.layers.append(nn.Linear(n_in=hidden_dim, n_out=output_dim))

    def forward_hidden(self, x, activation):
        y = x
        for layer in self.layers[:-1]:
            y = activation(layer(y))
        return y


class Generator(MLP):
    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__(
            input_dim=latent_dim,
            output_dim=28 * 28,
            hidden_dim=hidden_dim,
            n_hidden_layers=2,
        )

    def forward(self, z):
        hidden = self.forward_hidden(z, activation=ez.relu)
        logits = self.layers[-1](hidden)
        return ez.tanh(logits)


class Discriminator(MLP):
    def __init__(self, hidden_dim=256):
        super().__init__(
            input_dim=28 * 28,
            output_dim=1,
            hidden_dim=hidden_dim,
            n_hidden_layers=2,
        )

    def forward(self, x):
        hidden = self.forward_hidden(
            x,
            activation=lambda y: ez.relu(y),
        )
        return self.layers[-1](hidden)


def preprocess_images(images):
    flat = images.reshape(images.shape[0], -1).astype(np.float32)
    return (flat / 127.5) - 1.0


def sample_noise(batch_size, latent_dim):
    return ez.tensor(
        np.random.randn(batch_size, latent_dim).astype(np.float32),
        requires_grad=False,
    )


def discriminator_loss(discriminator, real_images, fake_images):
    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images)

    real_targets = ez.ones(real_logits.shape[0], 1, requires_grad=False) * np.float32(REAL_LABEL)
    fake_targets = ez.zeros(fake_logits.shape[0], 1, requires_grad=False)

    real_loss = ez.bce_with_logits_loss(real_logits, real_targets)
    fake_loss = ez.bce_with_logits_loss(fake_logits, fake_targets)
    return real_loss + fake_loss


def generator_loss(discriminator, fake_images):
    fake_logits = discriminator(fake_images)
    fool_targets = ez.ones(fake_logits.shape[0], 1, requires_grad=False)
    return ez.bce_with_logits_loss(fake_logits, fool_targets)


def make_image_grid(images, nrow):
    images = np.asarray(images, dtype=np.float32).reshape(-1, 28, 28)
    images = np.clip((images + 1.0) * 127.5, 0, 255).astype(np.uint8)
    n_images = images.shape[0]
    ncol = int(np.ceil(n_images / nrow))
    grid = np.zeros((ncol * 28, nrow * 28), dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = image

    return grid


@ez.no_grad
def save_samples(generator, output_dir, epoch, n_samples=SAMPLE_COUNT, latent_dim=LATENT_DIM):
    noise = sample_noise(n_samples, latent_dim)
    generated = generator(noise).numpy()
    grid = make_image_grid(generated, nrow=int(np.sqrt(n_samples)))
    Image.fromarray(grid, mode="L").save(output_dir / f"epoch_{epoch:03d}.png")

def train():
    np.random.seed(SEED)

    data_root = Path(__file__).resolve().parents[1] / "supervised_learning"
    output_dir = Path(__file__).resolve().parent / "generated_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MNISTDataset(root=str(data_root), train=True)
    loader = Dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    generator = Generator(latent_dim=LATENT_DIM, hidden_dim=GEN_HIDDEN_DIM)
    discriminator = Discriminator(hidden_dim=DISC_HIDDEN_DIM)
    g_optimizer = ez.Adam(generator.parameters(), lr=G_LR, betas=(0.5, 0.999))
    d_optimizer = ez.Adam(discriminator.parameters(), lr=D_LR, betas=(0.5, 0.999))

    save_samples(generator, output_dir=output_dir, epoch=0)

    for epoch in range(1, N_EPOCHS + 1):
        running_g_loss = 0.0
        running_d_loss = 0.0
        batch_count = 0

        for batch_idx, (images, _) in enumerate(loader, start=1):
            real_batch = ez.from_numpy(preprocess_images(images), requires_grad=False)

            noise = sample_noise(BATCH_SIZE, LATENT_DIM)
            with ez.no_grad_ctx():
                fake_batch = generator(noise)

            d_loss = discriminator_loss(discriminator, real_batch, fake_batch)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            noise = sample_noise(BATCH_SIZE, LATENT_DIM)
            fake_batch = generator(noise)
            g_loss = generator_loss(discriminator, fake_batch)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            running_d_loss += float(d_loss.numpy())
            running_g_loss += float(g_loss.numpy())
            batch_count += 1

            if batch_idx % LOG_INTERVAL == 0:
                print(
                    f"Epoch {epoch:03d} | Batch {batch_idx:04d}/{loader.num_batch:04d} "
                    f"| d_loss {running_d_loss / batch_count:.4f} "
                    f"| g_loss {running_g_loss / batch_count:.4f}"
                )
        # return

        print(
            f"Epoch {epoch:03d} complete | d_loss {running_d_loss / batch_count:.4f} "
            f"| g_loss {running_g_loss / batch_count:.4f}"
        )

        if epoch % SAMPLE_INTERVAL == 0:
            save_samples(generator, output_dir=output_dir, epoch=epoch)


if __name__ == "__main__":
    train()
