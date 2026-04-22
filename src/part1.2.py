import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import get_dataloader


# ============================================================
# 1. Model
# ============================================================

class Model(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, D)
        )

    def forward(self, z, t):
        # z.shape: (batch, D)
        # t.shape: (batch, 1)
        et = self.time_embedding(t)   # (batch, 128)
        x = torch.cat([z, et], dim=1)
        return self.net(x)

    def time_embedding(self, t):
        d = 128
        assert d % 2 == 0, "embedding dimension must be even"
        k = d // 2

        i = torch.arange(k, device=t.device, dtype=t.dtype)
        w = torch.exp(-i * math.log(10000.0) / (k - 1))   # (64,)

        angles = t * w   # (batch, 64) by broadcasting
        et = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return et


# ============================================================
# 2. Training
# ============================================================

def train_one_model(
        dataset_name,
        dim=2,
        lr=1e-3,
        train_steps=25000,
        batch_size=1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
        print_every=1000,
):
    data_loader = get_dataloader(dataset_name, dim=dim, shuffle=True, batch_size=batch_size)

    model = Model(D=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    losses = []
    vis_losses = []
    loader_iter = iter(data_loader)

    for step in range(1, train_steps + 1):
        try:
            x = next(loader_iter)
        except StopIteration:
            loader_iter = iter(data_loader)
            x = next(loader_iter)

        x = x.to(device).float()   # shape: (batch, D)
        B, D = x.shape

        # sample t and epsilon
        t = torch.rand(B, 1, device=device)
        eps = torch.randn(B, D, device=device)

        # forward process
        z_t = (1.0 - t) * x + t * eps

        # velocity target
        v_target = eps - x

        # predict velocity
        v_pred = model(z_t, t)

        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % print_every == 0:
            avg_loss = sum(losses[-print_every:]) / print_every
            vis_losses.append(loss.item())
            print(f"[{dataset_name}] step {step:6d} | loss = {avg_loss:.6f}")

    return model, vis_losses, data_loader


# ============================================================
# 3. Sampling with Euler ODE
# ============================================================

@torch.no_grad()
def sample_euler(
        model,
        num_samples=5000,
        dim=2,
        num_steps=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()
    z = torch.randn(num_samples, dim, device=device)

    dt = -1.0 / num_steps   # go from t=1 to t=0

    for step in range(num_steps):
        t_value = 1.0 + step * dt
        t = torch.full((num_samples, 1), t_value, device=device)

        v_pred = model(z, t)
        z = z + v_pred * dt

    return z.cpu()


# ============================================================
# 4. Plotting
# ============================================================

def plot_loss(losses, dataset_name, save_dir="figures_part12", interval=1000):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(range(0, len(losses)*interval, 1000), losses)
    plt.title(f"Training Loss - {dataset_name}")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_loss.png"), dpi=200)
    # plt.show()


def plot_ground_truth_vs_generated(data_loader, generated_samples, dataset_name, save_dir="figures_part12"):
    os.makedirs(save_dir, exist_ok=True)

    # take one batch of real samples for display
    real_batch = next(iter(data_loader)).float().cpu()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(real_batch[:, 0], real_batch[:, 1], s=4, alpha=0.6)
    plt.title(f"{dataset_name} - Ground Truth")
    plt.gca().set_aspect("equal")

    plt.subplot(1, 2, 2)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], s=4, alpha=0.6)
    plt.title(f"{dataset_name} - Generated")
    plt.gca().set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_generated_vs_real.png"), dpi=200)
    # plt.show()


# ============================================================
# 5. Run Part 1.2 on all 3 datasets at D=2
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    datasets = ["swiss_roll", "gaussians", "circles"]

    for dataset_name in datasets:
        print("=" * 60)
        print(f"Training {dataset_name} at D=2")

        model, losses, data_loader = train_one_model(
            dataset_name=dataset_name,
            dim=2,
            lr=1e-3,
            train_steps=25000,
            batch_size=1024,
            device=device,
            print_every=1000,
        )

        generated_samples = sample_euler(
            model=model,
            num_samples=1024,
            dim=2,
            num_steps=50,
            device=device,
        )

        plot_loss(losses, dataset_name,
                  interval=1000 # similar to print_every
                  )
        plot_ground_truth_vs_generated(data_loader, generated_samples, dataset_name)

        os.makedirs("checkpoints_part12", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"checkpoints_part12/{dataset_name}_vpred_d2.pt"
        )

        print(f"Finished {dataset_name}\n")


if __name__ == "__main__":
    main()