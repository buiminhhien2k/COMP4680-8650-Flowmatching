import math
import os
import argparse

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
        et = self.time_embedding(t)   # (batch, 128)
        x = torch.cat([z, et], dim=1)
        return self.net(x)

    def time_embedding(self, t):
        d = 128
        assert d % 2 == 0, "embedding dimension must be even"
        k = d // 2

        i = torch.arange(k, device=t.device, dtype=t.dtype)
        w = torch.exp(-i * math.log(10000.0) / (k - 1))   # (64,)

        angles = t * w   # (batch, 64)
        et = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return et


# ============================================================
# 2. Conversion helpers
# ============================================================

def x_from_v(z_t, v_pred, t):
    # x = z_t - t v
    return z_t - t * v_pred


def v_from_x(z_t, x_pred, t, eps_clip=1e-5):
    # v = (z_t - x) / t
    # clip t to avoid division near t=0
    t_safe = t.clamp(min=eps_clip)
    return (z_t - x_pred) / t_safe


# ============================================================
# 3. Training
# ============================================================

def train_one_model(
        dataset_name,
        dim=2,
        pred_type="v",   # "x" or "v"
        loss_type="v",   # "x" or "v"
        lr=1e-3,
        train_steps=25000,
        batch_size=1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
        print_every=1000,
        eps_clip=2e-2,
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

        x = x.to(device).float()   # (B, D)
        B, D = x.shape

        # sample t and epsilon
        # clip away from exactly 0 to protect x->v conversion
        t = torch.rand(B, 1, device=device).clamp(min=eps_clip, max=1.0 - eps_clip)
        eps = torch.randn(B, D, device=device)

        # forward process
        z_t = (1.0 - t) * x + t * eps

        # targets
        x_target = x
        v_target = eps - x

        # model output meaning depends on pred_type
        raw_pred = model(z_t, t)

        if pred_type == "x":
            x_pred = raw_pred
            v_pred = v_from_x(z_t, x_pred, t, eps_clip=eps_clip)
        elif pred_type == "v":
            v_pred = raw_pred
            x_pred = x_from_v(z_t, v_pred, t)
        else:
            raise ValueError(f"Unknown pred_type: {pred_type}")

        # loss meaning depends on loss_type
        if loss_type == "x":
            loss = F.mse_loss(x_pred, x_target)
        elif loss_type == "v":
            loss = F.mse_loss(v_pred, v_target)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % print_every == 0:
            avg_loss = sum(losses[-print_every:]) / print_every
            vis_losses.append(avg_loss)
            print(
                f"[{dataset_name} | D={dim} | pred={pred_type} | loss={loss_type}] "
                f"step {step:6d} | loss = {avg_loss:.6f}"
            )

    return model, vis_losses, data_loader


# ============================================================
# 4. Sampling with Euler ODE
# ============================================================

@torch.no_grad()
def sample_euler(
        model,
        pred_type="v",   # "x" or "v"
        num_samples=5000,
        dim=2,
        num_steps=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eps_clip=1e-5,
):
    model.eval()
    z = torch.randn(num_samples, dim, device=device)

    dt = -1.0 / num_steps   # go from t=1 to t=0

    for step in range(num_steps):
        t_value = 1.0 + step * dt
        t = torch.full((num_samples, 1), t_value, device=device).clamp(min=eps_clip)

        raw_pred = model(z, t)

        # Euler update always needs velocity
        if pred_type == "v":
            v_pred = raw_pred
        elif pred_type == "x":
            x_pred = raw_pred
            v_pred = v_from_x(z, x_pred, t, eps_clip=eps_clip)
        else:
            raise ValueError(f"Unknown pred_type: {pred_type}")

        z = z + v_pred * dt

    return z.cpu()


# ============================================================
# 5. Plotting
# ============================================================

def plot_loss(losses, dataset_name, dim, pred_type, loss_type,
              save_dir="figures_part22", interval=1000):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(range(interval, interval * len(losses) + 1, interval), losses)
    plt.title(f"Loss - {dataset_name} | D={dim} | {pred_type}-pred + {loss_type}-loss")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    filename = f"{dataset_name}_d{dim}_{pred_type}pred_{loss_type}loss_loss.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()


def plot_ground_truth_vs_generated(data_loader, generated_samples, dataset_name, dim,
                                   pred_type, loss_type, save_dir="figures_part22"):
    os.makedirs(save_dir, exist_ok=True)

    real_batch = next(iter(data_loader)).float().cpu()

    # project to 2D if needed
    if dim != 2:
        to_2d_func = data_loader.dataset.to_2d
        real_batch_2d = to_2d_func(real_batch)
        gen_2d = to_2d_func(generated_samples)
    else:
        real_batch_2d = real_batch
        gen_2d = generated_samples

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(real_batch_2d[:, 0], real_batch_2d[:, 1], s=4, alpha=0.6)
    plt.title(f"{dataset_name} - Ground Truth")
    plt.gca().set_aspect("equal")

    plt.subplot(1, 2, 2)
    plt.scatter(gen_2d[:, 0], gen_2d[:, 1], s=4, alpha=0.6)
    plt.title(f"{dataset_name} - Generated")
    plt.gca().set_aspect("equal")

    plt.suptitle(f"D={dim} | {pred_type}-pred + {loss_type}-loss")
    plt.tight_layout()

    filename = f"{dataset_name}_d{dim}_{pred_type}pred_{loss_type}loss_samples.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()


# ============================================================
# 6. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["swiss_roll", "gaussians", "circles"])
    parser.add_argument("--dim", type=int, required=True,
                        choices=[2, 8, 32])

    parser.add_argument("--pred-type", type=str, required=True,
                        choices=["x", "v"])
    parser.add_argument("--loss-type", type=str, required=True,
                        choices=["x", "v"])

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-sample-steps", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--eps-clip", type=float, default=2e-02)
    parser.add_argument("--save-dir", type=str, default="figures_part22")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_part22")

    return parser.parse_args()


# ============================================================
# 7. Main
# ============================================================

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_name = f"{args.dataset}_d{args.dim}_{args.pred_type}pred_{args.loss_type}loss.pt"
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)

    # 🔥 NEW: skip if checkpoint exists
    if os.path.exists(ckpt_path):
        print(f"[SKIP] Found existing checkpoint: {ckpt_path}")
        return

    print("==============================================")
    print(f"Training dataset={args.dataset}, D={args.dim}, "
          f"pred={args.pred_type}, loss={args.loss_type}")

    model, losses, data_loader = train_one_model(
        dataset_name=args.dataset,
        dim=args.dim,
        pred_type=args.pred_type,
        loss_type=args.loss_type,
        lr=args.lr,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        device=device,
        print_every=args.print_every,
        eps_clip=args.eps_clip,
    )

    generated_samples = sample_euler(
        model=model,
        pred_type=args.pred_type,
        num_samples=args.num_samples,
        dim=args.dim,
        num_steps=args.num_sample_steps,
        device=device,
        # eps_clip=args.eps_clip,

    )

    plot_loss(
        losses,
        dataset_name=args.dataset,
        dim=args.dim,
        pred_type=args.pred_type,
        loss_type=args.loss_type,
        save_dir=args.save_dir,
        interval=args.print_every,
    )

    plot_ground_truth_vs_generated(
        data_loader,
        generated_samples,
        dataset_name=args.dataset,
        dim=args.dim,
        pred_type=args.pred_type,
        loss_type=args.loss_type,
        save_dir=args.save_dir,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)

    torch.save(model.state_dict(), ckpt_path)

    print(f"[DONE] Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()