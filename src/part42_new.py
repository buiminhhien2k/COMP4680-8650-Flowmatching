
import torch.nn as nn
from torch.func import jvp, functional_call
import torch.nn.functional as F
import torch
import os
import math
import argparse
import matplotlib.pyplot as plt

from dataloader import get_dataloader


# ============================================================
# 1. MeanFlow Model
# ============================================================

class Model(nn.Module):
    def __init__(self, D, width=256):
        super().__init__()
        # width = max(256 + D, D**2)
        self.net = nn.Sequential(
            nn.Linear(D + 128 + 128, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, D)
        )

    def forward(self, z, t, h):
        et = self.time_embedding(t)
        eh = self.time_embedding(h)
        inp = torch.cat([z, et, eh], dim=1)
        return self.net(inp)

    def time_embedding(self, t):
        d = 128
        k = d // 2

        i = torch.arange(k, device=t.device, dtype=t.dtype)
        w = torch.exp(-i * math.log(10000.0) / (k - 1))

        angles = t * w
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


# ============================================================
# 2. Conversion helpers
# Assignment convention:
# z_t = (1 - t) x + t eps
# v = eps - x
# ============================================================

def x_from_v(z_t, v_pred, t):
    # z_t = x + t v  =>  x = z_t - t v
    return (1 - t) * v_pred + z_t


def v_from_x(z_t, x_pred, t, eps_clip=1e-5):
    # z_t = x + t v  =>  v = (z_t - x) / t
    t_safe = (1 - t).clamp(min=eps_clip)
    return (x_pred - z_t) / t_safe


# ============================================================
# 3. MeanFlow loss
# ============================================================


# ============================================================
# 3. MeanFlow loss (Corrected)
# ============================================================

def meanflow_loss(
        model,
        x,
        pred_type="v",
        loss_type="v",
        flow_matching_ratio=0.5,
        eps_clip=2e-2,
):
    B, D = x.shape
    device = x.device

    eps = torch.randn_like(x)

    # t is endpoint time
    t = torch.rand(B, 1, device=device).clamp(min=eps_clip, max=1.0 - eps_clip)

    # r is earlier time, r < t
    r = torch.rand(B, 1, device=device) * t
    h = t - r

    # 50% h = 0 standard FM safety net
    use_fm = torch.rand(B, 1, device=device) < flow_matching_ratio
    h = torch.where(use_fm, torch.zeros_like(h), h)

    # assignment convention: t=0 data, t=1 noise
    z_t = (1.0 - t) * eps + t * x

    x_target = x
    v_target = (x_target - z_t) / (1 - t)

    raw_u = model(z_t, t, h)

    # h=0 branch approximates instantaneous velocity
    with torch.no_grad():
        raw_inst = model(z_t, t, torch.zeros_like(h))

        if pred_type == "v":
            v_inst = raw_inst
        elif pred_type == "x":
            v_inst = v_from_x(z_t, raw_inst, t, eps_clip=eps_clip)
        else:
            raise ValueError(f"Unknown pred_type: {pred_type}")

    def model_func(z_in, t_in, h_in):
        return model(z_in, t_in, h_in)

    # total derivative wrt t:
    # dz_t/dt = v_inst
    # dt/dt = 1
    # dh/dt = 1 because h = t - r, holding r fixed
    _, du_dt = jvp(
        model_func,
        (z_t, t, h),
        (v_inst, torch.zeros_like(t), torch.ones_like(h))
    )

    corrected_pred = raw_u + h * du_dt

    if pred_type == "v":
        v_pred = corrected_pred
        x_pred = x_from_v(z_t, v_pred, t)
    elif pred_type == "x":
        x_pred = corrected_pred
        v_pred = v_from_x(z_t, x_pred, t, eps_clip=eps_clip)
    else:
        raise ValueError(f"Unknown pred_type: {pred_type}")

    if loss_type == "v":
        loss = F.mse_loss(v_pred, v_target)
    elif loss_type == "x":
        loss = F.mse_loss(x_pred, x_target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss

# ============================================================
# 4. Training
# ============================================================

def train_one_model(
        dataset_name,
        dim=32,
        pred_type="v",
        loss_type="v",
        lr=1e-3,
        train_steps=25000,
        batch_size=1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
        print_every=1000,
        eps_clip=2e-2,
        flow_matching_ratio=0.5,
):
    data_loader = get_dataloader(
        dataset_name,
        dim=dim,
        shuffle=True,
        batch_size=batch_size
    )

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

        x = x.to(device).float()

        loss = meanflow_loss(
            model=model,
            x=x,
            pred_type=pred_type,
            loss_type=loss_type,
            flow_matching_ratio=flow_matching_ratio,
            eps_clip=eps_clip,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % print_every == 0:
            avg_loss = sum(losses[-print_every:]) / print_every
            vis_losses.append(avg_loss)

            print(
                f"[MeanFlow | {dataset_name} | D={dim} | "
                f"pred={pred_type} | loss={loss_type}] "
                f"step {step:6d} | loss = {avg_loss:.6f}"
            )

    return model, vis_losses, data_loader


# ============================================================
# 5. MeanFlow sampling
# ============================================================

@torch.no_grad()
def sample_meanflow(
        model,
        pred_type="v",
        num_samples=5000,
        dim=32,
        num_steps=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eps_clip=1e-5,
):
    model.eval()

    # start from pure noise at t=1
    z = torch.randn(num_samples, dim, device=device)

    # move from t=1 to t=0
    time_grid = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    # dt = 1.0/num_steps
    for step in range(num_steps):
        # t_value = 0 + step * dt
        # t = torch.full((num_samples, 1), t_value, device=device).clamp(min=eps_clip)

        t_value = time_grid[step]
        r_value = time_grid[step + 1]

        h_value = t_value - r_value

        t = torch.full((num_samples, 1), t_value, device=device)
        h = torch.full((num_samples, 1), h_value, device=device)

        raw_pred = model(z, t, h)

        if pred_type == "v":
            u = raw_pred
        elif pred_type == "x":
            x_pred = raw_pred
            u = v_from_x(z, x_pred, t, eps_clip=eps_clip)
        else:
            raise ValueError(f"Unknown pred_type: {pred_type}")

        # z_r = z_t - h * u
        z = z + h_value * u

    return z.cpu()


# ============================================================
# 6. Plotting
# ============================================================

def plot_loss(losses, dataset_name, dim, pred_type, loss_type,
              save_dir="results/figures_part42", interval=1000):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(range(interval, interval * len(losses) + 1, interval), losses)
    plt.title(f"MeanFlow Loss - {dataset_name} | D={dim} | {pred_type}-pred + {loss_type}-loss")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    filename = f"{dataset_name}_d{dim}_meanflow_{pred_type}pred_{loss_type}loss_loss.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()


def plot_ground_truth_vs_generated(
        data_loader,
        generated_samples,
        dataset_name,
        dim,
        pred_type,
        loss_type,
        num_steps,
        save_dir="results/figures_part42",
):
    os.makedirs(save_dir, exist_ok=True)

    real_batch = next(iter(data_loader)).float().cpu()

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
    plt.title(f"MeanFlow Generated - {num_steps} step(s)")
    plt.gca().set_aspect("equal")

    plt.suptitle(f"D={dim} | {pred_type}-pred + {loss_type}-loss")
    plt.tight_layout()

    filename = (
        f"{dataset_name}_d{dim}_meanflow_"
        f"{pred_type}pred_{loss_type}loss_{num_steps}steps_samples.png"
    )
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()


# ============================================================
# 7. CLI
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

    parser.add_argument("--num-sample-steps", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=1024)

    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--eps-clip", type=float, default=2e-2)
    parser.add_argument("--flow-matching-ratio", type=float, default=0.2)

    parser.add_argument("--save-dir", type=str, default="results/figures_part42")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_part42")

    return parser.parse_args()


# ============================================================
# 8. Main
# ============================================================

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_name = (
        f"{args.dataset}_d{args.dim}_meanflow_"
        f"{args.pred_type}pred_{args.loss_type}loss.pt"
    )
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)

    print("==============================================")
    print(
        f"Training MeanFlow dataset={args.dataset}, D={args.dim}, "
        f"pred={args.pred_type}, loss={args.loss_type}"
    )

    if os.path.exists(ckpt_path):
        print(f"[LOAD] Found checkpoint: {ckpt_path}")
        model = Model(D=args.dim).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        data_loader = get_dataloader(
            args.dataset,
            dim=args.dim,
            shuffle=True,
            batch_size=args.batch_size
        )
        losses = []
    else:
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
            flow_matching_ratio=args.flow_matching_ratio,
        )

        os.makedirs(args.ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[DONE] Saved checkpoint: {ckpt_path}")

    for i in [1,2,5]:
        generated_samples = sample_meanflow(
            model=model,
            pred_type=args.pred_type,
            num_samples=args.num_samples,
            dim=args.dim,
            num_steps=i,
            device=device,
            eps_clip=args.eps_clip,
        )

        plot_ground_truth_vs_generated(
            data_loader=data_loader,
            generated_samples=generated_samples,
            dataset_name=args.dataset,
            dim=args.dim,
            pred_type=args.pred_type,
            loss_type=args.loss_type,
            num_steps=i,
            save_dir=args.save_dir,
        )

    if len(losses) > 0:
        plot_loss(
            losses,
            dataset_name=args.dataset,
            dim=args.dim,
            pred_type=args.pred_type,
            loss_type=args.loss_type,
            save_dir=args.save_dir,
            interval=args.print_every,
        )


    print("[DONE] Sampling and plotting completed.")


if __name__ == "__main__":
    main()