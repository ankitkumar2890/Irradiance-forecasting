import multiprocessing
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── macOS fix: use 'spawn' to avoid fork-related crashes ────────────
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


def get_best_device():
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ERA5MODISDataset(Dataset):
    def __init__(self, npz_path, norm_stats=None):
        self._path = str(npz_path)

        # Temporarily open to get shape and compute norm stats, then close.
        tmp = np.load(self._path, mmap_mode="r")
        X_tmp = tmp["X"]  # [N, C, H, W]
        self.N, self.C, self.H, self.W = X_tmp.shape

        if norm_stats is None:
            ch_sum = np.zeros((1, self.C, 1, 1), dtype=np.float64)
            ch_sq_sum = np.zeros((1, self.C, 1, 1), dtype=np.float64)
            count_per_ch = np.zeros((1, self.C, 1, 1), dtype=np.float64)

            chunk = 256
            for i in range(0, self.N, chunk):
                xb = np.asarray(X_tmp[i : i + chunk], dtype=np.float32)
                xb = np.nan_to_num(xb)
                ch_sum += xb.sum(axis=(0, 2, 3), keepdims=True)
                ch_sq_sum += np.square(xb).sum(axis=(0, 2, 3), keepdims=True)
                count_per_ch += xb.shape[0] * self.H * self.W

            mean = (ch_sum / count_per_ch).astype(np.float32)
            var = (ch_sq_sum / count_per_ch) - np.square(mean, dtype=np.float32)
            std = np.sqrt(np.maximum(var, 0.0), dtype=np.float32) + 1e-6
            self.norm_stats = {"mean": mean, "std": std}
        else:
            self.norm_stats = {
                "mean": norm_stats["mean"].astype(np.float32),
                "std": norm_stats["std"].astype(np.float32),
            }

        del X_tmp, tmp  # close the temporary handle

        self.mean = self.norm_stats["mean"]
        self.std = self.norm_stats["std"]

        x = torch.linspace(0, 1, self.H)
        y = torch.linspace(0, 1, self.W)
        Xc, Yc = torch.meshgrid(x, y, indexing="ij")
        self.coords = torch.stack([Xc, Yc], dim=0)

        # Will be lazily opened per-worker in __getitem__.
        self._npz = None
        self.X_np = None
        self.Y_np = None

    def _ensure_open(self):
        """Lazily open the npz file (once per worker process)."""
        if self._npz is None:
            self._npz = np.load(self._path, mmap_mode="r")
            self.X_np = self._npz["X"]
            self.Y_np = self._npz["Y"]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        self._ensure_open()

        x = np.asarray(self.X_np[idx], dtype=np.float32)
        y = np.asarray(self.Y_np[idx], dtype=np.float32)

        m = ~np.isnan(y)
        y = np.nan_to_num(y)
        x = np.nan_to_num(x)
        x = (x - self.mean[0]) / self.std[0]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        m = torch.tensor(m, dtype=torch.float32)
        x = torch.cat([x, self.coords], dim=0)
        return x, y, m


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # MPS does not support complex64 natively; fallback to CPU for FFT if needed.
        if x.device.type == "mps":
            x_cpu = x.detach().cpu()
            x_ft = torch.fft.rfft2(x_cpu, norm="ortho")
            out_ft = torch.zeros(
                B, self.weights.shape[1], H, W // 2 + 1, dtype=torch.cfloat
            )
            out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, : self.modes1, : self.modes2],
                self.weights.detach().cpu(),
            )
            return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").to(x.device)
        else:
            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros(
                B, self.weights.shape[1], H, W // 2 + 1, device=x.device, dtype=torch.cfloat
            )
            out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, : self.modes1, : self.modes2],
                self.weights,
            )
            return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


def masked_mse(pred, target, mask):
    denom = mask.sum().clamp_min(1.0)
    return ((pred - target) ** 2 * mask).sum() / denom


def masked_mae(pred, target, mask):
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def masked_mape(pred, target, mask, eps=1e-6):
    denom = mask.sum().clamp_min(1.0)
    frac = torch.abs((pred - target) / (torch.abs(target) + eps))
    return (frac * mask).sum() / denom * 100.0


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_mse = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    total_mape = 0.0

    # Use non_blocking transfers for async CPU→GPU copy.
    non_blocking = device.type in ("cuda", "mps")

    for x, y, m in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        m = m.to(device, non_blocking=non_blocking)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            mse = masked_mse(pred, y, m)
            rmse = torch.sqrt(mse)
            mae = masked_mae(pred, y, m)
            mape = masked_mape(pred, y, m)
            if is_train:
                optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
                mse.backward()
                optimizer.step()

        total_mse += mse.item()
        total_rmse += rmse.item()
        total_mae += mae.item()
        total_mape += mape.item()

    n = max(len(loader), 1)
    return {
        "mse": total_mse / n,
        "rmse": total_rmse / n,
        "mae": total_mae / n,
        "mape": total_mape / n,
    }


def main():
    # Use local dataset directory (relative to project root)
    base_dir = Path("/Users/IRFAN/Desktop/Irradiance-forecasting")
    dataset_dir = base_dir / "dataset"

    # Prefer cleaned files; fallback to merged raw files
    train_npz = dataset_dir / "train_clean.npz"
    val_npz = dataset_dir / "validate_clean.npz"
    if not train_npz.exists():
        train_npz = dataset_dir / "train.npz"
    if not val_npz.exists():
        val_npz = dataset_dir / "validate.npz"

    if not train_npz.exists() or not val_npz.exists():
        raise FileNotFoundError(
            f"Could not find dataset files.\n"
            f"Tried train: {dataset_dir / 'train_clean.npz'} and {dataset_dir / 'train.npz'}\n"
            f"Tried val:   {dataset_dir / 'validate_clean.npz'} and {dataset_dir / 'validate.npz'}"
        )

    device = get_best_device()
    is_gpu = device.type in ("cuda", "mps")
    print(f"Device: {device}")
    print(f"Train file: {train_npz}")
    print(f"Val file:   {val_npz}")

    train_dataset = ERA5MODISDataset(train_npz)
    val_dataset = ERA5MODISDataset(val_npz, norm_stats=train_dataset.norm_stats)

    # macOS with MPS: use fewer workers to avoid memory pressure; pin_memory only for CUDA.
    n_workers = 2 if device.type == "mps" else 4
    pin = device.type == "cuda"  # pin_memory is only useful for CUDA, not MPS
    prefetch = 2

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=n_workers, pin_memory=pin,
        prefetch_factor=prefetch, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=n_workers, pin_memory=pin,
        prefetch_factor=prefetch, persistent_workers=True,
    )

    model = FNO2d(
        modes1=16,
        modes2=16,
        width=64,
        in_channels=1 + 2,   # 1 ERA5 channel + 2 coordinate channels
        out_channels=1,
    ).to(device)

    epochs = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_state = None

    print(f"\nStarting training — {epochs} epochs, device={device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    for epoch in range(epochs):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        elapsed = time.time() - t0

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:02d}/{epochs} ({elapsed:.1f}s) | "
            f"Train MSE {train_metrics['mse']:.6f} | "
            f"Train RMSE {train_metrics['rmse']:.6f} | "
            f"Train MAE {train_metrics['mae']:.6f} | "
            f"Train MAPE {train_metrics['mape']:.2f}% | "
            f"Val MSE {val_metrics['mse']:.6f} | "
            f"Val RMSE {val_metrics['rmse']:.6f} | "
            f"Val MAE {val_metrics['mae']:.6f} | "
            f"Val MAPE {val_metrics['mape']:.2f}% | "
            f"LR {scheduler.get_last_lr()[0]:.2e}"
        )
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save outputs under fno_dataset_10/checkpoints
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "fno2d_best.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "modes1": 16,
            "modes2": 16,
            "width": 64,
            "in_channels": 1 + 2,
            "out_channels": 1,
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "norm_stats": {
            "mean": train_dataset.norm_stats["mean"],
            "std": train_dataset.norm_stats["std"],
        },
        "best_val_mse": best_val,
        "epochs_trained": epochs,
    }
    torch.save(checkpoint, ckpt_path)
    print(f"\nCheckpoint saved to: {ckpt_path}")

    np.savez(
        ckpt_dir / "norm_stats.npz",
        mean=train_dataset.norm_stats["mean"],
        std=train_dataset.norm_stats["std"],
    )
    print(f"Norm stats saved to: {ckpt_dir / 'norm_stats.npz'}")

    x, y, m = val_dataset[0]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).cpu()[0]

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.title("Prediction")
    plt.imshow(pred[0], cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(y[0], cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Mask")
    plt.imshow(m[0], cmap="gray")
    plt.savefig(ckpt_dir / "sample_prediction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Sample plot saved to: {ckpt_dir / 'sample_prediction.png'}")


if __name__ == "__main__":
    main()
