"""
FNO2d – Training Script with HLS Landsat Feature Support
==========================================================
Trains a Fourier Neural Operator on ERA5 + HLS Landsat + spatial/time features
to predict MODIS Cloud Optical Thickness at 1 km resolution.

Input channel layout (when HLS is enabled in fno_dataset.py):
  Channels 0-9  : ERA5 atmospheric variables (10 channels)
                   tclw, tciw, tcwv, r@700, r@850, t@700, t@850, u@850, v@850, w@700
  Channels 10-16: HLS Landsat surface features (7 channels)
                   NDMI, NDVI, NDWI, MNDWI, SWIR1, brightness_temp, elevation
  Channel 17    : HLS availability binary mask (1=valid, 0=zero-padded)
  --- added at runtime by dataset class ---
  Channels 18-19: Normalised spatial coordinates (latitude, longitude)
  Channel 20    : Time-of-day encoding (sin)
  Channel 21    : Day-of-year encoding (sin)
  Total: 22 input channels → 1 output channel (log1p COT)

Without HLS (ERA5-only):
  Channels 0-9  : ERA5 (10)
  Channels 10-11: Spatial (2)
  Channels 12-13: Time (2)
  Total: 14 input channels
"""

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

# ── Channel layout constants ────────────────────────────────────────
# These must match the output of data/fno_dataset.py
N_ERA5_CHANNELS = 10
# HLS Landsat features: NDMI, NDVI, NDWI, MNDWI, SWIR1, brightness_temp, elevation
N_HLS_CHANNELS = 7
N_HLS_MASK_CHANNELS = 1  # binary availability mask
# Added at runtime by dataset class
N_SPATIAL_CHANNELS = 2   # normalised lat, lon
N_TIME_CHANNELS = 2      # sin(time_of_day), sin(day_of_year)

ERA5_FEATURE_NAMES = [
    "tclw", "tciw", "tcwv", "r@700", "r@850",
    "t@700", "t@850", "u@850", "v@850", "w@700",
]
HLS_FEATURE_NAMES = [
    "NDMI", "NDVI", "NDWI", "MNDWI", "SWIR1", "brightness_temp", "elevation",
]


def get_best_device():
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_channels(n_data_channels: int) -> str:
    """Return a human-readable breakdown of input channels."""
    lines = []
    idx = 0

    # ERA5 channels
    n_era5 = min(N_ERA5_CHANNELS, n_data_channels)
    lines.append(f"  Channels {idx}-{idx + n_era5 - 1}: ERA5 atmospheric ({n_era5} ch)")
    idx += n_era5

    remaining = n_data_channels - N_ERA5_CHANNELS
    has_hls = remaining >= (N_HLS_CHANNELS + N_HLS_MASK_CHANNELS)

    if has_hls:
        lines.append(f"  Channels {idx}-{idx + N_HLS_CHANNELS - 1}: HLS Landsat features ({N_HLS_CHANNELS} ch)")
        idx += N_HLS_CHANNELS
        lines.append(f"  Channel  {idx}: HLS availability mask ({N_HLS_MASK_CHANNELS} ch)")
        idx += N_HLS_MASK_CHANNELS

    # Runtime-added channels
    lines.append(f"  Channels {idx}-{idx + N_SPATIAL_CHANNELS - 1}: Spatial coordinates ({N_SPATIAL_CHANNELS} ch)")
    idx += N_SPATIAL_CHANNELS
    lines.append(f"  Channels {idx}-{idx + N_TIME_CHANNELS - 1}: Time encodings ({N_TIME_CHANNELS} ch)")
    idx += N_TIME_CHANNELS
    lines.append(f"  Total: {idx} input channels")

    return "\n".join(lines)


class ERA5HLSDataset(Dataset):
    """
    PyTorch Dataset for FNO training.

    Loads pre-built .npz shards containing:
      X: [N, C, H, W]  — input features (ERA5 + optional HLS + mask)
      Y: [N, 1, H, W]  — target (log1p Cloud Optical Thickness)
      tile_id: [N]      — spatial tile index
      timestamp: [N]    — datetime strings

    At runtime, appends spatial (lat/lon) and time (sin-encoded) channels.
    """
    # Shared MODIS domain used during dataset creation.
    DOMAIN_NORTH = 17.0
    DOMAIN_SOUTH = 8.0
    DOMAIN_WEST = 72.5
    DOMAIN_EAST = 81.5

    def __init__(self, npz_path, norm_stats=None):
        self._path = str(npz_path)

        # Temporarily open to get shape and compute norm stats, then close.
        tmp = np.load(self._path, mmap_mode="r")
        X_tmp = tmp["X"]  # [N, C, H, W]
        self.N, self.C, self.H, self.W = X_tmp.shape
        self.tile_ids = np.asarray(tmp["tile_id"], dtype=np.int32)
        self.timestamps = np.asarray(tmp["timestamp"])

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

        self._build_spatial_feature_bank()
        self._build_time_features()

        # Will be lazily opened per-worker in __getitem__.
        self._npz = None
        self.X_np = None
        self.Y_np = None

    def _build_spatial_feature_bank(self):
        max_tile_id = int(np.max(self.tile_ids))
        n_tiles = max_tile_id + 1
        tiles_w = int(round(np.sqrt(n_tiles)))
        if tiles_w * tiles_w != n_tiles:
            raise ValueError(
                f"Could not infer square tile layout from tile_id max={max_tile_id}."
            )
        tiles_h = tiles_w

        global_h = tiles_h * self.H
        global_w = tiles_w * self.W
        lat_step = (self.DOMAIN_NORTH - self.DOMAIN_SOUTH) / global_h
        lon_step = (self.DOMAIN_EAST - self.DOMAIN_WEST) / global_w

        lat_all = self.DOMAIN_NORTH - (np.arange(global_h, dtype=np.float32) + 0.5) * lat_step
        lon_all = self.DOMAIN_WEST + (np.arange(global_w, dtype=np.float32) + 0.5) * lon_step
        lat_mean, lat_std = float(lat_all.mean()), float(lat_all.std() + 1e-6)
        lon_mean, lon_std = float(lon_all.mean()), float(lon_all.std() + 1e-6)

        unique_tiles = np.unique(self.tile_ids)
        self.spatial_by_tile = {}
        for tid in unique_tiles:
            tr = int(tid) // tiles_w
            tc = int(tid) % tiles_w
            r0, r1 = tr * self.H, (tr + 1) * self.H
            c0, c1 = tc * self.W, (tc + 1) * self.W

            lat_tile = lat_all[r0:r1]
            lon_tile = lon_all[c0:c1]
            lat2d = np.repeat(lat_tile[:, None], self.W, axis=1)
            lon2d = np.repeat(lon_tile[None, :], self.H, axis=0)

            lat_z = (lat2d - lat_mean) / lat_std
            lon_z = (lon2d - lon_mean) / lon_std
            spatial = np.stack([lat_z, lon_z], axis=0).astype(np.float32)
            self.spatial_by_tile[int(tid)] = torch.from_numpy(spatial)

    def _build_time_features(self):
        ts = self.timestamps.astype("datetime64[m]")
        hour = (ts.astype("datetime64[h]").astype(np.int64) % 24).astype(np.float32)
        minute = (ts.astype(np.int64) % 60).astype(np.float32)
        tod = hour + minute / 60.0
        self.time_sin = np.sin(2.0 * np.pi * tod / 24.0).astype(np.float32)

        day = ts.astype("datetime64[D]")
        year_start = ts.astype("datetime64[Y]").astype("datetime64[D]")
        doy0 = (day - year_start).astype(np.int64).astype(np.float32)
        self.day_sin = np.sin(2.0 * np.pi * doy0 / 365.0).astype(np.float32)

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
        spatial = self.spatial_by_tile[int(self.tile_ids[idx])]
        time_ch = torch.full((1, self.H, self.W), float(self.time_sin[idx]), dtype=torch.float32)
        day_ch = torch.full((1, self.H, self.W), float(self.day_sin[idx]), dtype=torch.float32)
        x = torch.cat([x, spatial, time_ch, day_ch], dim=0)
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
            # IMPORTANT: do not detach here, otherwise gradients to the spectral branch are broken.
            x_cpu = x.to("cpu")
            w_cpu = self.weights.to("cpu")
            x_ft = torch.fft.rfft2(x_cpu, norm="ortho")
            out_ft = torch.zeros(
                B, self.weights.shape[1], H, W // 2 + 1, dtype=torch.cfloat
            )
            out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, : self.modes1, : self.modes2],
                w_cpu,
            )
            out_cpu = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
            return out_cpu.to(x.device)
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
    dataset_dir = base_dir / "dataset_new"

    # Prefer cleaned files; fallback to merged raw files
    train_npz = dataset_dir / "train_clean.npz"
    val_npz = dataset_dir / "validate_clean.npz"
    test_npz = dataset_dir / "test_clean.npz"
    if not train_npz.exists():
        train_npz = dataset_dir / "train.npz"
    if not val_npz.exists():
        val_npz = dataset_dir / "validate.npz"
    if not test_npz.exists():
        test_npz = dataset_dir / "test.npz"

    if not train_npz.exists() or not val_npz.exists() or not test_npz.exists():
        raise FileNotFoundError(
            f"Could not find dataset files.\n"
            f"Tried train: {dataset_dir / 'train_clean.npz'} and {dataset_dir / 'train.npz'}\n"
            f"Tried val:   {dataset_dir / 'validate_clean.npz'} and {dataset_dir / 'validate.npz'}\n"
            f"Tried test:  {dataset_dir / 'test_clean.npz'} and {dataset_dir / 'test.npz'}"
        )

    device = get_best_device()
    is_gpu = device.type in ("cuda", "mps")
    print(f"Device: {device}")
    print(f"Train file: {train_npz}")
    print(f"Val file:   {val_npz}")
    print(f"Test file:  {test_npz}")

    train_dataset = ERA5HLSDataset(train_npz)
    val_dataset = ERA5HLSDataset(val_npz, norm_stats=train_dataset.norm_stats)
    test_dataset = ERA5HLSDataset(test_npz, norm_stats=train_dataset.norm_stats)

    # ── Channel breakdown ────────────────────────────────────────────
    print(f"\nDataset channels in .npz: {train_dataset.C}")
    sample_x, _, _ = train_dataset[0]
    in_channels = int(sample_x.shape[0])
    print(f"Total input channels (with spatial/time): {in_channels}")
    print("Channel breakdown:")
    print(describe_channels(train_dataset.C))

    # Verify all splits have the same number of data channels
    assert val_dataset.C == train_dataset.C, (
        f"Val channels ({val_dataset.C}) != train channels ({train_dataset.C})"
    )
    assert test_dataset.C == train_dataset.C, (
        f"Test channels ({test_dataset.C}) != train channels ({train_dataset.C})"
    )

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
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        num_workers=n_workers, pin_memory=pin,
        prefetch_factor=prefetch, persistent_workers=True,
    )

    model = FNO2d(
        modes1=16,
        modes2=16,
        width=64,
        in_channels=in_channels,
        out_channels=1,
    ).to(device)

    epochs = 50
    patience = 10  # early stopping patience
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    # History for plotting
    train_history = {"mse": [], "rmse": [], "mae": [], "mape": []}
    val_history = {"mse": [], "rmse": [], "mae": [], "mape": []}

    print(f"\nStarting training — {epochs} epochs (patience={patience}), device={device}")
    print(f"Input channels: {in_channels}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    for epoch in range(epochs):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        elapsed = time.time() - t0

        # Record history
        for key in train_history:
            train_history[key].append(train_metrics[key])
            val_history[key].append(val_metrics[key])

        # Check for improvement
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            marker = " ★ new best"
        else:
            epochs_without_improvement += 1
            marker = ""

        print(
            f"Epoch {epoch + 1:02d}/{epochs} ({elapsed:.1f}s) | "
            f"Train MSE {train_metrics['mse']:.6f}  RMSE {train_metrics['rmse']:.6f}  "
            f"MAE {train_metrics['mae']:.6f}  MAPE {train_metrics['mape']:.2f}% | "
            f"Val MSE {val_metrics['mse']:.6f}  RMSE {val_metrics['rmse']:.6f}  "
            f"MAE {val_metrics['mae']:.6f}  MAPE {val_metrics['mape']:.2f}% | "
            f"LR {scheduler.get_last_lr()[0]:.2e}{marker}"
        )
        scheduler.step()

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Test evaluation ──────────────────────────────────────────────
    test_metrics = run_epoch(model, test_loader, device, optimizer=None)
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    print(f"  MSE  : {test_metrics['mse']:.6f}")
    print(f"  RMSE : {test_metrics['rmse']:.6f}")
    print(f"  MAE  : {test_metrics['mae']:.6f}")
    print(f"  MAPE : {test_metrics['mape']:.2f}%")
    print("=" * 60)

    # Save outputs under checkpoints/
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "fno2d_best.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "modes1": 16,
            "modes2": 16,
            "width": 64,
            "in_channels": in_channels,
            "out_channels": 1,
        },
        "channel_info": {
            "n_data_channels": train_dataset.C,
            "n_total_channels": in_channels,
            "n_era5": N_ERA5_CHANNELS,
            "n_hls": N_HLS_CHANNELS if train_dataset.C > N_ERA5_CHANNELS else 0,
            "n_hls_mask": N_HLS_MASK_CHANNELS if train_dataset.C > N_ERA5_CHANNELS else 0,
            "n_spatial": N_SPATIAL_CHANNELS,
            "n_time": N_TIME_CHANNELS,
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "norm_stats": {
            "mean": train_dataset.norm_stats["mean"],
            "std": train_dataset.norm_stats["std"],
        },
        "best_val_mse": best_val,
        "epochs_trained": epoch + 1,
        "test_metrics": test_metrics,
        "train_history": train_history,
        "val_history": val_history,
    }
    torch.save(checkpoint, ckpt_path)
    print(f"\nCheckpoint saved to: {ckpt_path}")

    np.savez(
        ckpt_dir / "norm_stats.npz",
        mean=train_dataset.norm_stats["mean"],
        std=train_dataset.norm_stats["std"],
    )
    print(f"Norm stats saved to: {ckpt_dir / 'norm_stats.npz'}")

    # ── Training curves plot ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_titles = {"mse": "MSE", "rmse": "RMSE", "mae": "MAE", "mape": "MAPE (%)"}
    for ax, (key, title) in zip(axes.flat, metric_titles.items()):
        ax.plot(train_history[key], label="Train", linewidth=1.5)
        ax.plot(val_history[key], label="Val", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("Training & Validation Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ckpt_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Training curves saved to: {ckpt_dir / 'training_curves.png'}")

    # ── Sample prediction plot ───────────────────────────────────────
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
