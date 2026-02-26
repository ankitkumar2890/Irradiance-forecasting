"""
FNO2d – Test / Inference Script
================================
Loads a saved checkpoint and evaluates on the validation (or test) dataset.

Usage:
    python fno_test.py                          # uses defaults
    python fno_test.py --checkpoint path/to/ckpt.pt --data path/to/val.npz
    python fno_test.py --num_samples 10         # visualise 10 random samples
"""

import argparse
import multiprocessing
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Re-use model & dataset classes from training script ──────────────
from fno_train import ERA5MODISDataset, FNO2d, get_best_device, masked_mae, masked_mape, masked_mse

# ── macOS fix: use 'spawn' to avoid fork-related crashes ─────────────
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


def load_model_from_checkpoint(ckpt_path, device):
    """Load the FNO2d model from a saved checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = checkpoint["model_config"]
    model = FNO2d(
        modes1=cfg["modes1"],
        modes2=cfg["modes2"],
        width=cfg["width"],
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from: {ckpt_path}")
    print(f"  Architecture : modes={cfg['modes1']}x{cfg['modes2']}, "
          f"width={cfg['width']}, in={cfg['in_channels']}, out={cfg['out_channels']}")
    print(f"  Best val MSE : {checkpoint.get('best_val_mse', 'N/A')}")
    print(f"  Epochs trained: {checkpoint.get('epochs_trained', 'N/A')}")

    return model, checkpoint


@torch.no_grad()
def evaluate(model, loader, device):
    """Run full evaluation and return aggregate metrics."""
    model.eval()
    total_mse = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    total_mape = 0.0
    n_batches = 0

    non_blocking = device.type in ("cuda", "mps")

    for x, y, m in loader:
        x = x.to(device, non_blocking=non_blocking)
        y = y.to(device, non_blocking=non_blocking)
        m = m.to(device, non_blocking=non_blocking)

        pred = model(x)

        mse = masked_mse(pred, y, m)
        rmse = torch.sqrt(mse)
        mae = masked_mae(pred, y, m)
        mape = masked_mape(pred, y, m)

        total_mse += mse.item()
        total_rmse += rmse.item()
        total_mae += mae.item()
        total_mape += mape.item()
        n_batches += 1

    n = max(n_batches, 1)
    metrics = {
        "mse": total_mse / n,
        "rmse": total_rmse / n,
        "mae": total_mae / n,
        "mape": total_mape / n,
    }
    return metrics


@torch.no_grad()
def visualise_samples(model, dataset, device, num_samples=6, save_path=None):
    """Generate a grid of prediction vs ground-truth plots."""
    model.eval()
    n = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), size=n, replace=False)
    indices.sort()

    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        x, y, m = dataset[idx]
        pred = model(x.unsqueeze(0).to(device)).cpu()[0]

        # Prediction
        im0 = axes[row, 0].imshow(pred[0], cmap="viridis")
        axes[row, 0].set_title(f"Prediction (sample {idx})")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        # Ground Truth
        im1 = axes[row, 1].imshow(y[0], cmap="viridis")
        axes[row, 1].set_title(f"Ground Truth (sample {idx})")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        # Absolute Error (masked)
        error = torch.abs(pred[0] - y[0]) * m[0]
        im2 = axes[row, 2].imshow(error, cmap="hot")
        axes[row, 2].set_title(f"Abs Error × Mask (sample {idx})")
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualisation saved to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="FNO2d Test / Inference")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint .pt file (default: auto-detect)",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to validation/test .npz file (default: auto-detect)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for evaluation (default: 16)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=6,
        help="Number of random samples to visualise (default: 6)",
    )
    parser.add_argument(
        "--save_plots", action="store_true",
        help="Save visualisation plots to the checkpoint directory",
    )
    args = parser.parse_args()

    # Use local dataset directory (same layout as fno_train.py)
    base_dir = Path("/Users/IRFAN/Desktop/Irradiance-forecasting")
    dataset_dir = base_dir / "dataset"
    ckpt_dir = base_dir / "checkpoints"

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path("/Users/IRFAN/Desktop/Irradiance-forecasting/checkpoints/fno2d_best.pt")

    if args.data:
        data_path = Path(args.data)
    else:
        data_path = dataset_dir / "validate_clean.npz"
        if not data_path.exists():
            data_path = dataset_dir / "validate.npz"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    device = get_best_device()
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────
    model, checkpoint = load_model_from_checkpoint(ckpt_path, device)

    # ── Load dataset with the same normalisation used during training ─
    norm_stats = checkpoint.get("norm_stats", None)
    if norm_stats is not None:
        print("Using normalisation stats from checkpoint.")
    else:
        # Fallback: try loading from separate file
        norm_path = ckpt_path.parent / "norm_stats.npz"
        if norm_path.exists():
            ns = np.load(norm_path)
            norm_stats = {"mean": ns["mean"], "std": ns["std"]}
            print(f"Using normalisation stats from: {norm_path}")
        else:
            print("WARNING: No normalisation stats found — will compute from test data!")
            norm_stats = None

    dataset = ERA5MODISDataset(data_path, norm_stats=norm_stats)

    pin = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin)
    print(f"Test dataset: {len(dataset)} samples from {data_path}")

    # ── Evaluate ─────────────────────────────────────────────────────
    print("\nEvaluating...")
    metrics = evaluate(model, loader, device)

    print("\n" + "=" * 50)
    print("  TEST RESULTS")
    print("=" * 50)
    print(f"  MSE  : {metrics['mse']:.6f}")
    print(f"  RMSE : {metrics['rmse']:.6f}")
    print(f"  MAE  : {metrics['mae']:.6f}")
    print(f"  MAPE : {metrics['mape']:.2f}%")
    print("=" * 50)

    # ── Visualise ────────────────────────────────────────────────────
    save_path = None
    if args.save_plots:
        save_path = ckpt_path.parent / "test_predictions.png"

    visualise_samples(
        model, dataset, device,
        num_samples=args.num_samples,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
