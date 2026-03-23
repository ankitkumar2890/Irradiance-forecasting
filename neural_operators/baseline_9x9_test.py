"""
Baseline test: predict each 1x1 pixel using the mean of its 9x9 tile.

Ground truth comes from Y in:
    /Users/IRFAN/Desktop/Irradiance-forecasting/dataset_new/test_clean.npz

Example:
    python neural_operators/baseline_9x9_test.py --num_images 1
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def block_average_prediction(y: np.ndarray, tile_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return full-size block-mean prediction and per-block means.

    Args:
        y: Ground-truth array of shape [N, C, H, W].
        tile_size: Block size (e.g., 9).

    Returns:
        pred: Full-size prediction [N, C, H, W].
        block_means: Per-block means [N, C, H//tile_size, W//tile_size].
    """
    if y.ndim != 4:
        raise ValueError(f"Expected y shape [N,C,H,W], got {y.shape}")

    n, c, h, w = y.shape
    if h % tile_size != 0 or w % tile_size != 0:
        raise ValueError(
            f"Image size ({h},{w}) is not divisible by tile_size={tile_size}."
        )

    hb = h // tile_size
    wb = w // tile_size

    # [N,C,H,W] -> [N,C,Hb,t,Wb,t]
    y_blocks = y.reshape(n, c, hb, tile_size, wb, tile_size)
    finite = np.isfinite(y_blocks)
    block_sum = np.where(finite, y_blocks, 0.0).sum(axis=(3, 5))
    block_count = finite.sum(axis=(3, 5))
    block_means = np.divide(
        block_sum,
        block_count,
        out=np.full_like(block_sum, np.nan, dtype=np.float32),
        where=block_count > 0,
    )

    # Expand each block mean back to full resolution.
    pred = np.repeat(np.repeat(block_means, tile_size, axis=2), tile_size, axis=3)
    return pred, block_means


def compute_metrics(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    """Compute masked metrics over finite target values."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    mask = np.isfinite(target) & np.isfinite(pred)
    if not np.any(mask):
        raise ValueError("No finite target values available for metric computation.")

    diff = pred - target

    mse = np.mean((diff[mask]) ** 2)
    rmse = float(np.sqrt(mse))
    mae = np.mean(np.abs(diff[mask]))
    mape = np.mean(np.abs(diff[mask]) / (np.abs(target[mask]) + eps)) * 100.0

    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "mape": float(mape),
    }


def compute_metrics_per_image(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> list[dict[str, float]]:
    """Compute metrics image-by-image, then return list of metric dicts."""
    n = target.shape[0]
    out: list[dict[str, float]] = []
    for i in range(n):
        out.append(compute_metrics(pred[i : i + 1], target[i : i + 1], eps=eps))
    return out


def print_debug_sample(y: np.ndarray, pred: np.ndarray, block_means: np.ndarray, tile_size: int, sample_idx: int) -> None:
    """Print detailed calculations for one sample image."""
    yi = y[sample_idx, 0]
    pi = pred[sample_idx, 0]
    bmi = block_means[sample_idx, 0]

    print("\n" + "=" * 70)
    print(f"DEBUG: Sample {sample_idx}")
    print("=" * 70)
    print(f"Ground truth shape: {yi.shape}")
    print(f"Prediction shape  : {pi.shape}")
    print(f"Block means shape : {bmi.shape} (each entry is one {tile_size}x{tile_size} tile mean)")

    y_finite = yi[np.isfinite(yi)]
    p_finite = pi[np.isfinite(pi)]
    print(
        "Ground truth stats (finite): "
        f"min={y_finite.min():.6f}, max={y_finite.max():.6f}, "
        f"mean={y_finite.mean():.6f}, std={y_finite.std():.6f}"
    )
    print(
        "Prediction stats   (finite): "
        f"min={p_finite.min():.6f}, max={p_finite.max():.6f}, "
        f"mean={p_finite.mean():.6f}, std={p_finite.std():.6f}"
    )

    flat_block_means = bmi.ravel()
    print("First 10 tile means:")
    print(np.array2string(flat_block_means[:10], precision=5, separator=", "))

    # Show one concrete tile calculation (first tile with finite mean, if available).
    valid_tiles = np.argwhere(np.isfinite(bmi))
    if len(valid_tiles) > 0:
        tr, tc = valid_tiles[0]
        r0, c0 = int(tr) * tile_size, int(tc) * tile_size
    else:
        r0, c0 = 0, 0
    gt_tile = yi[r0 : r0 + tile_size, c0 : c0 + tile_size]
    pred_tile = pi[r0 : r0 + tile_size, c0 : c0 + tile_size]
    tile_mask = np.isfinite(gt_tile)
    tile_mean = float(gt_tile[tile_mask].mean()) if np.any(tile_mask) else float("nan")

    print(f"\nExample tile at rows [{r0}:{r0+tile_size}] and cols [{c0}:{c0+tile_size}]:")
    print(f"Computed tile mean from GT = {tile_mean:.6f}")
    print(f"Prediction tile unique values = {np.unique(np.round(pred_tile, 8))}")
    print(f"All prediction tile values equal to tile mean? {np.allclose(pred_tile, tile_mean)}")


def save_visual(y: np.ndarray, pred: np.ndarray, out_path: Path, sample_idx: int, tile_size: int) -> None:
    """Save GT, prediction, and absolute error for one sample."""
    yi = y[sample_idx, 0]
    pi = pred[sample_idx, 0]
    abs_err = np.abs(pi - yi)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    im0 = axes[0].imshow(yi, cmap="viridis")
    axes[0].set_title(f"Ground Truth (sample {sample_idx})")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(pi, cmap="viridis")
    axes[1].set_title(f"9x9 Block-Mean Prediction (tile={tile_size})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(abs_err, cmap="hot")
    axes[2].set_title("Absolute Error")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="9x9 block-mean baseline on MODIS test set")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/Users/IRFAN/Desktop/Irradiance-forecasting/dataset_new/test_clean.npz",
        help="Path to test_clean.npz",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=9,
        help="Tile size for block averaging (default: 9)",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Which evaluated sample to print/plot in detail (default: 0)",
    )
    parser.add_argument(
        "--out_plot",
        type=str,
        default="/Users/IRFAN/Desktop/Irradiance-forecasting/checkpoints/baseline_9x9_sample.png",
        help="Path to save visualization image",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = np.load(data_path, allow_pickle=False)
    if "Y" not in data:
        raise KeyError(f"Expected key 'Y' in dataset. Found keys: {list(data.keys())}")

    y = np.asarray(data["Y"], dtype=np.float32)
    if y.ndim != 4:
        raise ValueError(f"Expected Y shape [N,C,H,W], got {y.shape}")

    total_n = y.shape[0]
    n_eval = total_n

    if not (0 <= args.sample_index < n_eval):
        raise ValueError(f"sample_index must be in [0, {n_eval - 1}]")

    print("=" * 70)
    print("9x9 BASELINE EVALUATION")
    print("=" * 70)
    print(f"Dataset path      : {data_path}")
    print(f"Total test images : {total_n}")
    print(f"Evaluating images : {n_eval} (full test set)")
    print(f"Y shape used      : {y.shape}")
    print(f"Tile size         : {args.tile_size} x {args.tile_size}")

    pred, block_means = block_average_prediction(y, tile_size=args.tile_size)

    per_image_metrics = compute_metrics_per_image(pred, y)
    aggregate = {
        "mse": float(np.mean([m["mse"] for m in per_image_metrics])),
        "rmse": float(np.mean([m["rmse"] for m in per_image_metrics])),
        "mae": float(np.mean([m["mae"] for m in per_image_metrics])),
        "mape": float(np.mean([m["mape"] for m in per_image_metrics])),
    }

    print("\nPer-image metrics (first 5 shown):")
    for i, m in enumerate(per_image_metrics[:5]):
        print(
            f"  sample {i:4d} | "
            f"MSE={m['mse']:.6f}, RMSE={m['rmse']:.6f}, MAE={m['mae']:.6f}, MAPE={m['mape']:.4f}%"
        )
    if n_eval > 5:
        print(f"  ... ({n_eval - 5} more samples not printed)")

    print("\n" + "=" * 70)
    print("AGGREGATE METRICS (mean across evaluated images)")
    print("=" * 70)
    print(f"MSE  : {aggregate['mse']:.6f}")
    print(f"RMSE : {aggregate['rmse']:.6f}")
    print(f"MAE  : {aggregate['mae']:.6f}")
    print(f"MAPE : {aggregate['mape']:.4f}%")

    print_debug_sample(
        y=y,
        pred=pred,
        block_means=block_means,
        tile_size=args.tile_size,
        sample_idx=args.sample_index,
    )

    out_plot = Path(args.out_plot)
    save_visual(y=y, pred=pred, out_path=out_plot, sample_idx=args.sample_index, tile_size=args.tile_size)
    print(f"\nSaved sample visual to: {out_plot}")


if __name__ == "__main__":
    main()
