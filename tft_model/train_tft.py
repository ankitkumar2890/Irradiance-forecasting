"""
train_tft.py — Train the Temporal Fusion Transformer on CAF sliding-window data.

Usage:
    python tft_model/train_tft.py                 # full training
    python tft_model/train_tft.py --smoke-test    # quick sanity run (1 epoch, tiny data)

Outputs:
    tft_model/checkpoints/tft_best.pt             — best model weights
    tft_model/results/tft_predictions.csv          — validation predictions
    tft_model/results/tft_metrics.json             — RMSE, MAE, etc.
    tft_model/results/tft_training_curves.png      — loss curves
    tft_model/results/tft_variable_importance.png  — learned feature importance

Also writes results/finetuned_predictions.csv (same format as Moirai) so that
the existing 06_evaluate.py can be reused without modification.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    PAST_STEPS, FUTURE_STEPS,
    ENCODER_INPUT_DIM, DECODER_INPUT_DIM,
    HIDDEN_SIZE, NUM_ATTENTION_HEADS, LSTM_LAYERS, DROPOUT,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, GRADIENT_CLIP,
    LOSS_FN, SEED,
)
from dataset import get_dataloader, CAFTimeSeriesDataset
from model import TemporalFusionTransformer


# ── Feature names for interpretability ───────────────────────────────────────
ENCODER_FEATURE_NAMES = [
    "CAF", "clear_sky_ghi", "cloud_cover",
    "zenith_angle", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
DECODER_FEATURE_NAMES = [
    "clear_sky_ghi", "cloud_cover",
    "zenith_angle", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(args):
    """Select compute device, preferring CUDA > CPU. MPS only if explicit."""
    if args.smoke_test:
        return torch.device("cpu")
    if args.device != "auto":
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_loss_fn():
    if LOSS_FN == "huber":
        return nn.SmoothL1Loss()
    return nn.MSELoss()


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for enc, dec, tgt in loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
        optimizer.zero_grad()
        preds, _, _, _ = model(enc, dec)
        loss = criterion(preds, tgt)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []
    for enc, dec, tgt in loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
        preds, _, _, _ = model(enc, dec)
        loss = criterion(preds, tgt)
        if torch.isfinite(loss):
            total_loss += loss.item()
            n_batches += 1
        all_preds.append(preds.cpu().numpy())
        all_targets.append(tgt.cpu().numpy())
    avg_loss = total_loss / max(n_batches, 1)
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    return avg_loss, preds_arr, targets_arr


@torch.no_grad()
def get_variable_importance(model, loader, device):
    """Average variable selection weights across the validation set."""
    model.eval()
    enc_w_sum, dec_w_sum = None, None
    count = 0
    for enc, dec, tgt in loader:
        enc, dec = enc.to(device), dec.to(device)
        _, enc_w, dec_w, _ = model(enc, dec)
        # Average over time dimension → (B, num_features)
        enc_w_avg = enc_w.mean(dim=1)
        dec_w_avg = dec_w.mean(dim=1)
        if enc_w_sum is None:
            enc_w_sum = enc_w_avg.sum(dim=0).cpu().numpy()
            dec_w_sum = dec_w_avg.sum(dim=0).cpu().numpy()
        else:
            enc_w_sum += enc_w_avg.sum(dim=0).cpu().numpy()
            dec_w_sum += dec_w_avg.sum(dim=0).cpu().numpy()
        count += enc.shape[0]
    return enc_w_sum / count, dec_w_sum / count


def plot_training_curves(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Train MSE", linewidth=1.5)
    ax.plot(epochs, val_losses, "r-", label="Val MSE", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("TFT Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_variable_importance(enc_weights, dec_weights, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Encoder
    ax = axes[0]
    y_pos = np.arange(len(ENCODER_FEATURE_NAMES))
    ax.barh(y_pos, enc_weights, color="#4c72b0", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ENCODER_FEATURE_NAMES)
    ax.set_xlabel("Importance Weight")
    ax.set_title("Encoder (Past) Variable Importance")
    ax.invert_yaxis()

    # Decoder
    ax = axes[1]
    y_pos = np.arange(len(DECODER_FEATURE_NAMES))
    ax.barh(y_pos, dec_weights, color="#dd8452", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(DECODER_FEATURE_NAMES)
    ax.set_xlabel("Importance Weight")
    ax.set_title("Decoder (Future) Variable Importance")
    ax.invert_yaxis()

    fig.suptitle("TFT Learned Variable Importance", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_predictions_csv(preds, targets, times, save_path, also_save_compat=None):
    """
    Save predictions in the same format as Moirai's finetuned_predictions.csv
    so that 06_evaluate.py can consume them directly.
    """
    rows = []
    n_windows = preds.shape[0]
    for i in range(n_windows):
        forecast_start = pd.Timestamp(times[i, 0])
        for h in range(FUTURE_STEPS):
            ts = pd.Timestamp(times[i, h])
            rows.append({
                "datetime": ts,
                "hour": ts.hour,
                "lead_time_h": h + 1,
                "forecast_start": forecast_start,
                "CAF_true": float(targets[i, h]),
                "CAF_pred": float(np.clip(preds[i, h], 0.0, 1.0)),
            })
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    df.to_csv(save_path, index=False)

    # Write a Moirai-compatible copy so 06_evaluate.py works unchanged
    if also_save_compat:
        df.to_csv(also_save_compat, index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="Train TFT on CAF sliding-window data.")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick sanity run: 2 epochs, small batch.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = select_device(args)
    print(f"Device: {device}")

    # ── Hyperparams (override for smoke test) ────────────────────────────────
    max_epochs = 2 if args.smoke_test else MAX_EPOCHS
    patience = 1 if args.smoke_test else PATIENCE
    batch_size = min(8, BATCH_SIZE) if args.smoke_test else BATCH_SIZE

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_loader = get_dataloader("train", batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader("val", batch_size=batch_size, shuffle=False)
    print(f"  Train: {len(train_loader.dataset)} windows ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset)} windows ({len(val_loader)} batches)")

    # ── Model ────────────────────────────────────────────────────────────────
    model = TemporalFusionTransformer(
        encoder_input_dim=ENCODER_INPUT_DIM,
        decoder_input_dim=DECODER_INPUT_DIM,
        hidden_dim=HIDDEN_SIZE,
        num_heads=NUM_ATTENTION_HEADS,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        forecast_horizon=FUTURE_STEPS,
    ).to(device)

    print(f"\nTFT Architecture:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Hidden dim: {HIDDEN_SIZE}")
    print(f"  Attention heads: {NUM_ATTENTION_HEADS}")
    print(f"  LSTM layers: {LSTM_LAYERS}")
    print(f"  Forecast horizon: {FUTURE_STEPS}h")

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6,
    )
    criterion = get_loss_fn()

    # ── Training Loop ────────────────────────────────────────────────────────
    print(f"\nStarting training...")
    print(f"  LR={LEARNING_RATE}  Epochs={max_epochs}  Patience={patience}  Batch={batch_size}")
    print(f"  Loss: {LOSS_FN}  Grad clip: {GRADIENT_CLIP}\n")

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, GRADIENT_CLIP,
        )
        val_loss, val_preds, val_targets = validate(
            model, val_loader, criterion, device,
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Compute RMSE / MAE on validation
        val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
        val_mae = np.mean(np.abs(val_preds - val_targets))
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - epoch_t0

        if epoch <= 5 or epoch % 5 == 0 or epoch == max_epochs:
            print(
                f"  Epoch {epoch:3d}/{max_epochs} | "
                f"train_mse={train_loss:.5f}  val_mse={val_loss:.5f}  "
                f"val_rmse={val_rmse:.4f}  val_mae={val_mae:.4f}  "
                f"lr={lr_now:.2e}  ({elapsed:.1f}s)"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    total_time = time.time() - t0
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best val MSE: {best_val_loss:.5f}  RMSE: {best_val_loss**0.5:.4f}")

    # ── Restore best weights ─────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    # ── Save checkpoint ──────────────────────────────────────────────────────
    ckpt_name = "tft_best_smoke.pt" if args.smoke_test else "tft_best.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "lstm_layers": LSTM_LAYERS,
        "dropout": DROPOUT,
        "encoder_input_dim": ENCODER_INPUT_DIM,
        "decoder_input_dim": DECODER_INPUT_DIM,
        "forecast_horizon": FUTURE_STEPS,
        "best_val_mse": float(best_val_loss),
        "epoch": epoch,
    }, ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}")

    # ── Final validation inference ───────────────────────────────────────────
    print("\nRunning final validation inference...")
    _, final_preds, final_targets = validate(model, val_loader, criterion, device)
    final_preds = np.clip(final_preds, 0.0, 1.0)

    final_rmse = np.sqrt(np.mean((final_preds - final_targets) ** 2))
    final_mae = np.mean(np.abs(final_preds - final_targets))
    print(f"  Final val RMSE: {final_rmse:.4f}  MAE: {final_mae:.4f}")

    # Load timestamps for CSV export
    val_ds = CAFTimeSeriesDataset("val")
    times = val_ds.times

    # Save TFT predictions
    tft_pred_path = RESULTS_DIR / "tft_predictions.csv"
    # Also save Moirai-compatible copy for 06_evaluate.py
    phase2_results = Path(__file__).resolve().parent.parent / "phase2_finetuning" / "results"
    phase2_results.mkdir(parents=True, exist_ok=True)
    compat_path = phase2_results / "finetuned_predictions.csv"

    save_predictions_csv(
        final_preds, final_targets, times,
        save_path=tft_pred_path,
        also_save_compat=compat_path,
    )
    print(f"  TFT predictions → {tft_pred_path}")
    print(f"  Moirai-compat copy → {compat_path}")

    # ── Metrics JSON ─────────────────────────────────────────────────────────
    metrics = {
        "model": "TFT",
        "best_val_mse": float(best_val_loss),
        "best_val_rmse": float(best_val_loss ** 0.5),
        "final_val_rmse": float(final_rmse),
        "final_val_mae": float(final_mae),
        "parameters": model.count_parameters(),
        "epochs_trained": epoch,
        "training_time_s": round(total_time, 1),
        "smoke_test": args.smoke_test,
    }
    with open(RESULTS_DIR / "tft_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Plot training curves ─────────────────────────────────────────────────
    plot_training_curves(
        train_losses, val_losses,
        RESULTS_DIR / "tft_training_curves.png",
    )
    print(f"  Training curves → {RESULTS_DIR / 'tft_training_curves.png'}")

    # ── Variable importance ──────────────────────────────────────────────────
    print("\nComputing variable importance...")
    enc_importance, dec_importance = get_variable_importance(model, val_loader, device)
    plot_variable_importance(
        enc_importance, dec_importance,
        RESULTS_DIR / "tft_variable_importance.png",
    )
    print(f"  Variable importance → {RESULTS_DIR / 'tft_variable_importance.png'}")

    # Print importance ranking
    print("\n  Encoder (Past) Feature Importance:")
    enc_order = np.argsort(enc_importance)[::-1]
    for rank, idx in enumerate(enc_order, 1):
        print(f"    {rank}. {ENCODER_FEATURE_NAMES[idx]:15s} → {enc_importance[idx]:.4f}")

    print("\n  Decoder (Future) Feature Importance:")
    dec_order = np.argsort(dec_importance)[::-1]
    for rank, idx in enumerate(dec_order, 1):
        print(f"    {rank}. {DECODER_FEATURE_NAMES[idx]:15s} → {dec_importance[idx]:.4f}")

    print("\n✓ Done. To evaluate, run:")
    print("    python phase2_finetuning/06_evaluate.py")


if __name__ == "__main__":
    main()
