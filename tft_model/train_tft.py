"""
train_tft.py — Improved Training Pipeline for CAF TFT.

Usage
-----
    python tft_model/train_tft.py                       # full training, point forecast
    python tft_model/train_tft.py --loss quantile       # quantile (p10/p50/p90) mode
    python tft_model/train_tft.py --smoke-test          # 2-epoch sanity check
    python tft_model/train_tft.py --device cuda         # explicit device

Outputs
-------
    checkpoints/tft_best.pt                    — best checkpoint
    results/tft_predictions.csv               — validation predictions (long format)
    results/tft_metrics.json                  — RMSE, MAE, MBE
    results/tft_training_curves.png           — loss / LR curves
    results/tft_variable_importance.png       — encoder + decoder feature weights
    phase2_finetuning/results/finetuned_predictions_tft.csv  — compat copy

Key improvements over original train_tft.py
-------------------------------------------
1.  QUANTILE LOSS SUPPORT
    When --loss quantile is used (or LOSS_FN="quantile" in config.py),
    the model trains with the pinball loss across p10, p50, p90.
    Metrics are computed from p50 so they remain comparable with MSE runs.

2.  WARMUP + COSINE ANNEALING LR SCHEDULE
    Original used CosineAnnealingLR starting at LEARNING_RATE immediately.
    We add a linear warmup for the first WARMUP_EPOCHS epochs, which
    prevents very large gradient updates in the first few steps when the
    model weights are still randomly initialised. After warmup, cosine
    annealing brings LR to eta_min over the remaining epochs.

3.  GRADIENT NORM TRACKING
    We log the raw gradient norm before clipping each epoch so you can
    see whether the model is experiencing gradient explosions or vanishing.

4.  EXPONENTIAL MOVING AVERAGE (EMA) OF WEIGHTS
    An EMA copy of the model weights is maintained during training.
    The EMA model is what is saved and used for inference. EMA smooths out
    checkpoint noise and typically yields 0.5–1% RMSE improvement without
    any change to the model architecture or data.

5.  PER-LEADTIME METRICS
    Validation metrics (RMSE, MAE) are broken down by forecast lead time
    (h+1 through h+24) so you can see where the model degrades.

6.  MBE (MEAN BIAS ERROR)
    Added as a metric: systematic over/under-prediction is important for
    solar operators. MBE = mean(pred - true); positive → over-forecast.

7.  IMPROVED CHECKPOINT FORMAT
    The checkpoint now includes model hyperparameters, EMA flag, and the
    full feature name lists so the checkpoint is self-contained for later
    inference without needing config.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CHECKPOINT_DIR, RESULTS_DIR,
    PAST_STEPS, FUTURE_STEPS,
    ENCODER_INPUT_DIM, DECODER_INPUT_DIM,
    HIDDEN_SIZE, NUM_ATTENTION_HEADS, LSTM_LAYERS, DROPOUT,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, PATIENCE, GRADIENT_CLIP,
    LOSS_FN, SEED,
)
from dataset import get_dataloader, CAFTimeSeriesDataset
from model import TemporalFusionTransformer, QuantileLoss


# ── Feature registry (must match config.py ordering exactly) ─────────────────
ENCODER_FEATURE_NAMES: List[str] = [
    "CAF",          # index 0 — past-only target
    "clear_sky_ghi",# index 1
    "tcc",          # index 2
    "lcc",          # index 3
    "mcc",          # index 4
    "hcc",          # index 5
    "u10",          # index 6
    "v10",          # index 7
    "zenith_angle", # index 8  ← used for daytime mask
    "hour_sin",     # index 9
    "hour_cos",     # index 10
    "doy_sin",      # index 11
    "doy_cos",      # index 12
]
DECODER_FEATURE_NAMES: List[str] = [
    "clear_sky_ghi",# index 0
    "tcc",          # index 1
    "lcc",          # index 2
    "mcc",          # index 3
    "hcc",          # index 4
    "u10",          # index 5
    "v10",          # index 6
    "zenith_angle", # index 7  ← used for daytime mask
    "hour_sin",     # index 8
    "hour_cos",     # index 9
    "doy_sin",      # index 10
    "doy_cos",      # index 11
]
# Zenith angle column indices within encoder/decoder tensors
_ENC_ZENITH_IDX = ENCODER_FEATURE_NAMES.index("zenith_angle")  # 8
_DEC_ZENITH_IDX = DECODER_FEATURE_NAMES.index("zenith_angle")  # 7

QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)
WARMUP_EPOCHS: int = 5          # linear warmup before cosine annealing
EMA_DECAY: float = 0.995        # exponential moving average decay


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility + Device
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(args: argparse.Namespace) -> torch.device:
    if args.smoke_test:
        return torch.device("cpu")
    if args.device != "auto":
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Loss Selection
# ══════════════════════════════════════════════════════════════════════════════

def build_criterion(loss_name: str) -> Tuple[nn.Module, bool]:
    """
    Returns (criterion, use_quantiles).

    use_quantiles=True  → model outputs (B, T, Q), loss is pinball
    use_quantiles=False → model outputs (B, T),    loss is MSE or Huber
    """
    name = loss_name.lower()
    if name == "quantile":
        return QuantileLoss(QUANTILES), True
    if name == "huber":
        return nn.HuberLoss(delta=0.1), False
    # default: MSE
    return nn.MSELoss(), False


# ══════════════════════════════════════════════════════════════════════════════
# Exponential Moving Average
# ══════════════════════════════════════════════════════════════════════════════

class EMA:
    """
    Maintains an exponential moving average of model parameters.

    After each optimiser step, call ema.update(model).
    For evaluation, call ema.apply_shadow(model) to copy EMA weights
    into model, then ema.restore(model) to put training weights back.

    EMA is known to reduce variance in checkpoints and often yields
    slightly lower validation error than the raw training weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Copy EMA weights into model (for evaluation / checkpointing)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore training weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ══════════════════════════════════════════════════════════════════════════════
# LR Schedule: linear warmup + cosine annealing
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup for `warmup_epochs`, then cosine decay to `eta_min`.

    The base LR (set in optimizer) is used as the peak LR after warmup.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # Cosine from 1.0 → eta_min/base_lr over remaining epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale so that the minimum is eta_min relative to base LR
        base_lr = optimizer.param_groups[0]["initial_lr"]
        return max(eta_min / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# Training / Validation
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    use_quantiles: bool,
    ema: Optional[EMA] = None,
) -> Tuple[float, float]:
    """
    Returns (avg_loss, avg_grad_norm).
    """
    model.train()
    total_loss = 0.0
    total_gnorm = 0.0
    n_batches = 0

    for enc, dec, tgt in loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

        optimizer.zero_grad()
        preds, _, _, _ = model(enc, dec)

        if use_quantiles:
            # preds: (B, T, Q),  tgt: (B, T)
            loss = criterion(preds, tgt)
        else:
            loss = criterion(preds, tgt)

        if not torch.isfinite(loss):
            print("  [WARN] Non-finite loss detected, skipping batch.")
            continue

        loss.backward()

        # Track gradient norm before clipping
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        total_gnorm += gnorm.item()

        optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_gnorm = total_gnorm / max(n_batches, 1)
    return avg_loss, avg_gnorm


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_quantiles: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Returns (avg_loss, all_preds, all_targets).

    all_preds: (N, 24)     — point forecast or p50 for quantile mode
    all_targets: (N, 24)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []

    for enc, dec, tgt in loader:
        enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
        preds, _, _, _ = model(enc, dec)

        if use_quantiles:
            loss = criterion(preds, tgt)
        else:
            loss = criterion(preds, tgt)

        if torch.isfinite(loss):
            total_loss += loss.item()
            n_batches += 1

        # For metrics, always use point prediction
        if use_quantiles:
            mid = len(QUANTILES) // 2
            point_preds = preds[..., mid].clamp(0.0, 1.0)
        else:
            point_preds = preds.clamp(0.0, 1.0)

        all_preds.append(point_preds.cpu().numpy())
        all_targets.append(tgt.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    preds_arr = np.concatenate(all_preds, axis=0)      # (N, 24)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, 24)
    return avg_loss, preds_arr, targets_arr


@torch.no_grad()
def get_variable_importance(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average VSN variable weights across the validation set.

    Returns
    -------
    enc_importance : (encoder_input_dim,)
    dec_importance : (decoder_input_dim,)
    """
    model.eval()
    enc_acc = None
    dec_acc = None
    n_samples = 0

    for enc, dec, _ in loader:
        enc, dec = enc.to(device), dec.to(device)
        _, enc_w, dec_w, _ = model(enc, dec)
        # Average over time: (B, T, F) → (B, F)
        enc_mean = enc_w.mean(dim=1).sum(dim=0).cpu().numpy()
        dec_mean = dec_w.mean(dim=1).sum(dim=0).cpu().numpy()

        if enc_acc is None:
            enc_acc = enc_mean
            dec_acc = dec_mean
        else:
            enc_acc += enc_mean
            dec_acc += dec_mean
        n_samples += enc.shape[0]

    return enc_acc / n_samples, dec_acc / n_samples


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    preds, targets: (N, 24).

    Returns overall RMSE, MAE, MBE, and per-lead-time RMSE.
    """
    err = preds - targets
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    mbe  = float(np.mean(err))                       # + → over-forecast

    # Per-lead-time (h+1 … h+24)
    lt_rmse = [
        float(np.sqrt(np.mean((preds[:, h] - targets[:, h]) ** 2)))
        for h in range(preds.shape[1])
    ]
    return {
        "rmse": rmse,
        "mae": mae,
        "mbe": mbe,
        "per_lead_rmse": lt_rmse,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    gnorms: List[float],
    lrs: List[float],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(train_losses) + 1)

    ax = axes[0]
    ax.plot(epochs, train_losses, "b-", label="Train", linewidth=1.5)
    ax.plot(epochs, val_losses,   "r-", label="Val",   linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss Curves")
    ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, gnorms, "g-", linewidth=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Grad Norm"); ax.set_title("Gradient Norm")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, lrs, "purple", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("Learning Rate")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_variable_importance(
    enc_weights: np.ndarray,
    dec_weights: np.ndarray,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, weights, names, title, color in [
        (axes[0], enc_weights, ENCODER_FEATURE_NAMES, "Encoder (Past)", "#4c72b0"),
        (axes[1], dec_weights, DECODER_FEATURE_NAMES, "Decoder (Future)", "#dd8452"),
    ]:
        order = np.argsort(weights)
        ax.barh(
            [names[i] for i in order],
            weights[order],
            color=color, edgecolor="white",
        )
        ax.set_xlabel("Mean VSN Weight"); ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("TFT Variable Importance", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_lead_time_rmse(lt_rmse: List[float], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    hours = list(range(1, len(lt_rmse) + 1))
    ax.bar(hours, lt_rmse, color="#4c72b0", edgecolor="white", width=0.7)
    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("RMSE")
    ax.set_title("Validation RMSE by Forecast Lead Time")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CSV Export
# ══════════════════════════════════════════════════════════════════════════════

def save_predictions_csv(
    preds: np.ndarray,
    targets: np.ndarray,
    times,
    station_ids,
    save_path: Path,
    also_save_compat: Optional[Path] = None,
) -> pd.DataFrame:
    rows = []
    for i in range(preds.shape[0]):
        for h in range(FUTURE_STEPS):
            ts = pd.Timestamp(times[i, h])
            rows.append({
                "station_id": str(station_ids[i]) if station_ids is not None else "unknown",
                "datetime":   ts,
                "hour":       ts.hour,
                "lead_time_h": h + 1,
                "forecast_start": pd.Timestamp(times[i, 0]),
                "CAF_true":   float(targets[i, h]),
                "CAF_pred":   float(np.clip(preds[i, h], 0.0, 1.0)),
            })
    if not rows:
        raise ValueError("No prediction rows generated.")
    sort_cols = ["station_id", "datetime", "lead_time_h"]
    df = pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(save_path, index=False)
    if also_save_compat:
        df.to_csv(also_save_compat, index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

import math  # needed here for the LR lambda


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TFT on CAF data.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="2 epochs, small batch — sanity check only.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--loss", choices=["mse", "huber", "quantile"],
        default=LOSS_FN,
        help="Loss function. 'quantile' enables p10/p50/p90 output.",
    )
    parser.add_argument("--no-ema", action="store_true",
                        help="Disable EMA weight averaging.")
    args = parser.parse_args()

    set_seed(SEED)
    device = select_device(args)
    use_ema = not args.no_ema
    print(f"Device: {device}  |  Loss: {args.loss}  |  EMA: {use_ema}")

    # ── Override params for smoke test ────────────────────────────────────────
    max_epochs  = 2           if args.smoke_test else MAX_EPOCHS
    patience    = 1           if args.smoke_test else PATIENCE
    batch_size  = min(8, BATCH_SIZE) if args.smoke_test else BATCH_SIZE
    warmup      = 0           if args.smoke_test else WARMUP_EPOCHS

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_loader = get_dataloader("train", batch_size=batch_size, shuffle=True)
    val_loader   = get_dataloader("val",   batch_size=batch_size, shuffle=False)
    print(f"  Train: {len(train_loader.dataset):,} windows ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset):,} windows ({len(val_loader)} batches)")

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion, use_quantiles = build_criterion(args.loss)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TemporalFusionTransformer(
        encoder_input_dim=ENCODER_INPUT_DIM,
        decoder_input_dim=DECODER_INPUT_DIM,
        hidden_dim=HIDDEN_SIZE,
        num_heads=NUM_ATTENTION_HEADS,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        forecast_horizon=FUTURE_STEPS,
        past_steps=PAST_STEPS,
        use_quantiles=use_quantiles,
        quantiles=QUANTILES,
    ).to(device)

    print(f"\nTFT parameters: {model.count_parameters():,}")

    # ── Optimiser + LR Schedule ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # Store initial_lr for the LR lambda
    for pg in optimizer.param_groups:
        pg["initial_lr"] = LEARNING_RATE

    scheduler = build_scheduler(optimizer, warmup, max_epochs, eta_min=1e-6)

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = EMA(model, decay=EMA_DECAY) if use_ema else None

    # ── Training Loop ────────────────────────────────────────────────────────
    print(
        f"\nStarting training: epochs={max_epochs}  patience={patience}  "
        f"batch={batch_size}  warmup={warmup}\n"
    )

    best_val_loss   = float("inf")
    patience_ctr    = 0
    best_state      = None
    train_losses, val_losses = [], []
    gnorm_history, lr_history = [], []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()

        train_loss, avg_gnorm = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            GRADIENT_CLIP, use_quantiles, ema,
        )
        scheduler.step()

        # Evaluate using EMA weights if available
        if ema is not None:
            ema.apply_shadow(model)

        val_loss, val_preds, val_targets = validate(
            model, val_loader, criterion, device, use_quantiles,
        )

        if ema is not None:
            ema.restore(model)

        # ── Record ──────────────────────────────────────────────────────────
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        gnorm_history.append(avg_gnorm)
        lr_history.append(optimizer.param_groups[0]["lr"])

        metrics = compute_metrics(val_preds, val_targets)
        elapsed = time.time() - ep_t0

        if epoch <= 5 or epoch % 5 == 0 or epoch == max_epochs:
            print(
                f"  Ep {epoch:3d}/{max_epochs} | "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
                f"MBE={metrics['mbe']:+.4f}  "
                f"gnorm={avg_gnorm:.3f}  "
                f"lr={lr_history[-1]:.2e}  ({elapsed:.1f}s)"
            )

        # ── Early Stopping ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            # Save EMA state if available, otherwise training state
            if ema is not None:
                ema.apply_shadow(model)
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                ema.restore(model)
            else:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    total_time = time.time() - t0
    print(
        f"\nTraining done in {total_time:.1f}s ({total_time/60:.1f} min)  "
        f"Best val loss: {best_val_loss:.5f}"
    )

    # ── Restore best weights ──────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    # ── Save Checkpoint ───────────────────────────────────────────────────────
    ckpt_name = "tft_best_smoke.pt" if args.smoke_test else "tft_best.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            # Hyperparameters (self-contained for future inference)
            "encoder_input_dim":  ENCODER_INPUT_DIM,
            "decoder_input_dim":  DECODER_INPUT_DIM,
            "hidden_size":        HIDDEN_SIZE,
            "num_attention_heads":NUM_ATTENTION_HEADS,
            "lstm_layers":        LSTM_LAYERS,
            "dropout":            DROPOUT,
            "forecast_horizon":   FUTURE_STEPS,
            "past_steps":         PAST_STEPS,
            "use_quantiles":      use_quantiles,
            "quantiles":          QUANTILES,
            "use_ema":            use_ema,
            # Feature names for interpretability / future inference
            "encoder_feature_names": ENCODER_FEATURE_NAMES,
            "decoder_feature_names": DECODER_FEATURE_NAMES,
            # Training metadata
            "best_val_loss":      float(best_val_loss),
            "loss_fn":            args.loss,
            "epochs_trained":     epoch,
            "smoke_test":         args.smoke_test,
        },
        ckpt_path,
    )
    print(f"  Checkpoint → {ckpt_path}")

    # ── Final Inference ───────────────────────────────────────────────────────
    print("\nRunning final validation inference...")
    _, final_preds, final_targets = validate(
        model, val_loader, criterion, device, use_quantiles,
    )
    final_metrics = compute_metrics(final_preds, final_targets)
    print(
        f"  RMSE={final_metrics['rmse']:.4f}  "
        f"MAE={final_metrics['mae']:.4f}  "
        f"MBE={final_metrics['mbe']:+.4f}"
    )

    # ── Load timestamps ───────────────────────────────────────────────────────
    val_ds = CAFTimeSeriesDataset("val")
    times = val_ds.times
    station_ids = val_ds.station_ids

    # ── Save Prediction CSV ───────────────────────────────────────────────────
    tft_pred_path = RESULTS_DIR / "tft_predictions.csv"
    phase2_results = Path(__file__).resolve().parent.parent / "phase2_finetuning" / "results"
    phase2_results.mkdir(parents=True, exist_ok=True)
    compat_path = phase2_results / "finetuned_predictions_tft.csv"

    save_predictions_csv(
        final_preds, final_targets, times, station_ids,
        save_path=tft_pred_path,
        also_save_compat=compat_path,
    )
    print(f"  Predictions → {tft_pred_path}")
    print(f"  Compat copy → {compat_path}")

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    metrics_out = {
        "model": "TFT",
        "loss_fn": args.loss,
        "use_ema": use_ema,
        "best_val_loss": float(best_val_loss),
        "final_val_rmse": final_metrics["rmse"],
        "final_val_mae": final_metrics["mae"],
        "final_val_mbe": final_metrics["mbe"],
        "per_lead_rmse": final_metrics["per_lead_rmse"],
        "parameters": model.count_parameters(),
        "epochs_trained": epoch,
        "training_time_s": round(total_time, 1),
        "smoke_test": args.smoke_test,
    }
    with open(RESULTS_DIR / "tft_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(
        train_losses, val_losses, gnorm_history, lr_history,
        RESULTS_DIR / "tft_training_curves.png",
    )
    print(f"  Training curves → {RESULTS_DIR / 'tft_training_curves.png'}")

    plot_lead_time_rmse(
        final_metrics["per_lead_rmse"],
        RESULTS_DIR / "tft_lead_time_rmse.png",
    )
    print(f"  Lead-time RMSE → {RESULTS_DIR / 'tft_lead_time_rmse.png'}")

    print("\nComputing variable importance...")
    enc_imp, dec_imp = get_variable_importance(model, val_loader, device)
    plot_variable_importance(enc_imp, dec_imp, RESULTS_DIR / "tft_variable_importance.png")
    print(f"  Variable importance → {RESULTS_DIR / 'tft_variable_importance.png'}")

    print("\n  Encoder Feature Importance (ranked):")
    for rank, idx in enumerate(np.argsort(enc_imp)[::-1], 1):
        print(f"    {rank:2d}. {ENCODER_FEATURE_NAMES[idx]:15s}  {enc_imp[idx]:.4f}")
    print("\n  Decoder Feature Importance (ranked):")
    for rank, idx in enumerate(np.argsort(dec_imp)[::-1], 1):
        print(f"    {rank:2d}. {DECODER_FEATURE_NAMES[idx]:15s}  {dec_imp[idx]:.4f}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
