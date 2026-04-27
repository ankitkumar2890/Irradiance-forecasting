"""Fine-tune a custom Moirai 1.1 encoder-decoder model for direct GHI."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    CHECKPOINT_DIR,
    CLUSTER_NAME,
    CONTEXT_LENGTH,
    DATASET_DIR,
    DAYLIGHT_ZENITH_DEG,
    FT_BATCH_SIZE,
    FT_GRADIENT_CLIP,
    FT_LR,
    FT_MAX_EPOCHS,
    FT_PATIENCE,
    FT_WEIGHT_DECAY,
    GHI_SCALE_FACTOR,
    MODEL_ID,
    NIGHT_ZENITH_DEG,
    PREDICTION_LENGTH,
)
from moirai1_1_seq2seq import Moirai1Seq2Seq  # noqa: E402


MOIRAI1_MODEL_ID = "Salesforce/moirai-1.1-R-small"


def _build_daylight_mask(future_zenith_deg: torch.Tensor) -> torch.Tensor:
    daylight_span = max(NIGHT_ZENITH_DEG - DAYLIGHT_ZENITH_DEG, 1e-6)
    mask = (NIGHT_ZENITH_DEG - future_zenith_deg.float()) / daylight_span
    return mask.clamp(min=0.0, max=1.0)


class WindowDataset(Dataset):
    def __init__(self, split: str, max_windows: int | None = None) -> None:
        self.X_past = np.load(DATASET_DIR / f"X_past_{split}.npy")
        self.X_future = np.load(DATASET_DIR / f"X_future_{split}.npy")
        self.y_future = np.load(DATASET_DIR / f"y_future_{split}.npy")
        if max_windows is not None:
            self.X_past = self.X_past[:max_windows]
            self.X_future = self.X_future[:max_windows]
            self.y_future = self.y_future[:max_windows]

    def __len__(self) -> int:
        return len(self.X_past)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x_past = torch.from_numpy(self.X_past[idx]).float()
        x_future = torch.from_numpy(self.X_future[idx]).float()
        target = torch.from_numpy(self.y_future[idx]).float()
        return {
            "past_target": x_past[:, :1] / GHI_SCALE_FACTOR,
            "past_dynamic_real": x_past[:, 1:],
            "future_dynamic_real": x_future,
            "target": target,
            "daylight_mask": _build_daylight_mask(x_future[:, 0]),
        }


def _resolve_device(device_arg: str, smoke_test: bool) -> torch.device:
    if smoke_test:
        return torch.device("cpu")
    if device_arg != "auto":
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device_arg == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    daylight_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted_error = (preds - target) * daylight_mask
    mse = weighted_error.square().sum() / daylight_mask.sum().clamp_min(1.0)
    mae = weighted_error.abs().sum() / daylight_mask.sum().clamp_min(1.0)
    return mse, mae


def _run_epoch(
    model: Moirai1Seq2Seq,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_mae = 0.0
    total_examples = 0

    for batch in loader:
        past_target = batch["past_target"].to(device)
        past_dynamic_real = batch["past_dynamic_real"].to(device)
        future_dynamic_real = batch["future_dynamic_real"].to(device)
        target = batch["target"].to(device)
        daylight_mask = batch["daylight_mask"].to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        preds = model(past_target, past_dynamic_real, future_dynamic_real)
        preds = torch.clamp(preds * daylight_mask, min=0.0)
        loss, mae = _compute_loss(preds, target, daylight_mask)

        if train_mode:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), FT_GRADIENT_CLIP)
            optimizer.step()

        batch_size = past_target.shape[0]
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        total_mae += float(mae.detach().cpu()) * batch_size

    return total_loss / max(total_examples, 1), total_mae / max(total_examples, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=FT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=FT_MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=FT_LR)
    parser.add_argument("--weight-decay", type=float, default=FT_WEIGHT_DECAY)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--model-id", default=MOIRAI1_MODEL_ID)
    parser.add_argument("--checkpoint-name", default=f"moirai1_seq2seq_{CLUSTER_NAME}.pt")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_download:
        args.local_files_only = False

    if args.smoke_test:
        args.epochs = 1
        args.batch_size = min(4, args.batch_size)
        args.max_train_windows = args.max_train_windows or 16
        args.max_val_windows = args.max_val_windows or 8

    device = _resolve_device(args.device, args.smoke_test)
    print(f"Training custom Moirai1 seq2seq model on {device}...")
    print(f"Backbone: {args.model_id}  patch_size={args.patch_size}")
    print(f"Moirai2 baseline in config remains: {MODEL_ID}")

    train_ds = WindowDataset("train", max_windows=args.max_train_windows)
    val_ds = WindowDataset("val", max_windows=args.max_val_windows)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    sample_future_dim = train_ds[0]["future_dynamic_real"].shape[-1]
    model = Moirai1Seq2Seq(
        future_feat_dim=sample_future_dim,
        model_id=args.model_id,
        patch_size=args.patch_size,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        local_files_only=args.local_files_only,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = math.inf
    best_epoch = -1
    patience_left = 1 if args.smoke_test else FT_PATIENCE
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint_name

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = _run_epoch(model, train_loader, device, optimizer=optimizer)
        with torch.no_grad():
            val_loss, val_mae = _run_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.5f} train_mae={train_mae:.5f} | "
            f"val_loss={val_loss:.5f} val_mae={val_mae:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_left = 1 if args.smoke_test else FT_PATIENCE
            payload = {
                "state_dict": model.state_dict(),
                "model_args": {
                    "future_feat_dim": sample_future_dim,
                    "model_id": args.model_id,
                    "patch_size": args.patch_size,
                    "decoder_layers": args.decoder_layers,
                    "decoder_heads": args.decoder_heads,
                    "dropout": args.dropout,
                    "freeze_backbone": args.freeze_backbone,
                    "local_files_only": args.local_files_only,
                },
                "metrics": {
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                },
            }
            torch.save(payload, checkpoint_path)
            with open(checkpoint_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cluster": CLUSTER_NAME,
                        "model_family": "moirai1_seq2seq",
                        "base_model": args.model_id,
                        "context_length": CONTEXT_LENGTH,
                        "prediction_length": PREDICTION_LENGTH,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                    },
                    f,
                    indent=2,
                )
            print(f"  Saved best checkpoint -> {checkpoint_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    print(f"Done. Best epoch={best_epoch}, val_loss={best_val_loss:.5f}")


if __name__ == "__main__":
    main()
