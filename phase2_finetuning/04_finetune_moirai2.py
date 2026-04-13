"""LoRA fine-tuning of Moirai 2.0 on CAF data using uni2ts + peft."""
import argparse
import sys, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    DATASET_DIR, CHECKPOINT_DIR,
    CONTEXT_LENGTH, PREDICTION_LENGTH, TARGET_DIM, FEAT_DIM,
    LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES, LORA_DROPOUT,
    FT_LR, FT_WEIGHT_DECAY, FT_MAX_EPOCHS, FT_PATIENCE,
    FT_BATCH_SIZE, FT_GRADIENT_CLIP,
)

MODEL_ID = "Salesforce/moirai-2.0-R-small"


def _median_quantile_index(module) -> int:
    quantiles = list(getattr(module, "quantile_levels", [0.5]))
    return quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2


def _extract_median_point_forecast(preds: torch.Tensor, median_idx: int) -> torch.Tensor:
    """Return the median forecast as a [batch, horizon] tensor."""
    if preds.ndim == 4:
        return preds[:, median_idx, :, 0]
    if preds.ndim == 3:
        return preds[:, median_idx, :]
    raise RuntimeError(f"Unexpected Moirai2 forecast shape: {tuple(preds.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Moirai 2.0 on CAF data.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one tiny epoch to verify the pipeline on a local machine.",
    )
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device. 'auto' prefers cuda, then cpu, and uses mps only if explicitly requested.",
    )
    args = parser.parse_args()

    max_epochs = 1 if args.smoke_test else FT_MAX_EPOCHS
    patience = 1 if args.smoke_test else FT_PATIENCE
    batch_size = min(4, FT_BATCH_SIZE) if args.smoke_test else FT_BATCH_SIZE
    max_train_windows = args.max_train_windows if args.max_train_windows is not None else (8 if args.smoke_test else None)
    max_val_windows = args.max_val_windows if args.max_val_windows is not None else (4 if args.smoke_test else None)
    if args.smoke_test:
        print("SMOKE TEST MODE: 1 epoch, limited train/val windows, no full-quality training.")

    # ---- 1. Load pre-trained Moirai 2.0 ----
    print(f"Loading {MODEL_ID}...")
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    try:
        config_path = hf_hub_download(MODEL_ID, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            model_kwargs = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Could not load model config for {MODEL_ID}. "
            "Make sure the checkpoint exists and is cached or reachable."
        ) from e

    if isinstance(model_kwargs.get("quantile_levels"), list):
        model_kwargs["quantile_levels"] = tuple(model_kwargs["quantile_levels"])

    module = Moirai2Module.from_pretrained(MODEL_ID, **model_kwargs)

    # ---- 2. Inject LoRA adapters ----
    print(f"Injecting LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    module = get_peft_model(module, lora_config)
    module.print_trainable_parameters()

    # ---- 3. Build forecast wrapper ----
    model = Moirai2Forecast(
        module=module,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        target_dim=TARGET_DIM,
        feat_dynamic_real_dim=FEAT_DIM,
        past_feat_dynamic_real_dim=0,
    )

    median_idx = _median_quantile_index(module)

    # ---- 4. Build dataloaders from .npy windows ----
    print("Building dataloaders from .npy windows...")
    from gluonts.dataset.common import ListDataset

    def npy_to_gluonts(split):
        X_past = np.load(DATASET_DIR / f"X_past_{split}.npy")
        X_future = np.load(DATASET_DIR / f"X_future_{split}.npy")
        times = np.load(DATASET_DIR / f"times_{split}.npy", allow_pickle=True)

        items = []
        for i in range(len(X_past)):
            target = X_past[i, :, 0].astype(np.float32)
            past_feats = X_past[i, :, 1:].T
            fut_feats = X_future[i].T
            dyn_feats = np.hstack([past_feats, fut_feats]).astype(np.float32)

            forecast_start = pd.Timestamp(times[i, 0])
            context_start = forecast_start - pd.Timedelta(hours=CONTEXT_LENGTH)

            items.append({
                "start": pd.Period(context_start, freq="h"),
                "target": target,
                "feat_dynamic_real": dyn_feats,
            })
        return ListDataset(items, freq="h")

    train_ds = npy_to_gluonts("train")
    val_ds = npy_to_gluonts("val")
    train_items = list(train_ds)
    val_items = list(val_ds)
    if max_train_windows is not None:
        train_items = train_items[:max_train_windows]
    if max_val_windows is not None:
        val_items = val_items[:max_val_windows]
    train_ds = ListDataset(train_items, freq="h")
    val_ds = ListDataset(val_items, freq="h")
    print(f"  Train: {len(train_items)} windows")
    print(f"  Val:   {len(val_items)} windows")

    # ---- 5. Training loop ----
    print("\nStarting LoRA fine-tuning...")
    print(f"  LR={FT_LR}  Epochs={max_epochs}  Patience={patience}  Batch={batch_size}")

    if args.smoke_test:
        device_name = "cpu"
    elif args.device != "auto":
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        device_name = args.device
    elif torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"
        if torch.backends.mps.is_available():
            print("  MPS detected, but Moirai fine-tuning defaults to CPU because required ops are not fully supported on MPS.")
            print("  If you want to try it anyway, rerun with --device mps and set PYTORCH_ENABLE_MPS_FALLBACK=1.")

    device = torch.device(device_name)
    print(f"  Device: {device}")

    module.to(device)
    module.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, module.parameters()),
        lr=FT_LR, weight_decay=FT_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    y_train = np.load(DATASET_DIR / "y_future_train.npy")
    y_val = np.load(DATASET_DIR / "y_future_val.npy")
    if max_train_windows is not None:
        y_train = y_train[:max_train_windows]
    if max_val_windows is not None:
        y_val = y_val[:max_val_windows]

    def point_forecast(item):
        past_target = torch.tensor(
            item["target"], dtype=torch.float32, device=device
        ).view(1, CONTEXT_LENGTH, TARGET_DIM)
        past_observed = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros((1, CONTEXT_LENGTH), dtype=torch.bool, device=device)
        feat_dynamic = torch.tensor(
            item["feat_dynamic_real"].T, dtype=torch.float32, device=device
        ).unsqueeze(0)
        observed_feat = torch.ones_like(feat_dynamic, dtype=torch.bool)

        preds = model(
            past_target,
            past_observed,
            past_is_pad,
            feat_dynamic_real=feat_dynamic,
            observed_feat_dynamic_real=observed_feat,
        )
        return _extract_median_point_forecast(preds, median_idx)

    for epoch in range(1, max_epochs + 1):
        module.train()
        train_losses = []

        for batch_idx, item in enumerate(train_ds):
            optimizer.zero_grad()
            try:
                pred = point_forecast(item).squeeze(0)
                y_true = torch.tensor(y_train[batch_idx], dtype=torch.float32, device=device)

                if not torch.isfinite(pred).all():
                    if batch_idx == 0:
                        print("    Warning: non-finite prediction encountered; skipping batch.")
                    continue
                if not torch.isfinite(y_true).all():
                    if batch_idx == 0:
                        print("    Warning: non-finite target encountered; skipping batch.")
                    continue

                loss = torch.nn.functional.mse_loss(pred[:len(y_true)], y_true)
                if not torch.isfinite(loss):
                    if batch_idx == 0:
                        print("    Warning: non-finite loss encountered; skipping batch.")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, module.parameters()),
                    FT_GRADIENT_CLIP,
                )
                optimizer.step()
                train_losses.append(loss.item())
            except Exception as e:
                if batch_idx == 0:
                    print(f"    Warning: {e}")
                continue

        scheduler.step()

        module.eval()
        val_losses = []
        with torch.no_grad():
            for val_idx, item in enumerate(val_ds):
                try:
                    pred = point_forecast(item).squeeze(0).detach().cpu().numpy()
                    y_true = y_val[val_idx]
                    val_loss = np.mean((pred[:len(y_true)] - y_true) ** 2)
                    if np.isfinite(val_loss):
                        val_losses.append(val_loss)
                    elif val_idx == 0:
                        print("    Warning: non-finite validation loss encountered; skipping batch.")
                except Exception:
                    continue

        avg_train = np.mean(train_losses) if train_losses else float("nan")
        avg_val = np.mean(val_losses) if val_losses else float("nan")

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_mse={avg_train:.5f}  val_mse={avg_val:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = {k: v.clone() for k, v in module.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ---- 6. Save LoRA adapter ----
    if best_state:
        module.load_state_dict(best_state)

    adapter_name = "moirai2_lora_adapter_smoke" if args.smoke_test else "moirai2_lora_adapter"
    adapter_path = CHECKPOINT_DIR / adapter_name
    module.save_pretrained(str(adapter_path))
    print(f"\n  LoRA adapter saved → {adapter_path}")

    lora_info = {
        "base_model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "target_modules": LORA_TARGET_MODULES,
        "best_val_mse": float(best_val_loss),
        "best_val_rmse": float(best_val_loss ** 0.5),
        "smoke_test": args.smoke_test,
        "model_family": "moirai2",
    }
    config_name = "lora_config_moirai2_smoke.json" if args.smoke_test else "lora_config_moirai2.json"
    with open(CHECKPOINT_DIR / config_name, "w") as f:
        json.dump(lora_info, f, indent=2)
    print(f"  Config saved → {config_name}")
    if np.isfinite(best_val_loss):
        print(f"  Best val RMSE: {best_val_loss**0.5:.4f}")
    else:
        print("  Best val RMSE: inf")


if __name__ == "__main__":
    main()
