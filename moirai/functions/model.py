"""Moirai model code: load + LoRA + train + inference + evaluate.

Three methods live in this file. The heavy lifting (loading the base
Moirai module, injecting LoRA, the training loop, single-window forecast
helper, adapter persistence, inference loop) is shared. Each method
then has its own thin orchestrator (``finetune_method{N}`` and the
inference / report functions in ``functions/results.py``) that wires
the shared helpers to the method-specific paths and config.

Training-loop notes
-------------------
The training loop now uses **gradient accumulation** over the configured
``ft_batch_size`` (so an "effective batch" really is 32 windows by
default rather than 1). Train pairs are ``(item, y)`` tuples and are
**shuffled together every epoch** so the index-coupling between the
GluonTS items and the ``y_future_*.npy`` arrays cannot drift.

Method-3 specifics
------------------
Method 3 (direct GHI) divides the target track and the y labels by
``GHI_SCALE_FACTOR`` before training, and the matching un-scaling lives
in ``functions/results._run_inference_direct_ghi``. The scaling is
applied here in ``finetune_method3`` (via ``target_scale``) so the
on-disk windows stay in physical units (W/m^2).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =====================================================================
# SHARED: Moirai loading + LoRA injection
# =====================================================================

# ---- Moirai 1.x ---------------------------------------------------------

def load_moirai_module(model_id):
    """Load a pretrained Moirai 1.x MoiraiModule from Hugging Face."""
    from huggingface_hub import hf_hub_download
    from hydra.utils import instantiate
    from uni2ts.model.moirai import MoiraiModule

    print(f"Loading {model_id} (Moirai 1.x)...")
    try:
        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            model_kwargs = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Could not load model config for {model_id}. "
            "Make sure the checkpoint exists and is cached or reachable."
        ) from e

    if isinstance(model_kwargs.get("distr_output"), dict):
        model_kwargs["distr_output"] = instantiate(
            model_kwargs["distr_output"], _convert_="all"
        )
    if isinstance(model_kwargs.get("patch_sizes"), list):
        model_kwargs["patch_sizes"] = tuple(model_kwargs["patch_sizes"])

    return MoiraiModule.from_pretrained(model_id, **model_kwargs)


# ---- Moirai 2.0 ---------------------------------------------------------

def load_moirai2_module(model_id):
    """Load a pretrained Moirai 2.0 Moirai2Module from Hugging Face."""
    from huggingface_hub import hf_hub_download
    from uni2ts.model.moirai2 import Moirai2Module

    print(f"Loading {model_id} (Moirai 2.0)...")
    try:
        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            model_kwargs = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Could not load model config for {model_id}. "
            "Make sure the checkpoint exists and is cached or reachable."
        ) from e

    if isinstance(model_kwargs.get("quantile_levels"), list):
        model_kwargs["quantile_levels"] = tuple(model_kwargs["quantile_levels"])

    return Moirai2Module.from_pretrained(model_id, **model_kwargs)


def median_quantile_index(module) -> int:
    """Return the index of the 0.5 quantile in a Moirai 2.0 module."""
    quantiles = list(getattr(module, "quantile_levels", [0.5]))
    return quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2


def _extract_quantile_point_forecast(preds, q_idx: int):
    """Reduce Moirai 2.0 quantile preds to a [batch, horizon] tensor at quantile index ``q_idx``.

    (Renamed from ``_extract_median_point_forecast`` because the function
    is used for any quantile index, not only the median.)
    """
    if preds.ndim == 4:
        return preds[:, q_idx, :, 0]
    if preds.ndim == 3:
        return preds[:, q_idx, :]
    raise RuntimeError(f"Unexpected Moirai2 forecast shape: {tuple(preds.shape)}")


# Backwards-compatible alias (still imported elsewhere in the codebase).
_extract_median_point_forecast = _extract_quantile_point_forecast


# ---- Variant dispatcher -------------------------------------------------

def load_module_for_variant(model_id, model_variant: str):
    """Dispatch base-module loading by ``model_variant`` ('moirai1' or 'moirai2')."""
    if model_variant == "moirai1":
        return load_moirai_module(model_id)
    if model_variant == "moirai2":
        return load_moirai2_module(model_id)
    raise ValueError(f"Unknown model_variant: {model_variant!r}")


def inject_lora(module, *, lora_rank, lora_alpha, lora_target_modules, lora_dropout):
    """Wrap a Moirai module with PEFT LoRA adapters."""
    from peft import LoraConfig, get_peft_model

    print(f"Injecting LoRA (rank={lora_rank}, alpha={lora_alpha})...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=list(lora_target_modules),
        lora_dropout=lora_dropout,
        bias="none",
    )
    module = get_peft_model(module, lora_config)
    module.print_trainable_parameters()
    return module


def build_forecast_model(
    module,
    *,
    prediction_length,
    context_length,
    target_dim,
    feat_dim,
    patch_size=16,
    past_feat_dynamic_real_dim=0,
):
    """Wrap a (LoRA-injected) Moirai 1.x module in MoiraiForecast."""
    from uni2ts.model.moirai import MoiraiForecast

    return MoiraiForecast(
        module=module,
        prediction_length=prediction_length,
        context_length=context_length,
        target_dim=target_dim,
        feat_dynamic_real_dim=feat_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        patch_size=patch_size,
    )


def build_forecast2_model(
    module,
    *,
    prediction_length,
    context_length,
    target_dim,
    feat_dim,
    past_feat_dynamic_real_dim=0,
):
    """Wrap a (LoRA-injected) Moirai 2.0 module in Moirai2Forecast."""
    from uni2ts.model.moirai2 import Moirai2Forecast

    return Moirai2Forecast(
        module=module,
        prediction_length=prediction_length,
        context_length=context_length,
        target_dim=target_dim,
        feat_dynamic_real_dim=feat_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
    )


def build_forecast_for_variant(
    module,
    *,
    model_variant: str,
    prediction_length,
    context_length,
    target_dim,
    feat_dim,
    patch_size=16,
    past_feat_dynamic_real_dim=0,
):
    """Dispatch forecast-wrapper construction by ``model_variant``."""
    if model_variant == "moirai1":
        return build_forecast_model(
            module,
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=target_dim, feat_dim=feat_dim,
            patch_size=patch_size,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        )
    if model_variant == "moirai2":
        return build_forecast2_model(
            module,
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=target_dim, feat_dim=feat_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        )
    raise ValueError(f"Unknown model_variant: {model_variant!r}")


def select_device(device_arg, smoke_test=False):
    """Resolve a torch device based on user request and availability."""
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
        print(
            "  MPS detected, but Moirai fine-tuning defaults to CPU because "
            "required ops are not fully supported on MPS."
        )
        print(
            "  If you want to try it anyway, rerun with --device mps and set "
            "PYTORCH_ENABLE_MPS_FALLBACK=1."
        )
    return torch.device("cpu")


# =====================================================================
# SHARED: .npy windows -> GluonTS dataset + single-window forecast
# =====================================================================

def npy_to_gluonts(
    split,
    *,
    dataset_dir,
    context_length,
    dyn_indices=None,
):
    """Convert saved sliding-window .npy files into a GluonTS ListDataset.

    GluonTS's ``feat_dynamic_real`` requires the same number of features over
    the past *and* the future segments. For Methods 1/2 we have ``PAST_FEATURES
    == [target, *FUTURE_FEATURES]``, so the past-feature columns line up with
    the future columns one-to-one (after dropping the target at column 0).

    If a method deliberately includes past-only extras, the caller must pass
    ``dyn_indices`` — a list of indices into
    ``X_past[..., 1:]`` (i.e. *after* the target column has been removed) that
    selects exactly the past-feature columns whose names match
    ``FUTURE_FEATURES``, in the same order. Anything else in ``X_past`` is
    simply not fed to the model as a dynamic covariate (matching the inference
    behaviour in ``results.py``).

    When ``dyn_indices`` is ``None`` we use every past-feature column, which
    is the right thing for Methods 1/2 and is fast-path back-compatible.
    """
    from gluonts.dataset.common import ListDataset

    dataset_dir = Path(dataset_dir)
    X_past = np.load(dataset_dir / f"X_past_{split}.npy")
    X_future = np.load(dataset_dir / f"X_future_{split}.npy")
    times = np.load(dataset_dir / f"times_{split}.npy", allow_pickle=True)

    items = []
    for i in range(len(X_past)):
        target = X_past[i, :, 0].astype(np.float32)

        if dyn_indices is None:
            past_feats = X_past[i, :, 1:].T  # (n_past_features, past_hours)
        else:
            # Selecting columns from the post-target slice — dyn_indices are
            # already expressed relative to ``X_past[..., 1:]``.
            past_feats = X_past[i, :, 1:][:, dyn_indices].T

        fut_feats = X_future[i].T            # (n_future_features, future_hours)

        if past_feats.shape[0] != fut_feats.shape[0]:
            raise ValueError(
                "feat_dynamic_real shape mismatch when building GluonTS items: "
                f"past has {past_feats.shape[0]} features, future has "
                f"{fut_feats.shape[0]}. This usually means the calling "
                "method did not pass dyn_indices to align past-only "
                "features with FUTURE_FEATURES."
            )

        dyn_feats = np.hstack([past_feats, fut_feats]).astype(np.float32)

        forecast_start = pd.Timestamp(times[i, 0])
        context_start = forecast_start - pd.Timedelta(hours=context_length)

        items.append(
            {
                "start": pd.Period(context_start, freq="h"),
                "target": target,
                "feat_dynamic_real": dyn_feats,
            }
        )
    return ListDataset(items, freq="h")


def _compute_dyn_indices(past_features, future_features):
    """Map FUTURE_FEATURES to their positions inside ``PAST_FEATURES[1:]``.

    Returns ``None`` when the past-feature list (with the target stripped) is
    already exactly ``future_features`` in order — this is the common case
    (Methods 1 and 2) and lets ``npy_to_gluonts`` take its fast path.
    """
    if not past_features or not future_features:
        return None
    past_no_target = list(past_features[1:])
    if past_no_target == list(future_features):
        return None
    try:
        return [past_no_target.index(f) for f in future_features]
    except ValueError as exc:
        raise ValueError(
            "FUTURE_FEATURES must all appear inside PAST_FEATURES[1:] for "
            "GluonTS feat_dynamic_real alignment. Offending name: "
            f"{exc}. PAST_FEATURES[1:] = {past_no_target}; "
            f"FUTURE_FEATURES = {list(future_features)}"
        ) from exc


def forecast_mean(
    model,
    target,
    feat_dynamic_real,
    *,
    device,
    context_length,
    target_dim,
    patch_size=16,
):
    """Moirai 1.x single-window forward pass; returns the mean forecast tensor."""
    past_target = torch.tensor(
        target, dtype=torch.float32, device=device
    ).view(1, context_length, target_dim)
    past_observed = torch.ones_like(past_target, dtype=torch.bool)
    past_is_pad = torch.zeros((1, context_length), dtype=torch.bool, device=device)
    feat_dynamic = torch.tensor(
        feat_dynamic_real.T, dtype=torch.float32, device=device
    ).unsqueeze(0)
    observed_feat = torch.ones_like(feat_dynamic, dtype=torch.bool)

    distr = model._get_distr(
        patch_size,
        past_target,
        past_observed,
        past_is_pad,
        feat_dynamic,
        observed_feat,
    )
    formatted = model._format_preds(patch_size, distr.mean.unsqueeze(0), target_dim)
    return formatted[:, 0, :]


def forecast_quantiles_v1(
    model,
    target,
    feat_dynamic_real,
    *,
    device,
    context_length,
    target_dim,
    quantiles: list[float],
    patch_size: int = 16,
    num_samples: int = 200,
):
    """Moirai 1.x quantile forecast for a single window.

    Strategy:
      * First, try ``distr.icdf(q)`` directly. PyTorch's StudentT, Normal,
        etc. expose an analytical inverse CDF, which is fast and exact.
      * If the underlying distribution does not support ``icdf`` (e.g. a
        ``MixtureSameFamily``), fall back to Monte-Carlo sampling and
        empirical quantiles via ``torch.quantile`` over ``num_samples``
        sample paths.

    Returns ``{label: numpy [horizon]}`` with ``label = "p10"``,
    ``"p50"``, ``"p90"`` etc.
    """
    past_target = torch.tensor(
        target, dtype=torch.float32, device=device
    ).view(1, context_length, target_dim)
    past_observed = torch.ones_like(past_target, dtype=torch.bool)
    past_is_pad = torch.zeros((1, context_length), dtype=torch.bool, device=device)
    feat_dynamic = torch.tensor(
        feat_dynamic_real.T, dtype=torch.float32, device=device
    ).unsqueeze(0)
    observed_feat = torch.ones_like(feat_dynamic, dtype=torch.bool)

    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        distr = model._get_distr(
            patch_size,
            past_target,
            past_observed,
            past_is_pad,
            feat_dynamic,
            observed_feat,
        )

        analytical_ok = True
        quantile_tensors: dict[float, torch.Tensor] = {}
        try:
            for q in quantiles:
                qt = distr.icdf(torch.tensor(float(q), device=device))
                quantile_tensors[q] = qt
        except (NotImplementedError, AttributeError, Exception):
            analytical_ok = False

        if not analytical_ok:
            samples = distr.sample(torch.Size([num_samples]))
            for q in quantiles:
                qt = torch.quantile(samples, float(q), dim=0)
                quantile_tensors[q] = qt

        for q, qt in quantile_tensors.items():
            formatted = model._format_preds(patch_size, qt.unsqueeze(0), target_dim)
            path = formatted[:, 0, :].squeeze(0).detach().cpu().numpy()
            out[f"p{int(round(q * 100))}"] = path

    return out


def forecast_median_v2(
    model,
    target,
    feat_dynamic_real,
    *,
    device,
    context_length,
    target_dim,
    median_idx: int,
):
    """Moirai 2.0 single-window forward pass; returns the median quantile tensor."""
    past_target = torch.tensor(
        target, dtype=torch.float32, device=device
    ).view(1, context_length, target_dim)
    past_observed = torch.ones_like(past_target, dtype=torch.bool)
    past_is_pad = torch.zeros((1, context_length), dtype=torch.bool, device=device)
    feat_dynamic = torch.tensor(
        feat_dynamic_real.T, dtype=torch.float32, device=device
    ).unsqueeze(0)
    observed_feat = torch.ones_like(feat_dynamic, dtype=torch.bool)

    preds = model(
        past_target,
        past_observed,
        past_is_pad,
        feat_dynamic_real=feat_dynamic,
        observed_feat_dynamic_real=observed_feat,
    )
    return _extract_quantile_point_forecast(preds, median_idx)


def forecast_point(
    model,
    target,
    feat_dynamic_real,
    *,
    device,
    context_length,
    target_dim,
    model_variant: str,
    patch_size: int = 16,
    median_idx=None,
):
    """Variant-aware single-window point forecast. Returns a [1, horizon] tensor."""
    if model_variant == "moirai1":
        return forecast_mean(
            model, target, feat_dynamic_real,
            device=device, context_length=context_length,
            target_dim=target_dim, patch_size=patch_size,
        )
    if model_variant == "moirai2":
        if median_idx is None:
            raise ValueError(
                "median_idx is required for Moirai 2.0 single-window inference."
            )
        return forecast_median_v2(
            model, target, feat_dynamic_real,
            device=device, context_length=context_length,
            target_dim=target_dim, median_idx=median_idx,
        )
    raise ValueError(f"Unknown model_variant: {model_variant!r}")


# =====================================================================
# SHARED: training loop + adapter persistence
# =====================================================================

def train_lora(
    *,
    model,
    module,
    train_pairs,
    val_pairs,
    device,
    max_epochs,
    patience,
    ft_lr,
    ft_weight_decay,
    ft_gradient_clip,
    ft_batch_size: int,
    context_length,
    target_dim,
    model_variant: str = "moirai1",
    median_idx=None,
    log_every: int = 1,
    patch_size=16,
    seed: int = 0,
):
    """Generic LoRA fine-tuning loop with mini-batching via gradient accumulation.

    Parameters
    ----------
    train_pairs, val_pairs
        Lists of ``(item, y)`` tuples produced by :func:`_finetune_generic`.
        ``item`` is the GluonTS dict (with ``target`` and
        ``feat_dynamic_real``); ``y`` is the matching ``y_future`` numpy
        array. Pairing items with their y by tuple is what guarantees
        alignment cannot drift when we shuffle.
    ft_batch_size
        Effective batch size (gradient accumulation). The optimizer
        steps every ``ft_batch_size`` per-window losses, scaled by
        ``1/ft_batch_size`` so the gradient magnitude matches a true
        mini-batch.
    seed
        Seed for the per-epoch shuffle so runs are reproducible.
    """
    if ft_batch_size is None or ft_batch_size < 1:
        raise ValueError(f"ft_batch_size must be >= 1, got {ft_batch_size!r}")

    module.to(device)
    module.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, module.parameters()),
        lr=ft_lr,
        weight_decay=ft_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    rng = random.Random(seed)
    train_pairs = list(train_pairs)
    val_pairs = list(val_pairs)

    print(
        f"  Training: {len(train_pairs)} train pairs, "
        f"{len(val_pairs)} val pairs, batch={ft_batch_size} (gradient accumulation)."
    )

    for epoch in range(1, max_epochs + 1):
        module.train()
        rng.shuffle(train_pairs)

        train_losses: list[float] = []
        accum_count = 0
        optimizer.zero_grad()
        epoch_start = time.time()

        for batch_idx, (item, y_arr) in enumerate(train_pairs):
            try:
                pred = forecast_point(
                    model, item["target"], item["feat_dynamic_real"],
                    device=device, context_length=context_length,
                    target_dim=target_dim,
                    model_variant=model_variant,
                    patch_size=patch_size, median_idx=median_idx,
                ).squeeze(0)
                y_true = torch.tensor(
                    y_arr, dtype=torch.float32, device=device
                )

                if not torch.isfinite(pred).all():
                    if batch_idx == 0:
                        print("    Warning: non-finite prediction; skipping window.")
                    continue
                if not torch.isfinite(y_true).all():
                    if batch_idx == 0:
                        print("    Warning: non-finite target; skipping window.")
                    continue

                per_window_loss = torch.nn.functional.mse_loss(pred[: len(y_true)], y_true)
                if not torch.isfinite(per_window_loss):
                    if batch_idx == 0:
                        print("    Warning: non-finite loss; skipping window.")
                    continue

                # Scale so that the accumulated gradient magnitude
                # matches a true mini-batch of size ft_batch_size.
                (per_window_loss / ft_batch_size).backward()
                train_losses.append(float(per_window_loss.item()))
                accum_count += 1

                if accum_count >= ft_batch_size:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, module.parameters()),
                        ft_gradient_clip,
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0
            except Exception as e:
                if batch_idx == 0:
                    print(f"    Warning: {e}")
                continue

        # Step the optimizer on any remaining < ft_batch_size leftovers.
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, module.parameters()),
                ft_gradient_clip,
            )
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        module.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for val_idx, (item, y_arr) in enumerate(val_pairs):
                try:
                    pred = forecast_point(
                        model, item["target"], item["feat_dynamic_real"],
                        device=device, context_length=context_length,
                        target_dim=target_dim,
                        model_variant=model_variant,
                        patch_size=patch_size, median_idx=median_idx,
                    ).squeeze(0).detach().cpu().numpy()
                    val_loss = float(np.mean((pred[: len(y_arr)] - y_arr) ** 2))
                    if np.isfinite(val_loss):
                        val_losses.append(val_loss)
                    elif val_idx == 0:
                        print("    Warning: non-finite val loss; skipping window.")
                except Exception:
                    continue

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        epoch_elapsed = time.time() - epoch_start

        # One line per epoch: train/val loss + lr + wall-clock for the epoch.
        # ``log_every`` is kept as an opt-in for sparser logging; default is 1
        # (i.e. every epoch).
        if log_every <= 1 or epoch % log_every == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{max_epochs} | train_mse={avg_train:.5f}  "
                f"val_mse={avg_val:.5f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                f"elapsed={epoch_elapsed/60:.1f}m"
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = {k: v.clone() for k, v in module.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        module.load_state_dict(best_state)
    return best_val_loss


def adapter_name_for(model_variant: str, smoke_test: bool = False) -> str:
    """Standard adapter folder name. Variant-aware so 1.x and 2.0 can coexist."""
    base = f"{model_variant}_lora_adapter"
    return f"{base}_smoke" if smoke_test else base


def save_lora_adapter(
    module,
    *,
    checkpoint_dir,
    smoke_test,
    model_id,
    lora_rank,
    lora_alpha,
    lora_target_modules,
    best_val_loss,
    model_variant: str = "moirai1",
    target_scale: float = 1.0,
):
    """Persist the trained LoRA adapter and a JSON summary."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = checkpoint_dir / adapter_name_for(model_variant, smoke_test)
    module.save_pretrained(str(adapter_path))
    print(f"\n  LoRA adapter saved -> {adapter_path}")

    lora_info = {
        "base_model": model_id,
        "model_variant": model_variant,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": list(lora_target_modules),
        "best_val_mse": float(best_val_loss),
        "best_val_rmse": (
            float(best_val_loss ** 0.5) if np.isfinite(best_val_loss) else float("inf")
        ),
        "smoke_test": smoke_test,
        "target_scale": float(target_scale),
    }
    config_basename = f"lora_config_{model_variant}"
    config_name = (
        f"{config_basename}_smoke.json" if smoke_test else f"{config_basename}.json"
    )
    with open(checkpoint_dir / config_name, "w") as f:
        json.dump(lora_info, f, indent=2)
    print(f"  Config saved -> {config_name}")
    if np.isfinite(best_val_loss):
        print(f"  Best val RMSE: {best_val_loss**0.5:.4f}")
    else:
        print("  Best val RMSE: inf")
    return adapter_path


def _build_pairs(items, y_array, target_scale: float = 1.0):
    """Pair GluonTS items with matching y rows; optionally scale both target tracks."""
    pairs = []
    if target_scale != 1.0:
        scale = float(target_scale)
        for i, item in enumerate(items):
            scaled_item = dict(item)
            scaled_item["target"] = item["target"] / scale
            pairs.append((scaled_item, y_array[i] / scale))
    else:
        for i, item in enumerate(items):
            pairs.append((item, y_array[i]))
    return pairs


def _finetune_generic(
    *,
    dataset_dir,
    checkpoint_dir,
    model_id,
    context_length,
    prediction_length,
    target_dim,
    feat_dim,
    lora_rank,
    lora_alpha,
    lora_target_modules,
    lora_dropout,
    ft_lr,
    ft_weight_decay,
    ft_max_epochs,
    ft_patience,
    ft_batch_size,
    ft_gradient_clip,
    smoke_test=False,
    max_train_windows=None,
    max_val_windows=None,
    device_arg="auto",
    model_variant: str = "moirai1",
    target_scale: float = 1.0,
    past_features=None,
    future_features=None,
):
    """Generic end-to-end LoRA fine-tuning loop used by every method.

    ``model_variant`` selects between Moirai 1.x and 2.0. ``target_scale``
    divides both the past target track and the y labels by this factor
    before training (Method 3 uses ``GHI_SCALE_FACTOR``); it is also
    saved in the adapter's lora_config JSON so inference can apply the
    inverse without guessing.

    ``past_features`` / ``future_features`` are the column-name lists from
    the method's config and are used to align ``feat_dynamic_real`` when
    ``PAST_FEATURES`` includes past-only extras. When omitted we fall back
    to using every past column, which is correct for Methods 1 and 2 and
    for Method 3's current direct-GHI layout.
    """
    max_epochs = 1 if smoke_test else ft_max_epochs
    patience = 1 if smoke_test else ft_patience
    batch_size = min(4, ft_batch_size) if smoke_test else ft_batch_size
    if max_train_windows is None and smoke_test:
        max_train_windows = 8
    if max_val_windows is None and smoke_test:
        max_val_windows = 4
    if smoke_test:
        print(
            "SMOKE TEST MODE: 1 epoch, limited train/val windows, "
            "no full-quality training."
        )

    print(f"  Moirai variant: {model_variant}")
    print(f"  target_scale:   {target_scale}")

    base_module = load_module_for_variant(model_id, model_variant)
    module = inject_lora(
        base_module,
        lora_rank=lora_rank, lora_alpha=lora_alpha,
        lora_target_modules=lora_target_modules, lora_dropout=lora_dropout,
    )
    model = build_forecast_for_variant(
        module,
        model_variant=model_variant,
        prediction_length=prediction_length, context_length=context_length,
        target_dim=target_dim, feat_dim=feat_dim,
    )

    median_idx = median_quantile_index(module) if model_variant == "moirai2" else None

    print("Building dataloaders from .npy windows...")
    dyn_indices = _compute_dyn_indices(past_features, future_features)
    if dyn_indices is not None:
        print(
            f"  feat_dynamic_real alignment: selecting {len(dyn_indices)} past "
            f"columns (of {len(past_features) - 1}) that match FUTURE_FEATURES; "
            "past-only extras are dropped from feat_dynamic_real."
        )
    train_ds = npy_to_gluonts(
        "train",
        dataset_dir=dataset_dir,
        context_length=context_length,
        dyn_indices=dyn_indices,
    )
    val_ds = npy_to_gluonts(
        "val",
        dataset_dir=dataset_dir,
        context_length=context_length,
        dyn_indices=dyn_indices,
    )
    train_items = list(train_ds)
    val_items = list(val_ds)
    if max_train_windows is not None:
        train_items = train_items[:max_train_windows]
    if max_val_windows is not None:
        val_items = val_items[:max_val_windows]
    print(f"  Train: {len(train_items)} windows")
    print(f"  Val:   {len(val_items)} windows")

    print("\nStarting LoRA fine-tuning...")
    print(
        f"  LR={ft_lr}  Epochs={max_epochs}  Patience={patience}  Batch={batch_size}"
    )

    device = select_device(device_arg, smoke_test=smoke_test)
    print(f"  Device: {device}")

    dataset_dir = Path(dataset_dir)
    y_train = np.load(dataset_dir / "y_future_train.npy")
    y_val = np.load(dataset_dir / "y_future_val.npy")
    if max_train_windows is not None:
        y_train = y_train[:max_train_windows]
    if max_val_windows is not None:
        y_val = y_val[:max_val_windows]

    if len(train_items) != len(y_train):
        raise RuntimeError(
            f"Train item count ({len(train_items)}) does not match y_train "
            f"({len(y_train)}). Re-run the dataset step."
        )
    if len(val_items) != len(y_val):
        raise RuntimeError(
            f"Val item count ({len(val_items)}) does not match y_val "
            f"({len(y_val)}). Re-run the dataset step."
        )

    train_pairs = _build_pairs(train_items, y_train, target_scale=target_scale)
    val_pairs = _build_pairs(val_items, y_val, target_scale=target_scale)

    best_val_loss = train_lora(
        model=model, module=module,
        train_pairs=train_pairs, val_pairs=val_pairs,
        device=device,
        max_epochs=max_epochs, patience=patience,
        ft_lr=ft_lr, ft_weight_decay=ft_weight_decay,
        ft_gradient_clip=ft_gradient_clip,
        ft_batch_size=batch_size,
        context_length=context_length, target_dim=target_dim,
        model_variant=model_variant, median_idx=median_idx,
    )

    save_lora_adapter(
        module,
        checkpoint_dir=checkpoint_dir,
        smoke_test=smoke_test,
        model_id=model_id,
        lora_rank=lora_rank, lora_alpha=lora_alpha,
        lora_target_modules=lora_target_modules,
        best_val_loss=best_val_loss,
        model_variant=model_variant,
        target_scale=target_scale,
    )


# =====================================================================
# SHARED: model loading for inference (the actual inference loop and
# evaluation/reporting live in functions/results.py).
# =====================================================================

def load_finetuned_model(
    *,
    model_id,
    adapter_path,
    prediction_length,
    context_length,
    target_dim,
    feat_dim,
    past_feat_dynamic_real_dim=0,
    model_variant: str = "moirai1",
):
    """Load the base Moirai (1.x or 2.0) + a saved LoRA adapter for inference."""
    from peft import PeftModel

    print(f"Loading {model_id} + LoRA adapter ({model_variant})...")
    base_module = load_module_for_variant(model_id, model_variant)
    module = PeftModel.from_pretrained(base_module, str(adapter_path))
    module.eval()
    model = build_forecast_for_variant(
        module,
        model_variant=model_variant,
        prediction_length=prediction_length,
        context_length=context_length,
        target_dim=target_dim, feat_dim=feat_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
    )
    return model, module


# =====================================================================
# METHOD 1 - CAF (PVLib clear-sky + ERA5 covariates)
# =====================================================================

def finetune_method1(**kwargs):
    """Method 1 LoRA fine-tune entrypoint (Moirai 1.x by default)."""
    print("\n=== METHOD 1: CAF (PVLib + ERA5) - LoRA fine-tune ===\n")
    kwargs.setdefault("model_variant", "moirai1")
    kwargs.setdefault("target_scale", 1.0)
    _finetune_generic(**kwargs)


# =====================================================================
# METHOD 2 - Multi-station + NSRDB clear-sky + CAF -> GHI
# =====================================================================

def finetune_method2(**kwargs):
    """Method 2 LoRA fine-tune entrypoint (multi-station, CAF target).

    Reuses ``_finetune_generic``; the dataset stage in
    ``preprocess.build_dataset_method2`` already concatenates per-station
    windows so nothing in this function is multi-station-aware.
    """
    print("\n=== METHOD 2: CAF (NSRDB + multi-station) - LoRA fine-tune ===\n")
    kwargs.setdefault("model_variant", "moirai1")
    kwargs.setdefault("target_scale", 1.0)
    _finetune_generic(**kwargs)


# =====================================================================
# METHOD 3 - Direct GHI forecasting (multi-station, GHI scaling)
# =====================================================================

def finetune_method3(**kwargs):
    """Method 3 LoRA fine-tune entrypoint (direct GHI, multi-station).

    The on-disk windows are in physical units (W/m^2). This function
    divides the past target track and the y labels by ``target_scale``
    (= ``cfg.GHI_SCALE_FACTOR``) before training. The same scale is
    written into the adapter's lora_config JSON so the matching
    inference path can multiply back without guessing.
    """
    print("\n=== METHOD 3: Direct GHI - LoRA fine-tune ===\n")
    target_scale = kwargs.pop("target_scale", None)
    if target_scale is None or target_scale <= 0:
        raise ValueError(
            "finetune_method3 requires target_scale > 0 (typically cfg.GHI_SCALE_FACTOR)."
        )
    kwargs.setdefault("model_variant", "moirai2")
    _finetune_generic(target_scale=float(target_scale), **kwargs)
