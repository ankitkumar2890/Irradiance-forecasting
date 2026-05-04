"""results.py - Inference + master-file-driven reporting for all 3 methods.

Each method exposes two entrypoints::

    run_inference_method{N}(*, cfg, dataset_dir, checkpoint_dir,
                            results_dir, model_variant)

    build_report_method{N}(*, cfg, final_csv_dir, results_dir,
                           validation_ghi_filter_wm2)

The orchestrator (``moirai/moirai.py``) dispatches the CLI sub-commands
``infer`` and ``evaluate`` into these per-method functions.

Conventions
-----------
* ``forecast_start`` is uniformly the **issue time** (the last hour of
  the encoder context, equivalently ``first_forecasted_hour - 1h``) for
  every method. ``lead_time_h`` runs from 1 to ``PREDICTION_LENGTH``.
* The persistence-skill baseline is no longer computed; it was only
  meaningful when measured on the same windowed validation slice as the
  model, which is not what the previous implementation did.
* Multi-station evaluation requires both the predictions CSV and the
  prepared CSV to carry ``station_id``; we hard-fail rather than join
  on ``datetime`` only (which silently corrupted rows).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------
# Make moirai/master_files/ importable. export_pdf.py does
# ``from master_metrics import calculate_metrics``, so master_files/
# itself must be on sys.path (not its parent).
# ---------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_MASTER_DIR = _THIS_DIR.parent / "master_files"
if str(_MASTER_DIR) not in sys.path:
    sys.path.insert(0, str(_MASTER_DIR))

from master_metrics import calculate_metrics, print_metrics  # noqa: E402
from master_plots import (  # noqa: E402
    plot_4panel_evaluation,
    plot_time_series,
    plot_two_week_comparison,
)
from export_pdf import build_finetuned_pdf  # noqa: E402

from functions.model import (  # noqa: E402
    _extract_quantile_point_forecast,
    _compute_dyn_indices,
    adapter_name_for,
    forecast_point,
    forecast_quantiles_v1,
    load_finetuned_model,
    median_quantile_index,
)
from functions.preprocess import load_final_csv  # noqa: E402


# =====================================================================
# Shared helpers
# =====================================================================

def compute_metrics(
    y_true,
    y_pred,
    label: str = "",
    *,
    mape_threshold: float = 0.0,
) -> dict:
    """Thin wrapper around ``master_metrics.calculate_metrics``.

    Callers still choose the relevant row filter before passing arrays in
    here. ``mape_threshold`` only controls the denominator floor / row
    eligibility inside the shared master-metrics implementation.
    """
    metrics = calculate_metrics(y_true, y_pred, mape_threshold=mape_threshold)
    return {
        "label": label,
        "MAE": float(metrics["MAE"]),
        "RMSE": float(metrics["RMSE"]),
        "nRMSE_pct": float(metrics["nRMSE"]),
        "MAPE_pct": float(metrics["MAPE"]),
        "N": int(metrics["N"]),
    }


def _select_device() -> torch.device:
    """Pick a torch device for inference.

    Moirai 1.x's ``_generate_time_id`` uses ops (e.g. ``aten::_cummax_helper``)
    that are not implemented on MPS, so we cannot pick MPS automatically even
    though the inference forward pass is otherwise read-only. Prefer CUDA,
    otherwise fall back to CPU. (If you really want to try MPS, you can run
    with ``PYTORCH_ENABLE_MPS_FALLBACK=1`` set in your shell and edit this
    function to return ``torch.device("mps")``.)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _read_target_scale(checkpoint_dir: Path, model_variant: str, smoke_test: bool) -> float:
    """Look up the ``target_scale`` saved by ``save_lora_adapter`` (defaults to 1.0)."""
    suffix = "_smoke" if smoke_test else ""
    cfg_path = Path(checkpoint_dir) / f"lora_config_{model_variant}{suffix}.json"
    if not cfg_path.exists():
        return 1.0
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return 1.0
    return float(payload.get("target_scale", 1.0))


def _forecast_start_from_first_future_ts(first_future_ts: pd.Timestamp) -> pd.Timestamp:
    """Uniform ``forecast_start`` = issue time = last context hour."""
    return pd.Timestamp(first_future_ts) - pd.Timedelta(hours=1)


# =====================================================================
# Generic inference loop (Methods 1 & 2 - CAF target, no scaling)
# =====================================================================

def _run_inference_caf(
    *,
    cfg,
    dataset_dir,
    checkpoint_dir,
    results_dir,
    model_id,
    target_col_name: str = "CAF",
    model_variant: str = "moirai1",
    smoke_test: bool = False,
) -> Path:
    """CAF-target inference for Methods 1 & 2.

    If ``dataset_dir / 'station_ids_val.npy'`` exists, each prediction
    row gets a ``station_id`` column (Method 2).
    """
    dataset_dir = Path(dataset_dir)
    checkpoint_dir = Path(checkpoint_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = checkpoint_dir / adapter_name_for(model_variant, smoke_test)
    target_scale = _read_target_scale(checkpoint_dir, model_variant, smoke_test)
    dyn_indices = _compute_dyn_indices(
        list(getattr(cfg, "PAST_FEATURES", [])),
        list(getattr(cfg, "FUTURE_FEATURES", [])),
    )

    model, module = load_finetuned_model(
        model_id=model_id,
        adapter_path=adapter_path,
        prediction_length=cfg.PREDICTION_LENGTH,
        context_length=cfg.CONTEXT_LENGTH,
        target_dim=cfg.TARGET_DIM,
        feat_dim=cfg.FEAT_DIM,
        model_variant=model_variant,
    )

    device = _select_device()
    module.to(device)

    # Quantile bookkeeping: every method now writes p10/p50/p90 to the
    # predictions CSV. Moirai 2.0 reads them straight off the model's
    # quantile head; Moirai 1.x derives them from the predicted
    # distribution (analytical icdf if available, MC sampling otherwise).
    quantile_levels = [0.1, 0.5, 0.9]
    if model_variant == "moirai2":
        quantile_indices = {
            f"p{int(round(q * 100))}": _quantile_index_for(module, q)
            for q in quantile_levels
        }
    else:
        quantile_indices = None

    X_past = np.load(dataset_dir / "X_past_val.npy")
    X_future = np.load(dataset_dir / "X_future_val.npy")
    y_future = np.load(dataset_dir / "y_future_val.npy")
    times = np.load(dataset_dir / "times_val.npy", allow_pickle=True).astype(
        "datetime64[ns]"
    )
    station_ids_path = dataset_dir / "station_ids_val.npy"
    station_ids = (
        np.load(station_ids_path, allow_pickle=True)
        if station_ids_path.exists()
        else None
    )

    print(f"Validation windows: {len(X_past)}")
    print(f"target_scale used at inference: {target_scale}")
    if station_ids is not None:
        print(f"Stations: {sorted(set(station_ids.tolist()))}")

    all_preds, all_true, csv_rows = [], [], []
    true_col = f"{target_col_name}_true"
    pred_col = f"{target_col_name}_pred"
    pred_p10_col = f"{target_col_name}_pred_p10"
    pred_p90_col = f"{target_col_name}_pred_p90"

    for i in range(len(X_past)):
        target = (X_past[i, :, 0] / target_scale).astype(np.float32)
        if dyn_indices is None:
            past_feats = X_past[i, :, 1:].T
        else:
            past_feats = X_past[i, :, 1:][:, dyn_indices].T
        fut_feats = X_future[i].T
        dyn_feats = np.hstack([past_feats, fut_feats]).astype(np.float32)

        first_future_ts = pd.Timestamp(times[i, 0])
        forecast_start = _forecast_start_from_first_future_ts(first_future_ts)

        if model_variant == "moirai1":
            q_dict = forecast_quantiles_v1(
                model, target, dyn_feats,
                device=device,
                context_length=cfg.CONTEXT_LENGTH,
                target_dim=cfg.TARGET_DIM,
                quantiles=quantile_levels,
            )
        else:
            q_dict = _forecast_quantiles_v2(
                model, target, dyn_feats,
                device=device,
                context_length=cfg.CONTEXT_LENGTH,
                target_dim=cfg.TARGET_DIM,
                quantile_indices=quantile_indices,
            )

        pred_p10 = q_dict["p10"][: cfg.PREDICTION_LENGTH] * target_scale
        pred = q_dict["p50"][: cfg.PREDICTION_LENGTH] * target_scale
        pred_p90 = q_dict["p90"][: cfg.PREDICTION_LENGTH] * target_scale

        all_preds.append(pred)
        all_true.append(y_future[i])

        for h in range(cfg.PREDICTION_LENGTH):
            ts = pd.Timestamp(times[i, h])
            lead_time_h = int((ts - forecast_start) / pd.Timedelta(hours=1))
            row = {
                "datetime": ts,
                "hour": ts.hour,
                "lead_time_h": lead_time_h,
                "forecast_start": forecast_start,
                true_col: float(y_future[i, h]),
                pred_col: float(pred[h]),
                pred_p10_col: float(pred_p10[h]),
                pred_p90_col: float(pred_p90[h]),
            }
            if station_ids is not None:
                row["station_id"] = str(station_ids[i])
            csv_rows.append(row)

        if i % 20 == 0:
            print(f"  {i}/{len(X_past)}")

    df_out = pd.DataFrame(csv_rows)
    sort_cols = [c for c in ["station_id", "datetime", "lead_time_h"] if c in df_out.columns]
    df_out = df_out.sort_values(sort_cols).reset_index(drop=True)
    out_csv = results_dir / "finetuned_predictions.csv"
    df_out.to_csv(out_csv, index=False)

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    rmse = float(np.sqrt(np.mean((all_preds - all_true) ** 2)))
    mae = float(np.mean(np.abs(all_preds - all_true)))
    print(f"\n  Validation - {target_col_name} RMSE: {rmse:.4f}  MAE: {mae:.4f}")

    metrics = {
        f"{target_col_name}_RMSE": rmse,
        f"{target_col_name}_MAE": mae,
        "model_variant": model_variant,
        "target_scale": target_scale,
    }
    with open(results_dir / "finetuned_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("  Saved -> results/finetuned_predictions.csv + finetuned_metrics.json")
    return out_csv


# =====================================================================
# Generic inference loop (Method 3 - Direct GHI: scaling + daylight mask
# + p10/p50/p90 quantile output, Moirai 2.0 only)
# =====================================================================

def _forecast_quantiles_v2(
    model,
    target,
    feat_dynamic_real,
    *,
    device,
    context_length,
    target_dim,
    quantile_indices: dict,
    past_feat_dynamic_real: Optional[np.ndarray] = None,
):
    """Single Moirai 2.0 forward pass returning each requested quantile."""
    past_target = torch.tensor(
        target, dtype=torch.float32, device=device
    ).view(1, context_length, target_dim)
    past_observed = torch.ones_like(past_target, dtype=torch.bool)
    past_is_pad = torch.zeros((1, context_length), dtype=torch.bool, device=device)
    feat_dynamic = torch.tensor(
        feat_dynamic_real.T, dtype=torch.float32, device=device
    ).unsqueeze(0)
    observed_feat = torch.ones_like(feat_dynamic, dtype=torch.bool)

    fwd_kwargs = {
        "feat_dynamic_real": feat_dynamic,
        "observed_feat_dynamic_real": observed_feat,
    }
    if past_feat_dynamic_real is not None and past_feat_dynamic_real.size > 0:
        past_feat_dynamic = torch.tensor(
            past_feat_dynamic_real, dtype=torch.float32, device=device
        ).unsqueeze(0)
        observed_past_feat = torch.ones_like(past_feat_dynamic, dtype=torch.bool)
        fwd_kwargs["past_feat_dynamic_real"] = past_feat_dynamic
        fwd_kwargs["past_observed_feat_dynamic_real"] = observed_past_feat

    preds = model(past_target, past_observed, past_is_pad, **fwd_kwargs)
    out = {}
    for label, q_idx in quantile_indices.items():
        out[label] = (
            _extract_quantile_point_forecast(preds, q_idx)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
    return out


def _quantile_index_for(module, target_quantile: float) -> int:
    """Find the index of ``target_quantile`` (or the closest one) in a Moirai 2.0 module."""
    quantiles = list(getattr(module, "quantile_levels", [0.5]))
    if target_quantile in quantiles:
        return quantiles.index(target_quantile)
    for attr in ("base_model", "model", "module"):
        nested = getattr(module, attr, None)
        if nested is not None and hasattr(nested, "quantile_levels"):
            return _quantile_index_for(nested, target_quantile)
    return min(
        range(len(quantiles)),
        key=lambda idx: abs(float(quantiles[idx]) - target_quantile),
    )


def _validate_method3_contract(cfg) -> tuple[list[str], list[str]]:
    """Verify the cfg layout that ``_run_inference_direct_ghi`` depends on.

    Returns ``(past_features, future_features)`` for callers to reuse.
    Raises a clear, actionable ``ValueError`` if any contract is broken.
    """
    past_features: list[str] = list(getattr(cfg, "PAST_FEATURES", []))
    future_features: list[str] = list(getattr(cfg, "FUTURE_FEATURES", []))
    measured_col = str(getattr(cfg, "MEASURED_GHI_COL", "w_ghr"))

    if not past_features or not future_features:
        raise ValueError(
            "Method 3 inference needs cfg.PAST_FEATURES and cfg.FUTURE_FEATURES "
            "to know which feature columns to send through which channel."
        )
    if past_features[0] != measured_col:
        raise ValueError(
            f"Method 3 expects PAST_FEATURES[0] == '{measured_col}' so the "
            f"autoregressive target track is the measured GHI series. Got "
            f"PAST_FEATURES[0]={past_features[0]!r}. Update cfg or rebuild the dataset."
        )
    missing_in_past = [f for f in future_features if f not in past_features]
    if missing_in_past:
        raise ValueError(
            "Method 3 expects every entry in FUTURE_FEATURES to also be in "
            f"PAST_FEATURES (used by the past/future index map). Missing: "
            f"{missing_in_past}."
        )
    return past_features, future_features


def _run_inference_direct_ghi(
    *,
    cfg,
    dataset_dir,
    checkpoint_dir,
    results_dir,
    model_id,
    model_variant: str = "moirai2",
    smoke_test: bool = False,
) -> Path:
    """Direct-GHI inference for Method 3.

    Differences from :func:`_run_inference_caf`:
      * Past target track is divided by ``cfg.GHI_SCALE_FACTOR`` (or the
        ``target_scale`` that was saved with the adapter, whichever is
        more authoritative); predictions are multiplied back at the end.
      * p10 / p50 / p90 quantiles are written to the predictions CSV.
    """
    if model_variant != "moirai2":
        raise ValueError(
            "Method 3 (direct GHI) requires model_variant='moirai2' because the "
            "p10/p50/p90 quantile output relies on Moirai 2.0's quantile head."
        )

    past_features, future_features = _validate_method3_contract(cfg)
    measured_col = str(getattr(cfg, "MEASURED_GHI_COL", "w_ghr"))

    dataset_dir = Path(dataset_dir)
    checkpoint_dir = Path(checkpoint_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dyn_indices = _compute_dyn_indices(past_features, future_features)

    cfg_scale = float(getattr(cfg, "GHI_SCALE_FACTOR", 1000.0))
    saved_scale = _read_target_scale(checkpoint_dir, model_variant, smoke_test)
    if saved_scale not in (cfg_scale, 1.0):
        print(
            f"  Note: cfg.GHI_SCALE_FACTOR={cfg_scale} but the adapter was trained with "
            f"target_scale={saved_scale}. Using the adapter's saved value to keep "
            f"training/inference consistent."
        )
    ghi_scale = saved_scale if saved_scale != 1.0 else cfg_scale
    adapter_path = checkpoint_dir / adapter_name_for(model_variant, smoke_test)
    model, module = load_finetuned_model(
        model_id=model_id,
        adapter_path=adapter_path,
        prediction_length=cfg.PREDICTION_LENGTH,
        context_length=cfg.CONTEXT_LENGTH,
        target_dim=cfg.TARGET_DIM,
        feat_dim=cfg.FEAT_DIM,
        model_variant=model_variant,
    )

    device = _select_device()
    module.to(device)

    quantile_indices = {
        "p10": _quantile_index_for(module, 0.1),
        "p50": _quantile_index_for(module, 0.5),
        "p90": _quantile_index_for(module, 0.9),
    }

    X_past = np.load(dataset_dir / "X_past_val.npy")
    X_future = np.load(dataset_dir / "X_future_val.npy")
    y_future = np.load(dataset_dir / "y_future_val.npy")
    times = np.load(dataset_dir / "times_val.npy", allow_pickle=True).astype(
        "datetime64[ns]"
    )
    station_ids_path = dataset_dir / "station_ids_val.npy"
    station_ids = (
        np.load(station_ids_path, allow_pickle=True)
        if station_ids_path.exists()
        else None
    )

    print(f"Validation windows: {len(X_past)}")
    print(f"GHI_SCALE_FACTOR (effective): {ghi_scale}")

    csv_rows = []
    all_preds, all_true = [], []
    for i in range(len(X_past)):
        target = (X_past[i, :, 0] / ghi_scale).astype(np.float32)
        if dyn_indices is None:
            past_feats = X_past[i, :, 1:].T
        else:
            past_feats = X_past[i, :, 1:][:, dyn_indices].T
        fut_feats = X_future[i].T
        dyn_feats = np.hstack([past_feats, fut_feats]).astype(np.float32)

        first_future_ts = pd.Timestamp(times[i, 0])
        forecast_start = _forecast_start_from_first_future_ts(first_future_ts)

        q = _forecast_quantiles_v2(
            model, target, dyn_feats,
            device=device,
            context_length=cfg.CONTEXT_LENGTH,
            target_dim=cfg.TARGET_DIM,
            quantile_indices=quantile_indices,
        )

        pred = q["p50"][: cfg.PREDICTION_LENGTH] * ghi_scale
        pred_p10 = q["p10"][: cfg.PREDICTION_LENGTH] * ghi_scale
        pred_p90 = q["p90"][: cfg.PREDICTION_LENGTH] * ghi_scale
        # ``y_future`` is saved by ``build_dataset_method3`` as the raw target
        # column (``w_ghr``) straight from the CSV, i.e. already in W/m^2.
        # Earlier versions of this file multiplied it by ``ghi_scale`` here,
        # which inflated every truth value 1000x and produced absurd RMSE
        # numbers (e.g. ~500,000 W/m^2 for a quantity bounded by ~1100 W/m^2).
        y_true_wm2 = y_future[i].astype(np.float32)

        all_preds.append(pred)
        all_true.append(y_true_wm2)

        for h in range(cfg.PREDICTION_LENGTH):
            ts = pd.Timestamp(times[i, h])
            lead_time_h = int((ts - forecast_start) / pd.Timedelta(hours=1))
            row = {
                "datetime": ts,
                "hour": ts.hour,
                "lead_time_h": lead_time_h,
                "forecast_start": forecast_start,
                "GHI_true": float(y_true_wm2[h]),
                "GHI_pred_p10": float(pred_p10[h]),
                "GHI_pred": float(pred[h]),
                "GHI_pred_p90": float(pred_p90[h]),
            }
            if station_ids is not None:
                row["station_id"] = str(station_ids[i])
            csv_rows.append(row)

        if i % 20 == 0:
            print(f"  {i}/{len(X_past)}")

    df_out = pd.DataFrame(csv_rows)
    sort_cols = [c for c in ["station_id", "datetime", "lead_time_h"] if c in df_out.columns]
    df_out = df_out.sort_values(sort_cols).reset_index(drop=True)
    out_csv = results_dir / "finetuned_predictions.csv"
    df_out.to_csv(out_csv, index=False)

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    rmse = float(np.sqrt(np.mean((all_preds - all_true) ** 2)))
    mae = float(np.mean(np.abs(all_preds - all_true)))
    mean_true = float(np.mean(all_true[all_true > 0])) if (all_true > 0).any() else 0.0
    nrmse = (rmse / mean_true * 100) if mean_true > 0 else float("nan")
    print(
        f"\n  Direct GHI Validation - RMSE: {rmse:.2f} W/m^2  "
        f"MAE: {mae:.2f} W/m^2  nRMSE: {nrmse:.2f}%"
    )

    metrics = {
        "GHI_RMSE_wm2": rmse,
        "GHI_MAE_wm2": mae,
        "GHI_nRMSE_pct": nrmse,
        "ghi_scale_factor": ghi_scale,
        "model_variant": model_variant,
    }
    with open(results_dir / "finetuned_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("  Saved -> results/finetuned_predictions.csv + finetuned_metrics.json")
    return out_csv


# =====================================================================
# Generic report builder
# =====================================================================

def _stratified_caf_rmse(df: pd.DataFrame) -> Iterable[tuple[str, int, float]]:
    """Yield (label, N, RMSE) tuples for clear/partly/overcast CAF buckets."""
    if "CAF_true" not in df.columns or "CAF_pred" not in df.columns:
        return
    for name, lo, hi in [
        ("Clear (>0.7)", 0.7, 1.01),
        ("Partly (0.3-0.7)", 0.3, 0.7),
        ("Overcast (<0.3)", -0.01, 0.3),
    ]:
        mask = (df["CAF_true"] >= lo) & (df["CAF_true"] < hi)
        sub = df[mask]
        if len(sub):
            r = float(np.sqrt(np.mean((sub["CAF_pred"] - sub["CAF_true"]) ** 2)))
            yield (name, int(len(sub)), r)


def _horizon_rmse(df: pd.DataFrame, target_col: str) -> dict:
    """Return {lead_time_h: RMSE} on the chosen target column."""
    out = {}
    if "lead_time_h" not in df.columns:
        return out
    for lead_time, group in df.groupby("lead_time_h"):
        true = group[f"{target_col}_true"].to_numpy() if target_col == "CAF" else group["GHI_true"].to_numpy()
        pred = group[f"{target_col}_pred"].to_numpy() if target_col == "CAF" else group["GHI_pred"].to_numpy()
        out[int(lead_time)] = float(np.sqrt(np.mean((pred - true) ** 2)))
    return out


def _build_report_generic(
    *,
    method: int,
    cfg,
    final_csv_dir,
    results_dir,
    target_col: str,                    # "CAF" or "GHI"
    clearsky_col: str = "clear_sky_ghi",
    measured_col: str = "w_ghr",
    zenith_col: str = "zenith_angle",
    by_station: bool = False,
    validation_ghi_filter_wm2: float = 20.0,
    title_prefix: str = "Method",
    station_label_default: str = "unknown",
):
    """Read predictions, do CAF->GHI if needed, render plots/PDF/metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    pred_file = results_dir / "finetuned_predictions.csv"
    if not pred_file.exists():
        print(f"Error: {pred_file} not found. Run inference first.")
        return

    df = pd.read_csv(pred_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # ---- Merge with the prepared CSV to get clear-sky / measured GHI / zenith
    proc = load_final_csv(final_csv_dir=final_csv_dir)
    needed_cols = {clearsky_col, zenith_col, measured_col}
    if by_station and "station_id" in proc.columns:
        needed_cols.add("station_id")
    missing = needed_cols - set(proc.columns)
    if missing:
        raise ValueError(
            f"Method {method} final CSV is missing columns required for "
            f"evaluation: {sorted(missing)}. Add them and rerun."
        )

    # Multi-station methods MUST have station_id on both sides; we no longer
    # silently fall back to a datetime-only join (which used to cross-pollute
    # rows from different stations).
    if by_station:
        df_has_station = "station_id" in df.columns
        proc_has_station = "station_id" in proc.columns
        if not (df_has_station and proc_has_station):
            raise ValueError(
                f"Method {method} is configured as multi-station (by_station=True) but "
                f"station_id is missing from "
                f"{'predictions CSV' if not df_has_station else 'prepared CSV'}. "
                "Either rebuild the dataset / predictions to include station_id, "
                "or call build_report with by_station=False."
            )

    proc_merge_cols = ["datetime", clearsky_col, zenith_col, measured_col]
    join_keys = ["datetime"]
    if by_station:
        proc_merge_cols.insert(0, "station_id")
        join_keys = ["station_id", "datetime"]
    proc_slim = proc[proc_merge_cols].drop_duplicates(subset=join_keys)

    df = df.merge(proc_slim, on=join_keys, how="left")
    df = df.dropna(subset=[clearsky_col, zenith_col, measured_col])
    if df.empty:
        print(
            f"No aligned evaluation rows for method {method}. Check that the "
            f"timestamps (and station_ids) in {pred_file.name} match the "
            f"prepared CSV in {final_csv_dir}."
        )
        return

    # ---- CAF -> GHI reconstruction (only for CAF targets)
    if target_col == "CAF":
        df["GHI_true"] = df[measured_col]
        df["GHI_pred"] = df["CAF_pred"] * df[clearsky_col]
        # Propagate CAF prediction interval -> GHI prediction interval so
        # the PDF's "p10/p90" page works for Method 1 and Method 2 too.
        if "CAF_pred_p10" in df.columns:
            df["GHI_pred_p10"] = df["CAF_pred_p10"] * df[clearsky_col]
        if "CAF_pred_p90" in df.columns:
            df["GHI_pred_p90"] = df["CAF_pred_p90"] * df[clearsky_col]
    else:
        if "GHI_true" not in df.columns:
            df["GHI_true"] = df[measured_col]
        if "GHI_pred" not in df.columns:
            raise ValueError(
                "Method 3 evaluation expects GHI_pred in finetuned_predictions.csv."
            )

    validation_df = df[df["GHI_true"] > validation_ghi_filter_wm2].copy()
    print(
        f"Total rows: {len(df):,}  Filtered rows "
        f"(GHI_true > {validation_ghi_filter_wm2:.0f}): {len(validation_df):,}"
    )
    if validation_df.empty:
        print("No validation rows passed the post-reconstruction GHI filter.")
        return

    # ---- Two-week window (always anchored at the start of the validation set)
    two_week_start = validation_df["datetime"].min()
    two_week_end = two_week_start + pd.Timedelta(days=14)
    two_week_df = validation_df[
        (validation_df["datetime"] >= two_week_start)
        & (validation_df["datetime"] < two_week_end)
    ].copy()

    # ---- Metrics
    ghi_m = compute_metrics(
        validation_df["GHI_true"].to_numpy(),
        validation_df["GHI_pred"].to_numpy(),
        f"GHI_method{method}",
        mape_threshold=validation_ghi_filter_wm2,
    )
    two_week_ghi_m = compute_metrics(
        two_week_df["GHI_true"].to_numpy(),
        two_week_df["GHI_pred"].to_numpy(),
        "GHI_2week",
        mape_threshold=validation_ghi_filter_wm2,
    )
    master_ghi_m = calculate_metrics(
        validation_df["GHI_true"].to_numpy(),
        validation_df["GHI_pred"].to_numpy(),
        mape_threshold=validation_ghi_filter_wm2,
    )

    caf_m = {}
    two_week_caf_m = {}
    if target_col == "CAF":
        caf_m = compute_metrics(
            validation_df["CAF_true"].to_numpy(),
            validation_df["CAF_pred"].to_numpy(),
            f"CAF_method{method}",
            mape_threshold=0.0,
        )
        two_week_caf_m = compute_metrics(
            two_week_df["CAF_true"].to_numpy(),
            two_week_df["CAF_pred"].to_numpy(),
            "CAF_2week",
            mape_threshold=0.0,
        )

    horizon_rmse = _horizon_rmse(validation_df, target_col)
    per_station_metrics = {}
    if by_station and "station_id" in validation_df.columns:
        for station_id, group in validation_df.groupby("station_id", sort=True):
            per_station_metrics[str(station_id)] = compute_metrics(
                group["GHI_true"].to_numpy(),
                group["GHI_pred"].to_numpy(),
                f"GHI_{station_id}",
                mape_threshold=validation_ghi_filter_wm2,
            )

    # ---- Console summary
    banner = "=" * 55
    print(f"\n{banner}\n  METHOD {method} - DAYTIME METRICS\n{banner}")
    if target_col == "CAF":
        print("\n  CAF:")
        for k, v in caf_m.items():
            if k == "label":
                continue
            print(f"    {k:18s}: {v:.4f}" if isinstance(v, float) else f"    {k:18s}: {v}")
    print("\n  GHI (W/m^2):")
    for k, v in ghi_m.items():
        if k == "label":
            continue
        print(f"    {k:18s}: {v:.4f}" if isinstance(v, float) else f"    {k:18s}: {v}")
    print_metrics(master_ghi_m, title=f"MASTER GHI METRICS (filter > {validation_ghi_filter_wm2:.0f} W/m^2)")
    print_metrics(two_week_ghi_m, title="TWO-WEEK GHI METRICS", unit="W/m^2")

    if target_col == "CAF":
        print("\n  Stratified CAF RMSE:")
        for name, n, r in _stratified_caf_rmse(validation_df):
            print(f"    {name:20s}: RMSE={r:.4f}  N={n}")
    print("\n  RMSE by forecast hour:")
    for h in [1, 6, 12, 24]:
        if h in horizon_rmse:
            print(f"    +{h:2d}h: RMSE={horizon_rmse[h]:.4f}")

    if per_station_metrics:
        print("\n  Per-station GHI metrics:")
        for sid, m in per_station_metrics.items():
            print(
                f"    {sid:14s}: RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  "
                f"nRMSE={m['nRMSE_pct']:.2f}%  N={m['N']}"
            )

    # ---- Persist evaluation_metrics.json + evaluation_report.txt
    all_metrics = {
        "method": method,
        "filters": {"ghi_true_gt_wm2": validation_ghi_filter_wm2},
        "ghi": ghi_m,
        "ghi_two_week": two_week_ghi_m,
        "ghi_master_metrics": {k: float(v) for k, v in master_ghi_m.items()},
        "horizon_rmse": horizon_rmse,
    }
    if target_col == "CAF":
        all_metrics["caf"] = caf_m
        all_metrics["caf_two_week"] = two_week_caf_m
    if per_station_metrics:
        all_metrics["per_station"] = per_station_metrics
    with open(results_dir / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    report_lines = [
        f"METHOD {method} ({title_prefix}) - EVALUATION REPORT",
        f"Filter: measured GHI_true > {validation_ghi_filter_wm2:.0f} W/m^2",
        f"Rows evaluated: {len(validation_df)}",
        "",
        "GHI Metrics (custom):",
        f"  RMSE: {ghi_m['RMSE']:.4f}",
        f"  MAE: {ghi_m['MAE']:.4f}",
        f"  nRMSE_pct: {ghi_m['nRMSE_pct']:.4f}",
        f"  MAPE_pct: {ghi_m['MAPE_pct']:.4f}",
        "",
        "GHI Metrics (master_metrics.calculate_metrics):",
        f"  RMSE: {master_ghi_m['RMSE']:.4f}",
        f"  nRMSE: {master_ghi_m['nRMSE']:.4f}",
        f"  MAE: {master_ghi_m['MAE']:.4f}",
        f"  MAPE: {master_ghi_m['MAPE']:.4f}",
        "",
        "Two-Week GHI Metrics:",
        f"  N: {two_week_ghi_m.get('N', 0)}",
        f"  RMSE: {two_week_ghi_m.get('RMSE', float('nan')):.4f}",
        f"  MAE: {two_week_ghi_m.get('MAE', float('nan')):.4f}",
    ]
    if target_col == "CAF":
        report_lines.extend([
            "",
            "CAF Metrics:",
            f"  RMSE: {caf_m['RMSE']:.4f}",
            f"  MAE: {caf_m['MAE']:.4f}",
            f"  nRMSE_pct: {caf_m['nRMSE_pct']:.4f}",
            f"  MAPE_pct: {caf_m['MAPE_pct']:.4f}",
        ])
    if horizon_rmse:
        report_lines.extend(["", f"Horizon RMSE ({target_col}):"])
        report_lines.extend(
            [f"  +{lead:02d}h: {rmse:.4f}" for lead, rmse in sorted(horizon_rmse.items())]
        )
    if per_station_metrics:
        report_lines.extend(["", "Per-station GHI Metrics:"])
        for sid, m in per_station_metrics.items():
            report_lines.append(
                f"  {sid:14s}  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  "
                f"nRMSE={m['nRMSE_pct']:.2f}%  N={m['N']}"
            )
    with open(results_dir / "evaluation_report.txt", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    # ---- Plots driven by master_plots.* helpers
    #
    # Design choice: the GHI > validation_ghi_filter_wm2 mask is intentionally
    # NOT applied to the plot inputs below. The mask only makes sense for
    # *metrics* (where dividing by tiny night-time GHI values blows up MAPE
    # and inflates nRMSE). For the time-series and 4-panel plots, the user
    # wants to see the full diurnal cycle (night sits at zero, daytime swings
    # up and down) so the predicted-vs-measured curves are interpretable.
    # ``validation_df`` (filtered) is still used for every metric below.
    #
    # For multi-station methods we collapse to a single representative station
    # before slicing the weekly window, otherwise iloc[: 7 * 24] would
    # interleave 9 stations into a meaningless first ~7 hours.
    if by_station and "station_id" in df.columns:
        plot_station = (
            df["station_id"].dropna().astype(str).value_counts().idxmax()
        )
        plot_full = df[df["station_id"].astype(str) == plot_station].copy()
        plot_two_week = (
            two_week_df[two_week_df["station_id"].astype(str) == plot_station].copy()
            if "station_id" in two_week_df.columns and not two_week_df.empty
            else two_week_df
        )
    else:
        plot_full = df.copy()
        plot_two_week = two_week_df

    plot_full = plot_full.sort_values("datetime").reset_index(drop=True)
    week = plot_full.iloc[: 7 * 24]
    if len(week):
        plot_time_series(
            week["datetime"].to_numpy(),
            week["GHI_true"].to_numpy(),
            week["GHI_pred"].to_numpy(),
            title=f"{title_prefix} - First Week Measured vs Predicted GHI",
            save_path=str(results_dir / "finetuned_timeseries.png"),
            n_points=len(week),
        )

    # For the two-week plot, replace the metrics-only ``two_week_df`` slice
    # with an unfiltered slice on the same [start, start+14d) window so the
    # plot still shows night periods. ``two_week_ghi_m`` (the printed metric
    # banner inside the plot) is computed on the filtered slice and passed
    # through unchanged.
    if not plot_two_week.empty:
        unfiltered_two_week = plot_full[
            (plot_full["datetime"] >= two_week_start)
            & (plot_full["datetime"] < two_week_start + pd.Timedelta(days=14))
        ]
        if not unfiltered_two_week.empty:
            plot_two_week_comparison(
                unfiltered_two_week["datetime"].to_numpy(),
                unfiltered_two_week["GHI_true"].to_numpy(),
                unfiltered_two_week["GHI_pred"].to_numpy(),
                start=two_week_start,
                window_days=14,
                title=f"{title_prefix} - Two-Week Measured vs Predicted GHI",
                save_path=str(results_dir / "finetuned_two_week_comparison.png"),
                n_points=len(unfiltered_two_week),
                metrics=two_week_ghi_m or None,
                ghi_threshold=validation_ghi_filter_wm2,
            )

    plot_4panel_evaluation(
        df["GHI_true"].to_numpy(),
        df["GHI_pred"].to_numpy(),
        hour_array=df["hour"].to_numpy() if "hour" in df.columns else None,
        title=f"{title_prefix} - Validation GHI Evaluation",
        save_path=str(results_dir / "finetuned_4panel.png"),
        n_points=len(df),
    )

    if horizon_rmse:
        fig, ax = plt.subplots(figsize=(10, 5))
        hours = sorted(horizon_rmse.keys())
        rmses = [horizon_rmse[h] for h in hours]
        ax.plot(hours, rmses, "o-", color="#1f77b4")
        ax.set_xlabel("Forecast Lead Time (h)")
        ax.set_ylabel(f"{target_col} RMSE")
        ax.set_title(f"{title_prefix} - Forecast Degradation Curve")
        ax.grid(True, alpha=0.3)
        fig.savefig(
            results_dir / "finetuned_horizon_rmse.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # ---- Validation report CSV (consumed by export_pdf.build_finetuned_pdf)
    #
    # We deliberately write the *unfiltered* ``df`` here, not ``validation_df``.
    # ``build_finetuned_pdf`` already applies its own ``GHI_true > ghi_filter``
    # internally (only for its metrics tables), and uses the unfiltered rows
    # for the weekly time-series plots and the 4-panel diagnostic. If we wrote
    # the pre-filtered slice the PDF would lose all night-time hours and the
    # weekly plots would look like discontinuous daylight clusters.
    report_cols = ["datetime", "hour", "lead_time_h", "forecast_start"]
    if target_col == "CAF":
        report_cols += ["CAF_true", "CAF_pred", clearsky_col, zenith_col]
    else:
        report_cols += [clearsky_col, zenith_col]
    report_cols += ["GHI_true", "GHI_pred"]
    if "GHI_pred_p10" in df.columns and "GHI_pred_p90" in df.columns:
        report_cols += ["GHI_pred_p10", "GHI_pred_p90"]

    available_cols = [c for c in report_cols if c in df.columns]
    validation_report = df[available_cols].copy()
    if "station_id" in df.columns:
        validation_report["station_id"] = df["station_id"].values
    else:
        validation_report["station_id"] = getattr(
            cfg, "FINETUNE_STATION", station_label_default
        )
    validation_report.to_csv(results_dir / "validation_report_data.csv", index=False)

    # ---- PDF assembled by master_files.export_pdf.build_finetuned_pdf
    build_finetuned_pdf(str(results_dir))

    print(f"\n  Plots saved -> {results_dir}/")
    print(f"  Validation CSV saved -> {results_dir / 'validation_report_data.csv'}")
    print(f"  Report saved -> {results_dir / 'evaluation_report.txt'}")
    print("Done.")


# =====================================================================
# METHOD 1 - CAF + PVLib clear-sky + ERA5 covariates (single station)
# =====================================================================

def run_inference_method1(
    *,
    cfg,
    dataset_dir,
    checkpoint_dir,
    results_dir,
    model_id,
    model_variant: str = "moirai1",
    smoke_test: bool = False,
):
    print("\n=== METHOD 1: CAF (PVLib + ERA5) - Inference ===\n")
    return _run_inference_caf(
        cfg=cfg, dataset_dir=dataset_dir, checkpoint_dir=checkpoint_dir,
        model_id=model_id,
        results_dir=results_dir, target_col_name="CAF",
        model_variant=model_variant, smoke_test=smoke_test,
    )


def build_report_method1(
    *,
    cfg,
    final_csv_dir,
    results_dir,
    validation_ghi_filter_wm2: float = 20.0,
):
    print("\n=== METHOD 1: CAF -> GHI evaluation ===\n")
    return _build_report_generic(
        method=1, cfg=cfg,
        final_csv_dir=final_csv_dir, results_dir=results_dir,
        target_col="CAF",
        clearsky_col="clear_sky_ghi",
        measured_col="w_ghr",
        zenith_col="zenith_angle",
        by_station=False,
        validation_ghi_filter_wm2=validation_ghi_filter_wm2,
        title_prefix="Method 1 (CAF + PVLib + ERA5)",
        station_label_default=getattr(cfg, "FINETUNE_STATION", "unknown"),
    )


# =====================================================================
# METHOD 2 - CAF + NSRDB clear-sky + multi-station
# =====================================================================

def run_inference_method2(
    *,
    cfg,
    dataset_dir,
    checkpoint_dir,
    results_dir,
    model_id,
    model_variant: str = "moirai1",
    smoke_test: bool = False,
):
    print("\n=== METHOD 2: CAF (NSRDB + multi-station) - Inference ===\n")
    return _run_inference_caf(
        cfg=cfg, dataset_dir=dataset_dir, checkpoint_dir=checkpoint_dir,
        model_id=model_id,
        results_dir=results_dir, target_col_name="CAF",
        model_variant=model_variant, smoke_test=smoke_test,
    )


def build_report_method2(
    *,
    cfg,
    final_csv_dir,
    results_dir,
    validation_ghi_filter_wm2: float = 20.0,
):
    print("\n=== METHOD 2: CAF -> GHI (NSRDB clearsky, multi-station) ===\n")
    return _build_report_generic(
        method=2, cfg=cfg,
        final_csv_dir=final_csv_dir, results_dir=results_dir,
        target_col="CAF",
        clearsky_col=getattr(cfg, "CLEARSKY_GHI_COL", "clearsky_ghi"),
        measured_col=getattr(cfg, "MEASURED_GHI_COL", "w_ghr"),
        zenith_col="zenith_angle",
        by_station=True,
        validation_ghi_filter_wm2=validation_ghi_filter_wm2,
        title_prefix="Method 2 (CAF + NSRDB + multi-station)",
    )


# =====================================================================
# METHOD 3 - Direct GHI (multi-station, scaling, daylight mask)
# =====================================================================

def run_inference_method3(
    *,
    cfg,
    dataset_dir,
    checkpoint_dir,
    results_dir,
    model_id,
    model_variant: str = "moirai2",
    smoke_test: bool = False,
):
    print("\n=== METHOD 3: Direct GHI - Inference ===\n")
    return _run_inference_direct_ghi(
        cfg=cfg, dataset_dir=dataset_dir, checkpoint_dir=checkpoint_dir,
        model_id=model_id,
        results_dir=results_dir,
        model_variant=model_variant, smoke_test=smoke_test,
    )


def build_report_method3(
    *,
    cfg,
    final_csv_dir,
    results_dir,
    validation_ghi_filter_wm2: float = 20.0,
):
    print("\n=== METHOD 3: Direct GHI evaluation ===\n")
    return _build_report_generic(
        method=3, cfg=cfg,
        final_csv_dir=final_csv_dir, results_dir=results_dir,
        target_col="GHI",
        clearsky_col=getattr(cfg, "CLEARSKY_GHI_COL", "clearsky_ghi"),
        measured_col=getattr(cfg, "MEASURED_GHI_COL", "w_ghr"),
        zenith_col="zenith_angle",
        by_station=True,
        validation_ghi_filter_wm2=validation_ghi_filter_wm2,
        title_prefix="Method 3 (Direct GHI + multi-station)",
    )
