"""Run fine-tuned Moirai 2.0 (base + LoRA) on the validation split — Direct GHI.

Phase 3: Predictions are GHI in W/m² directly — no CAF recovery step needed.
"""
import sys, json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    DATASET_DIR, CHECKPOINT_DIR, RESULTS_DIR, CLUSTER_NAME,
    CONTEXT_LENGTH, PREDICTION_LENGTH, TARGET_DIM, FEAT_DIM,
    GHI_SCALE_FACTOR, MODEL_ID,
    DAYLIGHT_ZENITH_DEG, NIGHT_ZENITH_DEG,
    PAST_FEATURES, FUTURE_FEATURES,
)


def _get_quantile_levels(module) -> list[float]:
    """Find quantile levels even when the model is wrapped by PEFT/container modules."""
    to_visit = [module]
    seen = set()

    while to_visit:
        current = to_visit.pop(0)
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        quantiles = getattr(current, "quantile_levels", None)
        if quantiles is not None:
            return [float(q) for q in quantiles]

        for attr in ("model", "module", "base_model"):
            nested = getattr(current, attr, None)
            if nested is not None:
                to_visit.append(nested)

    return [0.5]


def _quantile_index(module, target_quantile: float) -> int:
    quantiles = _get_quantile_levels(module)
    if target_quantile in quantiles:
        return quantiles.index(target_quantile)
    return min(range(len(quantiles)), key=lambda idx: abs(float(quantiles[idx]) - target_quantile))


def _extract_quantile_forecast(preds: torch.Tensor, quantile_idx: int) -> torch.Tensor:
    """Return one forecast quantile as a [batch, horizon] tensor."""
    if preds.ndim == 4:
        return preds[:, quantile_idx, :, 0]
    if preds.ndim == 3:
        return preds[:, quantile_idx, :]
    raise RuntimeError(f"Unexpected Moirai2 forecast shape: {tuple(preds.shape)}")


def _build_daylight_mask(future_zenith_deg: np.ndarray) -> np.ndarray:
    """Return a daylight attenuation mask for the forecast horizon."""
    daylight_span = max(NIGHT_ZENITH_DEG - DAYLIGHT_ZENITH_DEG, 1e-6)
    mask = (NIGHT_ZENITH_DEG - future_zenith_deg.astype(np.float32)) / daylight_span
    return np.clip(mask, 0.0, 1.0)


def main():
    # ── Load base model + LoRA adapter ──
    print(f"Loading {MODEL_ID} + LoRA adapter (Direct GHI, cluster={CLUSTER_NAME})...")
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from peft import PeftModel

    try:
        config_path = hf_hub_download(MODEL_ID, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            model_kwargs = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Could not load model config for {MODEL_ID}."
        ) from e

    if isinstance(model_kwargs.get("quantile_levels"), list):
        model_kwargs["quantile_levels"] = tuple(model_kwargs["quantile_levels"])

    base_module = Moirai2Module.from_pretrained(MODEL_ID, **model_kwargs)
    adapter_path = CHECKPOINT_DIR / f"direct_ghi_lora_{CLUSTER_NAME}"
    module = PeftModel.from_pretrained(base_module, str(adapter_path))
    module.eval()

    dyn_indices = [PAST_FEATURES.index(f) for f in FUTURE_FEATURES]
    past_only_indices = [
        i for i, f in enumerate(PAST_FEATURES)
        if f != "w_ghr" and f not in FUTURE_FEATURES
    ]
    past_only_dim = len(past_only_indices)

    model = Moirai2Forecast(
        module=module,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        target_dim=TARGET_DIM,
        feat_dynamic_real_dim=FEAT_DIM,
        past_feat_dynamic_real_dim=past_only_dim,
    )
    p10_idx = _quantile_index(module, 0.1)
    median_idx = _quantile_index(module, 0.5)
    p90_idx = _quantile_index(module, 0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)

    # ── Load validation data ──
    X_past = np.load(DATASET_DIR / "X_past_val.npy")
    X_future = np.load(DATASET_DIR / "X_future_val.npy")
    y_future = np.load(DATASET_DIR / "y_future_val.npy")  # scaled GHI
    times = np.load(DATASET_DIR / "times_val.npy", allow_pickle=True).astype("datetime64[ns]")
    station_ids = np.load(DATASET_DIR / "station_ids_val.npy", allow_pickle=True)

    print(f"Validation windows: {len(X_past)}")
    print(f"Stations: {sorted(set(station_ids.tolist()))}")
    print(f"GHI_SCALE_FACTOR: {GHI_SCALE_FACTOR}")
    print(
        f"Daylight mask: full<= {DAYLIGHT_ZENITH_DEG:.1f}°, "
        f"zero>= {NIGHT_ZENITH_DEG:.1f}°"
    )

    # ── Predict ──
    all_preds, all_true, csv_rows = [], [], []

    def forecast_quantiles(target, feat_dynamic_real, past_feat_dynamic_real=None):
        past_target = torch.tensor(
            target, dtype=torch.float32, device=device
        ).view(1, CONTEXT_LENGTH, TARGET_DIM)
        past_observed = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros((1, CONTEXT_LENGTH), dtype=torch.bool, device=device)
        feat_dynamic = torch.tensor(
            feat_dynamic_real.T, dtype=torch.float32, device=device
        ).unsqueeze(0)
        observed_feat = torch.ones_like(feat_dynamic, dtype=torch.bool)

        model_kwargs = {
            "feat_dynamic_real": feat_dynamic,
            "observed_feat_dynamic_real": observed_feat,
        }
        if past_feat_dynamic_real is not None:
            past_feat_dynamic = torch.tensor(
                past_feat_dynamic_real, dtype=torch.float32, device=device
            ).unsqueeze(0)
            observed_past_feat = torch.ones_like(past_feat_dynamic, dtype=torch.bool)
            model_kwargs["past_feat_dynamic_real"] = past_feat_dynamic
            model_kwargs["past_observed_feat_dynamic_real"] = observed_past_feat

        preds = model(
            past_target,
            past_observed,
            past_is_pad,
            **model_kwargs,
        )
        return {
            "p10": _extract_quantile_forecast(preds, p10_idx).squeeze(0).detach().cpu().numpy(),
            "p50": _extract_quantile_forecast(preds, median_idx).squeeze(0).detach().cpu().numpy(),
            "p90": _extract_quantile_forecast(preds, p90_idx).squeeze(0).detach().cpu().numpy(),
        }

    for i in range(len(X_past)):
        # First column of X_past is w_ghr — scale it for the model
        target = (X_past[i, :, 0] / GHI_SCALE_FACTOR).astype(np.float32)
        past_feats = X_past[i][:, dyn_indices].T
        fut_feats = X_future[i].T
        dyn_feats = np.hstack([past_feats, fut_feats]).astype(np.float32)
        past_only_feats = None
        if past_only_indices:
            past_only_feats = X_past[i][:, past_only_indices].astype(np.float32)
        future_zenith = X_future[i, :, 0].astype(np.float32)
        daylight_mask = _build_daylight_mask(future_zenith)

        forecast_start = pd.Timestamp(times[i, 0]) - pd.Timedelta(hours=1)

        pred_quantiles = forecast_quantiles(target, dyn_feats, past_only_feats)

        # Scale predictions back to W/m²
        pred = pred_quantiles["p50"][:PREDICTION_LENGTH] * GHI_SCALE_FACTOR * daylight_mask
        pred_p10 = pred_quantiles["p10"][:PREDICTION_LENGTH] * GHI_SCALE_FACTOR * daylight_mask
        pred_p90 = pred_quantiles["p90"][:PREDICTION_LENGTH] * GHI_SCALE_FACTOR * daylight_mask

        # y_future is also scaled — convert back to W/m²
        y_true_wm2 = y_future[i] * GHI_SCALE_FACTOR

        all_preds.append(pred)
        all_true.append(y_true_wm2)

        for h in range(PREDICTION_LENGTH):
            ts = pd.Timestamp(times[i, h])
            lead_time_h = int((ts - forecast_start) / pd.Timedelta(hours=1))
            csv_rows.append({
                "station_id": str(station_ids[i]),
                "datetime": ts,
                "hour": ts.hour,
                "lead_time_h": lead_time_h,
                "forecast_start": forecast_start,
                "GHI_true": float(y_true_wm2[h]),
                "GHI_p10": float(pred_p10[h]),
                "GHI_pred": float(pred[h]),
                "GHI_p90": float(pred_p90[h]),
            })

        if i % 20 == 0:
            print(f"  {i}/{len(X_past)}")

    # ── Save ──
    df_out = (
        pd.DataFrame(csv_rows)
        .sort_values(["station_id", "datetime", "lead_time_h"])
        .reset_index(drop=True)
    )
    df_out.to_csv(RESULTS_DIR / "direct_ghi_predictions.csv", index=False)

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    rmse = np.sqrt(np.mean((all_preds - all_true) ** 2))
    mae = np.mean(np.abs(all_preds - all_true))
    mean_true = np.mean(all_true[all_true > 0])
    nrmse = (rmse / mean_true * 100) if mean_true > 0 else float("nan")

    print(f"\n  Direct GHI Validation — RMSE: {rmse:.2f} W/m²  MAE: {mae:.2f} W/m²  nRMSE: {nrmse:.1f}%")

    metrics = {
        "GHI_RMSE_wm2": float(rmse),
        "GHI_MAE_wm2": float(mae),
        "GHI_nRMSE_pct": float(nrmse),
        "cluster": CLUSTER_NAME,
        "ghi_scale_factor": GHI_SCALE_FACTOR,
    }
    with open(RESULTS_DIR / "direct_ghi_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved → results/direct_ghi_predictions.csv")
    print(f"  Saved → results/direct_ghi_metrics.json")


if __name__ == "__main__":
    main()
