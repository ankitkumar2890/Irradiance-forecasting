"""Run fine-tuned Moirai 2.0 (base + LoRA adapter) on the validation split."""
import sys, json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    DATASET_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    CONTEXT_LENGTH, PREDICTION_LENGTH, TARGET_DIM, FEAT_DIM,
)

MODEL_ID = "Salesforce/moirai-2.0-R-small"


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


def main():
    # ---- Load base model + LoRA adapter ----
    print(f"Loading {MODEL_ID} + LoRA adapter...")
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from peft import PeftModel

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

    base_module = Moirai2Module.from_pretrained(MODEL_ID, **model_kwargs)
    adapter_path = CHECKPOINT_DIR / "moirai2_lora_adapter"
    module = PeftModel.from_pretrained(base_module, str(adapter_path))
    module.eval()

    model = Moirai2Forecast(
        module=module,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        target_dim=TARGET_DIM,
        feat_dynamic_real_dim=FEAT_DIM,
        past_feat_dynamic_real_dim=0,
    )
    p10_idx = _quantile_index(module, 0.1)
    median_idx = _quantile_index(module, 0.5)
    p90_idx = _quantile_index(module, 0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)

    # ---- Load validation data ----
    X_past = np.load(DATASET_DIR / "X_past_val.npy")
    X_future = np.load(DATASET_DIR / "X_future_val.npy")
    y_future = np.load(DATASET_DIR / "y_future_val.npy")
    times = np.load(DATASET_DIR / "times_val.npy", allow_pickle=True).astype("datetime64[ns]")
    station_ids = np.load(DATASET_DIR / "station_ids_val.npy", allow_pickle=True)

    print(f"Validation windows: {len(X_past)}")
    print(f"Stations: {sorted(set(station_ids.tolist()))}")

    # ---- Predict ----
    all_preds, all_true, csv_rows = [], [], []

    def forecast_quantiles(target, feat_dynamic_real):
        past_target = torch.tensor(
            target, dtype=torch.float32, device=device
        ).view(1, CONTEXT_LENGTH, TARGET_DIM)
        past_observed = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros((1, CONTEXT_LENGTH), dtype=torch.bool, device=device)
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
        return {
            "p10": _extract_quantile_forecast(preds, p10_idx).squeeze(0).detach().cpu().numpy(),
            "p50": _extract_quantile_forecast(preds, median_idx).squeeze(0).detach().cpu().numpy(),
            "p90": _extract_quantile_forecast(preds, p90_idx).squeeze(0).detach().cpu().numpy(),
        }

    for i in range(len(X_past)):
        target = X_past[i, :, 0].astype(np.float32)
        past_feats = X_past[i, :, 1:].T
        fut_feats = X_future[i].T
        dyn_feats = np.hstack([past_feats, fut_feats]).astype(np.float32)

        forecast_start = pd.Timestamp(times[i, 0])

        pred_quantiles = forecast_quantiles(target, dyn_feats)
        pred = pred_quantiles["p50"][:PREDICTION_LENGTH]
        pred_p10 = pred_quantiles["p10"][:PREDICTION_LENGTH]
        pred_p90 = pred_quantiles["p90"][:PREDICTION_LENGTH]

        all_preds.append(pred)
        all_true.append(y_future[i])

        for h in range(PREDICTION_LENGTH):
            ts = pd.Timestamp(times[i, h])
            csv_rows.append({
                "station_id": str(station_ids[i]),
                "datetime": ts, "hour": ts.hour,
                "lead_time_h": h + 1,
                "forecast_start": forecast_start,
                "CAF_true": float(y_future[i, h]),
                "CAF_p10": float(pred_p10[h]),
                "CAF_pred": float(pred[h]),
                "CAF_p90": float(pred_p90[h]),
            })

        if i % 20 == 0:
            print(f"  {i}/{len(X_past)}")

    # ---- Save ----
    df_out = (
        pd.DataFrame(csv_rows)
        .sort_values(["station_id", "datetime", "lead_time_h"])
        .reset_index(drop=True)
    )
    df_out.to_csv(RESULTS_DIR / "finetuned_predictions.csv", index=False)
    df_out.to_csv(RESULTS_DIR / "finetuned_predictions_moirai2.csv", index=False)

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    rmse = np.sqrt(np.mean((all_preds - all_true) ** 2))
    mae = np.mean(np.abs(all_preds - all_true))
    print(f"\n  Fine-tuned Validation — CAF RMSE: {rmse:.4f}  MAE: {mae:.4f}")

    metrics = {"CAF_RMSE": float(rmse), "CAF_MAE": float(mae)}
    with open(RESULTS_DIR / "finetuned_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(RESULTS_DIR / "finetuned_metrics_moirai2.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved → results/finetuned_predictions.csv + finetuned_metrics.json")
    print(f"  Saved → results/finetuned_predictions_moirai2.csv + finetuned_metrics_moirai2.json")


if __name__ == "__main__":
    main()
