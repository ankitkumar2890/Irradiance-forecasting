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
    median_idx = _median_quantile_index(module)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)

    # ---- Load validation data ----
    X_past = np.load(DATASET_DIR / "X_past_val.npy")
    X_future = np.load(DATASET_DIR / "X_future_val.npy")
    y_future = np.load(DATASET_DIR / "y_future_val.npy")
    times = np.load(DATASET_DIR / "times_val.npy", allow_pickle=True).astype("datetime64[ns]")

    print(f"Validation windows: {len(X_past)}")

    # ---- Predict ----
    all_preds, all_true, csv_rows = [], [], []

    def forecast_mean(target, feat_dynamic_real):
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
        return _extract_median_point_forecast(preds, median_idx).squeeze(0).detach().cpu().numpy()

    for i in range(len(X_past)):
        target = X_past[i, :, 0].astype(np.float32)
        past_feats = X_past[i, :, 1:].T
        fut_feats = X_future[i].T
        dyn_feats = np.hstack([past_feats, fut_feats]).astype(np.float32)

        forecast_start = pd.Timestamp(times[i, 0])

        pred = forecast_mean(target, dyn_feats)[:PREDICTION_LENGTH]

        all_preds.append(pred)
        all_true.append(y_future[i])

        for h in range(PREDICTION_LENGTH):
            ts = pd.Timestamp(times[i, h])
            csv_rows.append({
                "datetime": ts, "hour": ts.hour,
                "lead_time_h": h + 1,
                "forecast_start": forecast_start,
                "CAF_true": float(y_future[i, h]),
                "CAF_pred": float(pred[h]),
            })

        if i % 20 == 0:
            print(f"  {i}/{len(X_past)}")

    # ---- Save ----
    df_out = pd.DataFrame(csv_rows).sort_values("datetime").reset_index(drop=True)
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
