"""Run fine-tuned Moirai (base + LoRA adapter) on the validation split."""
import sys, json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_DIR, CHECKPOINT_DIR, RESULTS_DIR,
MODEL_ID, CONTEXT_LENGTH, PREDICTION_LENGTH, TARGET_DIM, FEAT_DIM,
)

QUANTILE_SAMPLE_COUNT = 256


def _formatted_to_sample_horizon(formatted: torch.Tensor) -> torch.Tensor:
    """Normalize formatted forecast tensors to [samples, horizon]."""
    if formatted.ndim == 4:
        return formatted[:, 0, :, 0]
    if formatted.ndim == 3:
        return formatted[:, 0, :]
    if formatted.ndim == 2:
        return formatted
    raise RuntimeError(f"Unexpected formatted prediction shape: {tuple(formatted.shape)}")


def main():
    # ---- Load base model + LoRA adapter ----
    print(f"Loading {MODEL_ID} + LoRA adapter...")
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
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

    if isinstance(model_kwargs.get("distr_output"), dict):
        model_kwargs["distr_output"] = instantiate(model_kwargs["distr_output"], _convert_="all")
    if isinstance(model_kwargs.get("patch_sizes"), list):
        model_kwargs["patch_sizes"] = tuple(model_kwargs["patch_sizes"])

    base_module = MoiraiModule.from_pretrained(MODEL_ID, **model_kwargs)
    adapter_path = CHECKPOINT_DIR / "moirai_lora_adapter"
    module = PeftModel.from_pretrained(base_module, str(adapter_path))
    module.eval()

    model = MoiraiForecast(
        module=module,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        target_dim=TARGET_DIM,
        feat_dynamic_real_dim=FEAT_DIM,
        past_feat_dynamic_real_dim=0,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
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

        distr = model._get_distr(
            16,
            past_target,
            past_observed,
            past_is_pad,
            feat_dynamic,
            observed_feat,
        )
        formatted_mean = model._format_preds(16, distr.mean.unsqueeze(0), TARGET_DIM)
        median = _formatted_to_sample_horizon(formatted_mean).squeeze(0)

        try:
            q10_raw = distr.icdf(torch.full_like(distr.mean, 0.1))
            q90_raw = distr.icdf(torch.full_like(distr.mean, 0.9))
            formatted_q10 = model._format_preds(16, q10_raw.unsqueeze(0), TARGET_DIM)
            formatted_q90 = model._format_preds(16, q90_raw.unsqueeze(0), TARGET_DIM)
            p10 = _formatted_to_sample_horizon(formatted_q10).squeeze(0)
            p90 = _formatted_to_sample_horizon(formatted_q90).squeeze(0)
        except Exception:
            samples = distr.sample((QUANTILE_SAMPLE_COUNT,))
            formatted_samples = model._format_preds(QUANTILE_SAMPLE_COUNT, samples, TARGET_DIM)
            sample_horizon = _formatted_to_sample_horizon(formatted_samples)
            p10 = torch.quantile(sample_horizon, 0.1, dim=0)
            p90 = torch.quantile(sample_horizon, 0.9, dim=0)

        return {
            "p10": p10.detach().cpu().numpy(),
            "p50": median.detach().cpu().numpy(),
            "p90": p90.detach().cpu().numpy(),
        }

    for i in range(len(X_past)):
        target = X_past[i, :, 0].astype(np.float32)
        past_feats = X_past[i, :, 1:].T
        fut_feats = X_future[i].T
        dyn_feats = np.hstack([past_feats, fut_feats]).astype(np.float32)

        forecast_start = pd.Timestamp(times[i, 0])
        context_start = forecast_start - pd.Timedelta(hours=CONTEXT_LENGTH)

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

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    rmse = np.sqrt(np.mean((all_preds - all_true) ** 2))
    mae = np.mean(np.abs(all_preds - all_true))
    print(f"\n  Fine-tuned Validation — CAF RMSE: {rmse:.4f}  MAE: {mae:.4f}")

    metrics = {"CAF_RMSE": float(rmse), "CAF_MAE": float(mae)}
    with open(RESULTS_DIR / "finetuned_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved → results/finetuned_predictions.csv + finetuned_metrics.json")


if __name__ == "__main__":
    main()
