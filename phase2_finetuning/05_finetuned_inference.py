"""Run fine-tuned Moirai (base + LoRA adapter) on the test set."""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    MODEL_ID, CONTEXT_LENGTH, PREDICTION_LENGTH, TARGET_DIM, FEAT_DIM,
)


def main():
    # ---- Load base model + LoRA adapter ----
    print(f"Loading {MODEL_ID} + LoRA adapter...")
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from peft import PeftModel

    base_module = MoiraiModule.from_pretrained(MODEL_ID)
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
    predictor = model.create_predictor(batch_size=1)

    # ---- Load test data ----
    X_past = np.load(DATASET_DIR / "X_past_test.npy")
    X_future = np.load(DATASET_DIR / "X_future_test.npy")
    y_future = np.load(DATASET_DIR / "y_future_test.npy")
    times = np.load(DATASET_DIR / "times_test.npy", allow_pickle=True).astype("datetime64[ns]")

    print(f"Test windows: {len(X_past)}")

    # ---- Predict ----
    from gluonts.dataset.common import ListDataset

    all_preds, all_true, csv_rows = [], [], []

    for i in range(len(X_past)):
        target = X_past[i, :, 0].astype(np.float32)
        past_feats = X_past[i, :, 1:].T
        fut_feats = X_future[i].T
        dyn_feats = np.hstack([past_feats, fut_feats]).astype(np.float32)

        forecast_start = pd.Timestamp(times[i, 0])
        context_start = forecast_start - pd.Timedelta(hours=CONTEXT_LENGTH)

        item = {
            "start": pd.Period(context_start, freq="h"),
            "target": target,
            "feat_dynamic_real": dyn_feats,
        }
        ds = ListDataset([item], freq="h")
        fc = next(iter(predictor.predict(ds)))
        pred = np.asarray(fc.quantile(0.5), dtype=np.float32)[:PREDICTION_LENGTH]

        all_preds.append(pred)
        all_true.append(y_future[i])

        for h in range(PREDICTION_LENGTH):
            ts = pd.Timestamp(times[i, h])
            csv_rows.append({
                "datetime": ts, "hour": ts.hour,
                "CAF_true": float(y_future[i, h]),
                "CAF_pred": float(pred[h]),
            })

        if i % 20 == 0:
            print(f"  {i}/{len(X_past)}")

    # ---- Save ----
    df_out = pd.DataFrame(csv_rows).sort_values("datetime").reset_index(drop=True)
    df_out.to_csv(RESULTS_DIR / "finetuned_predictions.csv", index=False)

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    rmse = np.sqrt(np.mean((all_preds - all_true) ** 2))
    mae = np.mean(np.abs(all_preds - all_true))
    print(f"\n  Fine-tuned Test — CAF RMSE: {rmse:.4f}  MAE: {mae:.4f}")

    metrics = {"CAF_RMSE": float(rmse), "CAF_MAE": float(mae)}
    with open(RESULTS_DIR / "finetuned_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved → results/finetuned_predictions.csv + finetuned_metrics.json")


if __name__ == "__main__":
    main()
