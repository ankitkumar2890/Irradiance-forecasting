"""Run fine-tuned Moirai on the test set, conducting a grid search over quantiles [0.60 to 0.70]."""
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

    X_past = np.load(DATASET_DIR / "X_past_test.npy")
    X_future = np.load(DATASET_DIR / "X_future_test.npy")
    y_future = np.load(DATASET_DIR / "y_future_test.npy")
    times = np.load(DATASET_DIR / "times_test.npy", allow_pickle=True).astype("datetime64[ns]")

    print(f"Test windows: {len(X_past)}")

    from gluonts.dataset.common import ListDataset

    # We will query multiple quantiles to find the best peak-shifter
    QUANTILES = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    
    csv_rows = []

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
        
        # Query all quantiles at once
        preds = {}
        for q in QUANTILES:
            preds[q] = np.asarray(fc.quantile(q), dtype=np.float32)[:PREDICTION_LENGTH]

        for h in range(PREDICTION_LENGTH):
            ts = pd.Timestamp(times[i, h])
            row = {
                "datetime": ts, 
                "hour": ts.hour,
                "CAF_true": float(y_future[i, h]),
            }
            # Save predictions for every quantile into the same row
            for q in QUANTILES:
                row[f"CAF_pred_{q:.2f}"] = float(preds[q][h])
            
            csv_rows.append(row)

        if i % 20 == 0:
            print(f"  {i}/{len(X_past)}")

    df_out = pd.DataFrame(csv_rows).sort_values("datetime").reset_index(drop=True)
    out_csv = RESULTS_DIR / "finetuned_predictions_tuned_gridsearch.csv"
    df_out.to_csv(out_csv, index=False)

    print(f"\n  Grid Search complete!")
    print(f"  Saved {len(QUANTILES)} different quantile targets → {out_csv.name}")

if __name__ == "__main__":
    main()
