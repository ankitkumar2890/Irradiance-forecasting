# ==============================================================================
# generate_synthetic.py — Apply frozen CloudMapper v6 to ERA5 2017–2019
#
# Generates synthetic ICON-like cloud cover from ERA5 data using the trained
# Residual MLP model. Includes isotonic calibration post-processing.
#
# Output: downloads/icon_synthetic_2017_2019.csv
# Run AFTER train_mapper.py:
#   python generate_synthetic.py
# ==============================================================================
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DOWNLOADS_DIR, CHECKPOINT_DIR,
    ERA5_FRACTION_COLS, ICON_COLS,
    HIDDEN_DIM, NUM_RES_BLOCKS, DROPOUT,
    TRAIN_STATION_ID, VERSION,
)
from features import build_all_features
from model_architecture import TAFResNet


def load_frozen_mapper(device: torch.device, input_size: int) -> TAFResNet:
    ckpt = CHECKPOINT_DIR / f"cloud_mapper_{VERSION}_frozen.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}\nRun train_mapper.py first.")

    mapper = TAFResNet(
        input_size=input_size,
        hidden_dim=HIDDEN_DIM,
        num_res_blocks=NUM_RES_BLOCKS,
        output_size=len(ICON_COLS),  # 1
        dropout=DROPOUT,
    )
    mapper.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    mapper.freeze()
    mapper.to(device)
    return mapper


def generate_synthetic(years_label: str = "2017_2019") -> None:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # Load ERA5
    era5 = pd.read_csv(DOWNLOADS_DIR / f"era5_{years_label}.csv")
    era5["datetime"] = pd.to_datetime(era5["datetime"], utc=True)
    if TRAIN_STATION_ID and "station_id" in era5.columns:
        era5 = era5[era5["station_id"] == TRAIN_STATION_ID].copy()
        print(f"  ERA5 station filter: {TRAIN_STATION_ID} → {len(era5)} rows")
    elif "station_id" in era5.columns:
        print(f"  ERA5 stations used: {sorted(era5['station_id'].dropna().unique())}")

    # Feature engineering (same pipeline as training)
    era5, input_cols = build_all_features(era5)

    n_before = len(era5)
    era5 = era5.dropna(subset=input_cols).sort_values("datetime").reset_index(drop=True)
    if n_before - len(era5) > 0:
        print(f"  Dropped {n_before - len(era5)} NaN rows")
    print(f"  ERA5 rows: {len(era5)}")
    print(f"  Features: {input_cols} ({len(input_cols)} total)")

    # Load model + scaler
    mapper = load_frozen_mapper(device, input_size=len(input_cols))
    scaler = joblib.load(CHECKPOINT_DIR / f"era5_scaler_{VERSION}.pkl")
    print(f"  Loaded frozen mapper ({mapper.param_count():,} params) + scaler")

    # Load isotonic calibrators if available
    iso_path = CHECKPOINT_DIR / f"isotonic_calibrators_{VERSION}.pkl"
    calibrators = None
    if iso_path.exists():
        calibrators = joblib.load(iso_path)
        print(f"  Loaded isotonic calibrators ({len(calibrators)} variables)")
    else:
        print(f"  No isotonic calibrators found — skipping calibration")

    # Scale features
    features_scaled = scaler.transform(era5[input_cols].values.astype(np.float32))
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    # Raw ERA5 total_cloud_cover for residual skip (unscaled)
    era5_raw = torch.tensor(
        era5[["total_cloud_cover"]].values.astype(np.float32),
        dtype=torch.float32,
    )

    # Batch inference
    N = len(era5)
    INFER_BATCH = 4096
    all_preds = []
    mapper.eval()
    with torch.no_grad():
        for s in range(0, N, INFER_BATCH):
            batch_feat = features_tensor[s:s + INFER_BATCH].to(device)
            batch_raw = era5_raw[s:s + INFER_BATCH].to(device)
            pred = mapper(batch_feat, era5_raw=batch_raw)
            all_preds.append(pred.cpu().numpy())

    synth = np.concatenate(all_preds).flatten()

    # Apply isotonic calibration if available
    if calibrators is not None:
        print(f"  Applying isotonic calibration...")
        synth_2d = synth.reshape(-1, 1)
        for idx, iso in enumerate(calibrators):
            before_mean = synth_2d[:, idx].mean()
            synth_2d[:, idx] = iso.predict(synth_2d[:, idx])
            after_mean = synth_2d[:, idx].mean()
            print(f"    cloud_cover    mean: {before_mean:.4f} → {after_mean:.4f}")
        synth = synth_2d.flatten()

    # Build output dataframe
    synth_df = pd.DataFrame({"cloud_cover": synth})
    synth_df["datetime"] = era5["datetime"].values
    if "station_id" in era5.columns:
        synth_df["station_id"] = era5["station_id"].values
        synth_df = synth_df[["datetime", "station_id", "cloud_cover"]]
    else:
        synth_df = synth_df[["datetime", "cloud_cover"]]

    out_path = DOWNLOADS_DIR / f"icon_synthetic_{years_label}.csv"
    synth_df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}  shape={synth_df.shape}")

    print(f"\n  Synthetic statistics:")
    print(f"    cloud_cover    mean={synth_df['cloud_cover'].mean():.3f}  "
          f"std={synth_df['cloud_cover'].std():.3f}")


if __name__ == "__main__":
    print("=" * 55)
    print(f"  Phase 1: Generating Synthetic ICON (2017-2019) — {VERSION}")
    print("=" * 55)
    generate_synthetic()
    print("\nDone. Run feature alignment (Cell 9) next.")
