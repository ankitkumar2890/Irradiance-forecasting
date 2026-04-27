"""Build TFT sliding-window datasets from the TFT-local processed ERA5 feature file."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_DIR,
    FUTURE_FEATURES,
    FUTURE_STEPS,
    PAST_FEATURES,
    PAST_STEPS,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)


def build_windows(df, split_name):
    """Build 72h-to-24h windows grouped by station and anchored at 06:00 local time."""
    x_past, x_future, y_future, times, station_ids = [], [], [], [], []

    for station_id, station_df in df.groupby("station_id", sort=True):
        station_df = station_df.sort_values("datetime").reset_index(drop=True)
        for i in range(len(station_df) - PAST_STEPS - FUTURE_STEPS + 1):
            past = station_df.iloc[i : i + PAST_STEPS]
            future = station_df.iloc[i + PAST_STEPS : i + PAST_STEPS + FUTURE_STEPS]

            if future.iloc[0]["datetime"].hour != 6:
                continue
            if past["datetime"].dt.date.nunique() < 3:
                continue

            x_past.append(past[PAST_FEATURES].values)
            x_future.append(future[FUTURE_FEATURES].values)
            y_future.append(future["CAF"].values)
            times.append(future["datetime"].values)
            station_ids.append(station_id)

    x_past = np.array(x_past, dtype=np.float32)
    x_future = np.array(x_future, dtype=np.float32)
    y_future = np.array(y_future, dtype=np.float32)
    times = np.array(times, dtype="datetime64[ns]")
    station_ids = np.array(station_ids, dtype=str)

    for name, arr in [
        ("X_past", x_past),
        ("X_future", x_future),
        ("y_future", y_future),
        ("times", times),
        ("station_ids", station_ids),
    ]:
        np.save(DATASET_DIR / f"{name}_{split_name}.npy", arr)

    print(
        f"  {split_name}: {len(x_past)} windows  "
        f"X_past={x_past.shape}  X_future={x_future.shape}"
    )


def main():
    processed_file = DATASET_DIR / "processed_data_2017_2019.csv"
    df = pd.read_csv(processed_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    print(f"Loaded {processed_file.name}: {len(df)} rows")
    print(f"  Stations: {df['station_id'].nunique()} {sorted(df['station_id'].unique().tolist())}")
    print(f"  Range: {df['datetime'].min()} → {df['datetime'].max()}")

    train_df = df[df["datetime"] <= TRAIN_END].reset_index(drop=True)
    val_df = df[(df["datetime"] >= VAL_START) & (df["datetime"] <= VAL_END)].reset_index(drop=True)
    test_df = df[df["datetime"] >= TEST_START].reset_index(drop=True)

    print("\nSplit sizes:")
    print(f"  Train: {len(train_df)} rows ({train_df['datetime'].min()} → {train_df['datetime'].max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['datetime'].min()} → {val_df['datetime'].max()})")
    if len(test_df) > 0:
        print(f"  Test:  {len(test_df)} rows ({test_df['datetime'].min()} → {test_df['datetime'].max()})")
    else:
        print("  Test:  0 rows (no held-out test range configured)")

    print("\n--- Sliding windows ---")
    build_windows(train_df, "train")
    build_windows(val_df, "val")
    build_windows(test_df, "test")

    print("\nDone. Next: python tft_model/train_tft.py")


if __name__ == "__main__":
    main()
