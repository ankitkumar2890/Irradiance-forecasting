"""
03_build_dataset.py — Temporal split → sliding window .npy + GluonTS Arrow for ERA5-direct fine-tuning.

Split (3 years):
  Train: Jan 2017 – Dec 2018
  Val:   Jan 2019 – Dec 2019
  Test:  reserved / empty unless later data is added

Two output formats:
  Format A: .npy sliding windows (for direct PyTorch inference)
  Format B: GluonTS Arrow datasets (for uni2ts fine-tuning CLI)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_DIR, ARROW_DIR,
    TRAIN_END, VAL_START, VAL_END, TEST_START,
    PAST_HOURS, FUTURE_HOURS, PAST_FEATURES, FUTURE_FEATURES,
    FINETUNE_STATION,
)


def build_windows(df, split_name):
    """Build sliding windows anchored at 06:00 on the normalized local clock."""
    X_past, X_future, y_future, times = [], [], [], []

    for i in range(len(df) - PAST_HOURS - FUTURE_HOURS + 1):
        past = df.iloc[i : i + PAST_HOURS]
        future = df.iloc[i + PAST_HOURS : i + PAST_HOURS + FUTURE_HOURS]

        if future.iloc[0]["datetime"].hour != 6:
            continue
        if past["datetime"].dt.date.nunique() < 3:
            continue

        X_past.append(past[PAST_FEATURES].values)
        X_future.append(future[FUTURE_FEATURES].values)
        y_future.append(future["CAF"].values)
        times.append(future["datetime"].values)

    X_past = np.array(X_past, dtype=np.float32)
    X_future = np.array(X_future, dtype=np.float32)
    y_future = np.array(y_future, dtype=np.float32)
    times = np.array(times, dtype="datetime64[ns]")

    for name, arr in [("X_past", X_past), ("X_future", X_future),
                       ("y_future", y_future), ("times", times)]:
        np.save(DATASET_DIR / f"{name}_{split_name}.npy", arr)

    print(f"  {split_name}: {len(X_past)} windows  "
          f"X_past={X_past.shape}  X_future={X_future.shape}")
    return X_past, X_future, y_future


def build_arrow(df, split_name):
    """Build GluonTS-compatible Arrow dataset for uni2ts fine-tuning."""
    from datasets import Dataset, Features, Sequence, Value

    target = df["CAF"].values.astype(np.float32)
    covariates = df[FUTURE_FEATURES].values.astype(np.float32).T.tolist()
    start_str = str(pd.Period(df["datetime"].iloc[0], freq="h"))

    sample = {
        "start": [start_str],
        "target": [target.tolist()],
        "feat_dynamic_real": [covariates],
        "freq": ["h"],
        "item_id": [f"{FINETUNE_STATION}_{split_name}"],
    }

    features = Features({
        "start": Value("string"),
        "target": Sequence(Value("float32")),
        "feat_dynamic_real": Sequence(Sequence(Value("float32"))),
        "freq": Value("string"),
        "item_id": Value("string"),
    })

    ds = Dataset.from_dict(sample, features=features)
    out_path = ARROW_DIR / split_name
    ds.save_to_disk(str(out_path))
    print(f"  Arrow {split_name}: {len(target)} timesteps → {out_path}")


def main():
    processed_file = DATASET_DIR / "processed_data_2017_2019.csv"
    df = pd.read_csv(processed_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    print(f"Loaded {processed_file.name}: {len(df)} rows")
    print(f"  Range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")

    # Temporal split
    train_df = df[df["datetime"] <= TRAIN_END].reset_index(drop=True)
    val_df = df[(df["datetime"] >= VAL_START) &
                (df["datetime"] <= VAL_END)].reset_index(drop=True)
    test_df = df[df["datetime"] >= TEST_START].reset_index(drop=True)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} rows  "
          f"({train_df['datetime'].iloc[0]} → {train_df['datetime'].iloc[-1]})")
    print(f"  Val:   {len(val_df)} rows  "
          f"({val_df['datetime'].iloc[0]} → {val_df['datetime'].iloc[-1]})")
    if len(test_df) > 0:
        print(f"  Test:  {len(test_df)} rows  "
              f"({test_df['datetime'].iloc[0]} → {test_df['datetime'].iloc[-1]})")
    else:
        print("  Test:  0 rows  (no held-out test range configured)")

    # Sliding windows (.npy)
    print("\n--- Sliding windows ---")
    build_windows(train_df, "train")
    build_windows(val_df, "val")
    build_windows(test_df, "test")

    # Arrow format for uni2ts
    print("\n--- Arrow datasets ---")
    try:
        build_arrow(train_df, "train")
        build_arrow(val_df, "val")
        if len(test_df) > 0:
            build_arrow(test_df, "test")
    except ImportError:
        print("  'datasets' package not installed — skipping Arrow.")
        print("  pip install datasets")

    print("\nDone. Next: python 04_finetune.py")


if __name__ == "__main__":
    main()
