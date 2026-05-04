"""Dataset building for the standalone Chronos Method 3 pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def load_final_csv(*, final_csv_dir: str | Path) -> pd.DataFrame:
    final_csv_dir = Path(final_csv_dir)
    csvs = sorted(p for p in final_csv_dir.glob("*.csv") if not p.name.startswith("."))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {final_csv_dir}")
    if len(csvs) > 1:
        print(f"  Multiple CSVs found in {final_csv_dir}; using {csvs[0].name}")

    df = pd.read_csv(csvs[0], parse_dates=["datetime"])
    print(f"Loaded {csvs[0].name}: {len(df)} rows")
    print(f"  Range: {df['datetime'].min()} -> {df['datetime'].max()}")
    return df


def temporal_split(df: pd.DataFrame, *, train_end, val_start, val_end, test_start):
    train_df = df[df["datetime"] <= train_end].reset_index(drop=True)
    val_df = df[(df["datetime"] >= val_start) & (df["datetime"] <= val_end)].reset_index(drop=True)
    test_df = df[df["datetime"] >= test_start].reset_index(drop=True)
    return train_df, val_df, test_df


def _normalize_anchor_hours(anchor_hours) -> Optional[list[int]]:
    if anchor_hours is None:
        return None
    if isinstance(anchor_hours, int):
        return [int(anchor_hours)]
    return sorted({int(h) for h in anchor_hours})


def build_windows(
    df: pd.DataFrame,
    split_name: str,
    *,
    dataset_dir: str | Path,
    target_col: str,
    past_hours: int,
    future_hours: int,
    station_col: str,
    anchor_hours: Optional[Sequence[int]] = None,
    min_past_dates: int = 1,
):
    anchor_hours = _normalize_anchor_hours(anchor_hours)

    contexts = []
    targets = []
    times = []
    station_ids = []
    n_total = 0
    n_dropped_anchor = 0
    n_dropped_past_dates = 0

    for station_id, station_df in df.groupby(station_col, sort=True):
        station_df = station_df.sort_values("datetime").reset_index(drop=True)
        candidate_windows = max(len(station_df) - past_hours - future_hours + 1, 0)
        n_total += candidate_windows

        for i in range(candidate_windows):
            past = station_df.iloc[i : i + past_hours]
            future = station_df.iloc[i + past_hours : i + past_hours + future_hours]

            if anchor_hours is not None and int(future.iloc[0]["datetime"].hour) not in anchor_hours:
                n_dropped_anchor += 1
                continue
            if min_past_dates > 1 and past["datetime"].dt.date.nunique() < min_past_dates:
                n_dropped_past_dates += 1
                continue

            contexts.append(past[target_col].to_numpy(dtype=np.float32))
            targets.append(future[target_col].to_numpy(dtype=np.float32))
            times.append(future["datetime"].to_numpy(dtype="datetime64[ns]"))
            station_ids.append(str(station_id))

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if contexts:
        X_past = np.asarray(contexts, dtype=np.float32)
        y_future = np.asarray(targets, dtype=np.float32)
        times = np.asarray(times, dtype="datetime64[ns]")
        station_ids = np.asarray(station_ids, dtype=object)
    else:
        X_past = np.empty((0, past_hours), dtype=np.float32)
        y_future = np.empty((0, future_hours), dtype=np.float32)
        times = np.empty((0, future_hours), dtype="datetime64[ns]")
        station_ids = np.empty((0,), dtype=object)

    np.save(dataset_dir / f"X_past_{split_name}.npy", X_past)
    np.save(dataset_dir / f"y_future_{split_name}.npy", y_future)
    np.save(dataset_dir / f"times_{split_name}.npy", times)
    np.save(dataset_dir / f"station_ids_{split_name}.npy", station_ids)

    n_kept = len(X_past)
    anchors_label = "all hours" if anchor_hours is None else f"hours {list(anchor_hours)}"
    print(f"  {split_name}: kept {n_kept}/{n_total} candidate windows ({anchors_label})")
    if n_dropped_anchor:
        print(f"    dropped {n_dropped_anchor} window(s) due to anchor-hour filter")
    if n_dropped_past_dates:
        print(f"    dropped {n_dropped_past_dates} window(s) due to past-date span filter")
    print(f"    shapes: X_past={X_past.shape}  y_future={y_future.shape}  times={times.shape}")


def build_dataset_method3(
    *,
    final_csv_dir,
    dataset_dir,
    train_end,
    val_start,
    val_end,
    test_start,
    target_col,
    station_col,
    past_hours,
    future_hours,
    anchor_hours=None,
    min_past_dates=1,
):
    df = load_final_csv(final_csv_dir=final_csv_dir)
    required = {"datetime", target_col, station_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    train_df, val_df, test_df = temporal_split(
        df,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
    )

    print("\nSplit sizes:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Test:  {len(test_df)} rows")
    print("\n--- Sliding windows ---")

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        build_windows(
            split_df,
            split_name,
            dataset_dir=dataset_dir,
            target_col=target_col,
            past_hours=past_hours,
            future_hours=future_hours,
            station_col=station_col,
            anchor_hours=anchor_hours,
            min_past_dates=min_past_dates,
        )
