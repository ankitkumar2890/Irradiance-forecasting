"""Preprocessing for the Moirai pipeline.

Three methods live in this file, one per approach. Each method reads its
own prepared CSV from ``moirai/final_csv/method{N}/`` and produces the
windowed dataset (``.npy``) that ``model.py`` consumes.

Layout::

  moirai/
  +-- final_csv/
  |   +-- method1/   <-- drop method-1 starting CSV here
  |   +-- method2/   <-- drop method-2 starting CSV here
  |   +-- method3/   <-- drop method-3 starting CSV here
  +-- dataset/method{N}/         (windowed .npy outputs)
  +-- functions/preprocess.py    (this file)
  +-- functions/model.py
  +-- moirai.py

Common helpers (load CSV, temporal split, build_windows) sit at the top.
Each method then has its own ``build_dataset_method{N}`` entry point so
the per-method preprocessing logic stays clearly separated.

Window selection
----------------
``build_windows`` is the single source of truth for which windows are
kept. It exposes ``anchor_hours`` (a list of allowed forecast-start hours
on the local clock; pass ``None`` to keep every hour) and *always* prints
how many candidate windows were considered, kept, and dropped, with the
reason for each rejection. Nothing is silently discarded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# =====================================================================
# SHARED HELPERS
# =====================================================================

def load_final_csv(*, final_csv_dir):
    """Locate and load the single prepared CSV in a method's final_csv folder."""
    final_csv_dir = Path(final_csv_dir)
    if not final_csv_dir.exists():
        raise FileNotFoundError(
            f"final_csv folder not found: {final_csv_dir}\n"
            "Create it and drop the prepared CSV inside before running."
        )
    csvs = sorted(p for p in final_csv_dir.glob("*.csv") if not p.name.startswith("."))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV found in {final_csv_dir}.\n"
            "Place exactly one prepared CSV (with datetime + features) inside."
        )
    if len(csvs) > 1:
        print(f"  Multiple CSVs found in {final_csv_dir}; using {csvs[0].name}")

    df = pd.read_csv(csvs[0])
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"Loaded {csvs[0].name}: {len(df)} rows")
    print(f"  Range: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")
    return df


def temporal_split(df, *, train_end, val_start, val_end, test_start):
    """Split a feature dataframe into train/val/test by datetime cuts."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    train_df = df[df["datetime"] <= train_end].reset_index(drop=True)
    val_df = df[
        (df["datetime"] >= val_start) & (df["datetime"] <= val_end)
    ].reset_index(drop=True)
    test_df = df[df["datetime"] >= test_start].reset_index(drop=True)
    return train_df, val_df, test_df


def _normalize_anchor_hours(anchor_hours) -> Optional[list[int]]:
    """Normalize the anchor_hours argument into either ``None`` (=all hours) or a sorted unique list."""
    if anchor_hours is None:
        return None
    if isinstance(anchor_hours, int):
        return [int(anchor_hours)]
    hours = sorted({int(h) for h in anchor_hours})
    for h in hours:
        if h < 0 or h > 23:
            raise ValueError(
                f"anchor_hours entries must be in [0, 23]; got {h}."
            )
    return hours


def build_windows(
    df,
    split_name,
    *,
    dataset_dir,
    past_hours,
    future_hours,
    past_features,
    future_features,
    target_col,
    anchor_hours: Optional[Sequence[int]] = None,
    min_past_dates: int = 1,
):
    """Build sliding windows and persist them as ``.npy`` arrays.

    Parameters
    ----------
    anchor_hours
        Allowed hours-of-day for the **first forecast hour** (i.e. the hour
        right after the last context hour). ``None`` keeps every hour
        (1 window per timestamp). A list like ``[6]`` reproduces the
        legacy "one forecast per day at 06:00" behaviour. ``[0, 6, 12, 18]``
        produces 4 forecasts per day.
    min_past_dates
        Reject a window if the past block does not span at least this
        many distinct calendar dates. Set to 1 to disable.

    The function prints a full audit of how many candidate windows were
    considered, kept, and dropped (with reasons). Nothing is silently
    discarded.
    """
    anchor_hours = _normalize_anchor_hours(anchor_hours)

    X_past, X_future, y_future, times = [], [], [], []
    n_total = max(len(df) - past_hours - future_hours + 1, 0)
    n_dropped_anchor = 0
    n_dropped_past_dates = 0

    for i in range(n_total):
        past = df.iloc[i : i + past_hours]
        future = df.iloc[i + past_hours : i + past_hours + future_hours]

        if anchor_hours is not None:
            if int(future.iloc[0]["datetime"].hour) not in anchor_hours:
                n_dropped_anchor += 1
                continue
        if min_past_dates > 1:
            if past["datetime"].dt.date.nunique() < min_past_dates:
                n_dropped_past_dates += 1
                continue

        X_past.append(past[past_features].values)
        X_future.append(future[future_features].values)
        y_future.append(future[target_col].values)
        times.append(future["datetime"].values)

    X_past = np.array(X_past, dtype=np.float32)
    X_future = np.array(X_future, dtype=np.float32)
    y_future = np.array(y_future, dtype=np.float32)
    times = np.array(times, dtype="datetime64[ns]")

    dataset_dir = Path(dataset_dir)
    for name, arr in [
        ("X_past", X_past),
        ("X_future", X_future),
        ("y_future", y_future),
        ("times", times),
    ]:
        np.save(dataset_dir / f"{name}_{split_name}.npy", arr)

    n_kept = len(X_past)
    anchors_label = "all hours" if anchor_hours is None else f"hours {anchor_hours}"
    print(
        f"  {split_name}: kept {n_kept}/{n_total} candidate windows "
        f"({anchors_label})"
    )
    if n_dropped_anchor:
        print(
            f"    dropped {n_dropped_anchor} window(s) because the first forecast hour "
            f"was not in the allowed anchor set."
        )
    if n_dropped_past_dates:
        print(
            f"    dropped {n_dropped_past_dates} window(s) because the past block "
            f"spanned fewer than {min_past_dates} distinct calendar dates."
        )
    print(
        f"    shapes: X_past={X_past.shape}  X_future={X_future.shape}  "
        f"y_future={y_future.shape}  times={times.shape}"
    )
    return X_past, X_future, y_future


def _split_and_window(
    df,
    *,
    dataset_dir,
    train_end,
    val_start,
    val_end,
    test_start,
    past_hours,
    future_hours,
    past_features,
    future_features,
    target_col,
    anchor_hours: Optional[Sequence[int]] = None,
    min_past_dates: int = 1,
):
    """Common: split into train/val/test and write ``.npy`` windows."""
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = temporal_split(
        df,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
    )

    print("\nSplit sizes:")
    print(
        f"  Train: {len(train_df)} rows  "
        f"({train_df['datetime'].iloc[0]} -> {train_df['datetime'].iloc[-1]})"
    )
    print(
        f"  Val:   {len(val_df)} rows  "
        f"({val_df['datetime'].iloc[0]} -> {val_df['datetime'].iloc[-1]})"
    )
    if len(test_df) > 0:
        print(
            f"  Test:  {len(test_df)} rows  "
            f"({test_df['datetime'].iloc[0]} -> {test_df['datetime'].iloc[-1]})"
        )
    else:
        print("  Test:  0 rows  (no held-out test range configured)")

    print("\n--- Sliding windows ---")
    if anchor_hours is None:
        print("  anchor_hours = None  (every hour kept; 1 window per timestamp)")
    else:
        print(f"  anchor_hours = {sorted(anchor_hours)}  (windows whose first forecast hour matches)")

    for split_name, split_df in [
        ("train", train_df), ("val", val_df), ("test", test_df),
    ]:
        build_windows(
            split_df, split_name,
            dataset_dir=dataset_dir,
            past_hours=past_hours, future_hours=future_hours,
            past_features=past_features, future_features=future_features,
            target_col=target_col,
            anchor_hours=anchor_hours,
            min_past_dates=min_past_dates,
        )


def _save_station_ids(*, dataset_dir, split_name, station_ids):
    """Persist station_ids alongside the .npy windows for multi-station methods."""
    dataset_dir = Path(dataset_dir)
    np.save(
        dataset_dir / f"station_ids_{split_name}.npy",
        np.array(station_ids, dtype=object),
    )


# =====================================================================
# METHOD 1 - CAF (PVLib clear-sky + ERA5 covariates)
# ---------------------------------------------------------------------
# Source: phase2_era5_direct.
# Target: CAF = w_ghr / clear_sky_ghi (clipped to [0, 1]).
# Starting CSV: moirai/final_csv/method1/<one>.csv
# Required columns:
#   datetime, CAF, clear_sky_ghi, w_ghr,
#   total_cloud_cover, low_cloud_cover, medium_cloud_cover, high_cloud_cover,
#   cloud_liquid_water, cloud_ice_water, water_vapour,
#   zenith_angle, hour_sin, hour_cos, doy_sin, doy_cos
# (CAF and the calendar columns are auto-derived if missing and the raw
# inputs ``w_ghr`` and ``clear_sky_ghi`` are present.)
# =====================================================================

def ensure_features_method1(df):
    """Fill in CAF and sin/cos calendar encodings if they are missing."""
    df = df.copy()

    if "CAF" not in df.columns:
        if {"w_ghr", "clear_sky_ghi"}.issubset(df.columns):
            df["CAF"] = np.where(
                df["clear_sky_ghi"] > 1.0,
                (df["w_ghr"] / df["clear_sky_ghi"]).clip(0.0, 1.0),
                0.0,
            )
        else:
            raise ValueError(
                "Method 1 CSV is missing 'CAF' and we cannot derive it without "
                "both 'w_ghr' and 'clear_sky_ghi' columns."
            )

    if "hour_sin" not in df.columns or "hour_cos" not in df.columns:
        hour = df["datetime"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if "doy_sin" not in df.columns or "doy_cos" not in df.columns:
        doy = df["datetime"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if "hour" not in df.columns:
        df["hour"] = df["datetime"].dt.hour

    return df


def build_dataset_method1(
    *,
    final_csv_dir,
    dataset_dir,
    train_end,
    val_start,
    val_end,
    test_start,
    past_hours,
    future_hours,
    past_features,
    future_features,
    item_id,                       # accepted for API symmetry; unused
    target_col="CAF",
    anchor_hours: Optional[Sequence[int]] = None,
    min_past_dates: int = 1,
    measured_col: str = "w_ghr",
):
    """Method 1 dataset builder: CSV -> CAF features -> ``.npy`` windows.

    The evaluation step needs ``measured_col`` (default ``w_ghr``) to
    reconstruct GHI from CAF, so we carry it through the slim feature
    frame and validate up-front. This is what makes ``evaluate`` fail
    fast at dataset time instead of after fine-tuning has finished.
    """
    del item_id  # API symmetry with methods 2/3
    df = load_final_csv(final_csv_dir=final_csv_dir)
    df = ensure_features_method1(df)

    needed = sorted({*past_features, *future_features, target_col, measured_col, "datetime"})
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Method 1 CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = (
        df[needed].dropna().sort_values("datetime").reset_index(drop=True)
    )

    _split_and_window(
        df,
        dataset_dir=dataset_dir,
        train_end=train_end, val_start=val_start,
        val_end=val_end, test_start=test_start,
        past_hours=past_hours, future_hours=future_hours,
        past_features=past_features, future_features=future_features,
        target_col=target_col,
        anchor_hours=anchor_hours,
        min_past_dates=min_past_dates,
    )


# =====================================================================
# METHOD 2 - Multi-station fine-tuning, NSRDB clear-sky, CAF -> GHI
# ---------------------------------------------------------------------
# Source: phase2_finetuning.
# Differences from Method 1:
#   * MULTI-station (3x3 grid around Tirunelveli, or the 5-station explicit
#     cluster) instead of a single station.
#   * Clear-sky GHI is taken FROM NSRDB (the same NSRDB query that returns
#     w_ghr also returns clearsky_ghi). PVLib is NOT used.
#   * Target is still CAF = w_ghr / clearsky_ghi, and predictions are
#     converted back to GHI at evaluation time:  GHI_pred = CAF_pred * cs_ghi.
#   * Covariates include ERA5 cloud cover (tcc/lcc/mcc/hcc), wind speed +
#     direction (derived from u10/v10), zenith + azimuth angles, station
#     elevation, and sin/cos calendar features.
# Starting CSV: moirai/final_csv/method2/<one>.csv
# Expected to contain (at minimum) per-row:
#   datetime, station_id, w_ghr, clearsky_ghi (from NSRDB),
#   tcc, lcc, mcc, hcc, u10/v10 (or wind_speed + wind_direction),
#   zenith_angle, azimuth_angle, elevation_m
# =====================================================================

def ensure_features_method2(df, *, clearsky_col: str = "clearsky_ghi", measured_col: str = "w_ghr"):
    """Method 2 feature backfill (multi-station, NSRDB clear-sky)."""
    df = df.copy()

    if "CAF" not in df.columns:
        if {measured_col, clearsky_col}.issubset(df.columns):
            df["CAF"] = np.where(
                df[clearsky_col] > 1.0,
                (df[measured_col] / df[clearsky_col]).clip(0.0, 1.0),
                0.0,
            )
        else:
            raise ValueError(
                f"Method 2 CSV is missing 'CAF' and we cannot derive it without "
                f"both '{measured_col}' and '{clearsky_col}' columns."
            )

    if "wind_speed" not in df.columns or "wind_direction" not in df.columns:
        if {"u10", "v10"}.issubset(df.columns):
            u, v = df["u10"].values, df["v10"].values
            df["wind_speed"] = np.sqrt(u * u + v * v)
            df["wind_direction"] = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    if "hour_sin" not in df.columns or "hour_cos" not in df.columns:
        hour = df["datetime"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if "doy_sin" not in df.columns or "doy_cos" not in df.columns:
        doy = df["datetime"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if "hour" not in df.columns:
        df["hour"] = df["datetime"].dt.hour
    return df


def build_dataset_method2(
    *,
    final_csv_dir,
    dataset_dir,
    train_end,
    val_start,
    val_end,
    test_start,
    past_hours,
    future_hours,
    past_features,
    future_features,
    item_id,                       # accepted for API symmetry; unused
    target_col="CAF",
    anchor_hours: Optional[Sequence[int]] = None,
    min_past_dates: int = 1,
    clearsky_col: str = "clearsky_ghi",
    measured_col: str = "w_ghr",
):
    """Method 2 dataset builder (multi-station, CAF target).

    Builds windows per ``station_id`` and persists a ``station_ids_{split}.npy``
    file alongside the X/y windows so the evaluation step can do per-station
    metrics.
    """
    del item_id
    df = load_final_csv(final_csv_dir=final_csv_dir)
    df = ensure_features_method2(df, clearsky_col=clearsky_col, measured_col=measured_col)

    if "station_id" not in df.columns:
        raise ValueError(
            "Method 2 CSV must contain a 'station_id' column (multi-station fine-tune)."
        )

    needed = sorted({
        *past_features, *future_features,
        target_col, measured_col, clearsky_col,
        "station_id", "datetime",
    })
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Method 2 CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = (
        df[needed].dropna().sort_values(["station_id", "datetime"]).reset_index(drop=True)
    )

    print("\nBuilding per-station windows for Method 2...")
    station_ids = sorted(df["station_id"].unique().tolist())
    print(f"  Stations: {station_ids}")

    splits: dict[str, dict] = {
        "train": {"X_past": [], "X_future": [], "y_future": [], "times": [], "station_ids": []},
        "val":   {"X_past": [], "X_future": [], "y_future": [], "times": [], "station_ids": []},
        "test":  {"X_past": [], "X_future": [], "y_future": [], "times": [], "station_ids": []},
    }

    for sid, sdf in df.groupby("station_id", sort=True):
        print(f"\n  station_id = {sid}  ({len(sdf)} rows)")
        train_df, val_df, test_df = temporal_split(
            sdf, train_end=train_end, val_start=val_start, val_end=val_end, test_start=test_start,
        )
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(split_df) == 0:
                print(f"    {split_name}: 0 rows; skipped.")
                continue
            X_past, X_future, y_future = _build_windows_in_memory(
                split_df,
                past_hours=past_hours, future_hours=future_hours,
                past_features=past_features, future_features=future_features,
                target_col=target_col,
                anchor_hours=anchor_hours,
                min_past_dates=min_past_dates,
                split_label=f"{split_name}/{sid}",
            )
            if len(X_past) == 0:
                continue
            splits[split_name]["X_past"].append(X_past)
            splits[split_name]["X_future"].append(X_future)
            splits[split_name]["y_future"].append(y_future)
            splits[split_name]["station_ids"].extend([sid] * len(X_past))
            split_df_idx = split_df.reset_index(drop=True)
            times = _times_for_windows(
                split_df_idx, past_hours=past_hours, future_hours=future_hours,
                anchor_hours=anchor_hours, min_past_dates=min_past_dates,
            )
            splits[split_name]["times"].append(times)

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split_name, payload in splits.items():
        if not payload["X_past"]:
            empty = np.zeros((0,), dtype=np.float32)
            np.save(dataset_dir / f"X_past_{split_name}.npy", empty)
            np.save(dataset_dir / f"X_future_{split_name}.npy", empty)
            np.save(dataset_dir / f"y_future_{split_name}.npy", empty)
            np.save(
                dataset_dir / f"times_{split_name}.npy",
                np.zeros((0,), dtype="datetime64[ns]"),
            )
            _save_station_ids(dataset_dir=dataset_dir, split_name=split_name, station_ids=[])
            print(f"  {split_name}: 0 windows across all stations.")
            continue
        X_past = np.concatenate(payload["X_past"], axis=0).astype(np.float32)
        X_future = np.concatenate(payload["X_future"], axis=0).astype(np.float32)
        y_future = np.concatenate(payload["y_future"], axis=0).astype(np.float32)
        times = np.concatenate(payload["times"], axis=0).astype("datetime64[ns]")
        np.save(dataset_dir / f"X_past_{split_name}.npy", X_past)
        np.save(dataset_dir / f"X_future_{split_name}.npy", X_future)
        np.save(dataset_dir / f"y_future_{split_name}.npy", y_future)
        np.save(dataset_dir / f"times_{split_name}.npy", times)
        _save_station_ids(
            dataset_dir=dataset_dir,
            split_name=split_name,
            station_ids=payload["station_ids"],
        )
        print(
            f"  {split_name}: {len(X_past)} windows total  "
            f"X_past={X_past.shape}  X_future={X_future.shape}"
        )


# =====================================================================
# METHOD 3 - Direct GHI forecasting (multi-station, GHI scaling, daylight mask)
# ---------------------------------------------------------------------
# Source: phase3_direct_ghi.
# Differences from Methods 1 and 2:
#   * Target is GHI (W/m^2) DIRECTLY - no CAF intermediate.
#   * GHI is NOT scaled here; the scale factor is applied on the model side
#     (model.finetune_method3) so the .npy windows on disk stay in
#     physical units.
#   * Multi-station with neighbor-context features.
# Starting CSV: moirai/final_csv/method3/<one>.csv
# =====================================================================

def ensure_features_method3(df, *, clearsky_col: str = "clearsky_ghi", measured_col: str = "w_ghr"):
    """Method 3 feature backfill (direct GHI, multi-station)."""
    df = df.copy()

    if "wind_speed" not in df.columns or "wind_direction" not in df.columns:
        if {"u10", "v10"}.issubset(df.columns):
            u, v = df["u10"].values, df["v10"].values
            df["wind_speed"] = np.sqrt(u * u + v * v)
            df["wind_direction"] = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

    if "hour_sin" not in df.columns or "hour_cos" not in df.columns:
        hour = df["datetime"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if "doy_sin" not in df.columns or "doy_cos" not in df.columns:
        doy = df["datetime"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    if "hour" not in df.columns:
        df["hour"] = df["datetime"].dt.hour
    return df


def build_dataset_method3(
    *,
    final_csv_dir,
    dataset_dir,
    train_end,
    val_start,
    val_end,
    test_start,
    past_hours,
    future_hours,
    past_features,
    future_features,
    item_id,
    target_col,
    anchor_hours: Optional[Sequence[int]] = None,
    min_past_dates: int = 1,
    clearsky_col: str = "clearsky_ghi",
    measured_col: str = "w_ghr",
):
    """Method 3 dataset builder (direct GHI, multi-station).

    Validates the per-method contracts that ``functions/results.py``
    relies on at inference time:

      * ``past_features[0]`` must be the GHI track (default ``w_ghr``).
      * Every entry in ``future_features`` must also be in ``past_features``.
    """
    del item_id

    if not past_features or past_features[0] != measured_col:
        raise ValueError(
            f"Method 3 expects PAST_FEATURES[0] == '{measured_col}' so the autoregressive "
            f"target track is the measured GHI series. Got PAST_FEATURES[0]="
            f"{past_features[0] if past_features else None!r}."
        )
    missing_in_past = [f for f in future_features if f not in past_features]
    if missing_in_past:
        raise ValueError(
            "Method 3 expects every FUTURE_FEATURE to also be in PAST_FEATURES "
            f"(needed by the inference-time index map). Missing: {missing_in_past}."
        )
    df = load_final_csv(final_csv_dir=final_csv_dir)
    df = ensure_features_method3(df, clearsky_col=clearsky_col, measured_col=measured_col)

    if "station_id" not in df.columns:
        raise ValueError(
            "Method 3 CSV must contain a 'station_id' column (multi-station)."
        )

    needed = sorted({
        *past_features, *future_features,
        target_col, measured_col, clearsky_col,
        "station_id", "datetime",
    })
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Method 3 CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = (
        df[needed].dropna().sort_values(["station_id", "datetime"]).reset_index(drop=True)
    )

    print("\nBuilding per-station windows for Method 3...")
    station_ids = sorted(df["station_id"].unique().tolist())
    print(f"  Stations: {station_ids}")

    splits: dict[str, dict] = {
        "train": {"X_past": [], "X_future": [], "y_future": [], "times": [], "station_ids": []},
        "val":   {"X_past": [], "X_future": [], "y_future": [], "times": [], "station_ids": []},
        "test":  {"X_past": [], "X_future": [], "y_future": [], "times": [], "station_ids": []},
    }

    for sid, sdf in df.groupby("station_id", sort=True):
        print(f"\n  station_id = {sid}  ({len(sdf)} rows)")
        train_df, val_df, test_df = temporal_split(
            sdf, train_end=train_end, val_start=val_start, val_end=val_end, test_start=test_start,
        )
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(split_df) == 0:
                print(f"    {split_name}: 0 rows; skipped.")
                continue
            X_past, X_future, y_future = _build_windows_in_memory(
                split_df,
                past_hours=past_hours, future_hours=future_hours,
                past_features=past_features, future_features=future_features,
                target_col=target_col,
                anchor_hours=anchor_hours,
                min_past_dates=min_past_dates,
                split_label=f"{split_name}/{sid}",
            )
            if len(X_past) == 0:
                continue
            splits[split_name]["X_past"].append(X_past)
            splits[split_name]["X_future"].append(X_future)
            splits[split_name]["y_future"].append(y_future)
            splits[split_name]["station_ids"].extend([sid] * len(X_past))
            split_df_idx = split_df.reset_index(drop=True)
            times = _times_for_windows(
                split_df_idx, past_hours=past_hours, future_hours=future_hours,
                anchor_hours=anchor_hours, min_past_dates=min_past_dates,
            )
            splits[split_name]["times"].append(times)

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split_name, payload in splits.items():
        if not payload["X_past"]:
            empty = np.zeros((0,), dtype=np.float32)
            np.save(dataset_dir / f"X_past_{split_name}.npy", empty)
            np.save(dataset_dir / f"X_future_{split_name}.npy", empty)
            np.save(dataset_dir / f"y_future_{split_name}.npy", empty)
            np.save(
                dataset_dir / f"times_{split_name}.npy",
                np.zeros((0,), dtype="datetime64[ns]"),
            )
            _save_station_ids(dataset_dir=dataset_dir, split_name=split_name, station_ids=[])
            print(f"  {split_name}: 0 windows across all stations.")
            continue
        X_past = np.concatenate(payload["X_past"], axis=0).astype(np.float32)
        X_future = np.concatenate(payload["X_future"], axis=0).astype(np.float32)
        y_future = np.concatenate(payload["y_future"], axis=0).astype(np.float32)
        times = np.concatenate(payload["times"], axis=0).astype("datetime64[ns]")
        np.save(dataset_dir / f"X_past_{split_name}.npy", X_past)
        np.save(dataset_dir / f"X_future_{split_name}.npy", X_future)
        np.save(dataset_dir / f"y_future_{split_name}.npy", y_future)
        np.save(dataset_dir / f"times_{split_name}.npy", times)
        _save_station_ids(
            dataset_dir=dataset_dir,
            split_name=split_name,
            station_ids=payload["station_ids"],
        )
        print(
            f"  {split_name}: {len(X_past)} windows total  "
            f"X_past={X_past.shape}  X_future={X_future.shape}"
        )


# =====================================================================
# Internal: in-memory window builders used by Methods 2 & 3 to assemble
# per-station windows and concatenate them across stations. These do
# NOT write .npy files themselves.
# =====================================================================

def _build_windows_in_memory(
    df,
    *,
    past_hours,
    future_hours,
    past_features,
    future_features,
    target_col,
    anchor_hours: Optional[Sequence[int]],
    min_past_dates: int,
    split_label: str,
):
    """Window-builder twin of :func:`build_windows` that returns arrays instead of saving.

    Prints the same kept/dropped audit, scoped to ``split_label`` (e.g.
    ``"train/chennai"``) so multi-station runs are easy to read.
    """
    anchor_hours = _normalize_anchor_hours(anchor_hours)
    X_past, X_future, y_future = [], [], []

    n_total = max(len(df) - past_hours - future_hours + 1, 0)
    n_dropped_anchor = 0
    n_dropped_past_dates = 0

    for i in range(n_total):
        past = df.iloc[i : i + past_hours]
        future = df.iloc[i + past_hours : i + past_hours + future_hours]

        if anchor_hours is not None:
            if int(future.iloc[0]["datetime"].hour) not in anchor_hours:
                n_dropped_anchor += 1
                continue
        if min_past_dates > 1:
            if past["datetime"].dt.date.nunique() < min_past_dates:
                n_dropped_past_dates += 1
                continue

        X_past.append(past[past_features].values)
        X_future.append(future[future_features].values)
        y_future.append(future[target_col].values)

    n_kept = len(X_past)
    anchors_label = "all hours" if anchor_hours is None else f"hours {sorted(anchor_hours)}"
    print(
        f"    {split_label}: kept {n_kept}/{n_total} candidate windows  ({anchors_label})"
    )
    if n_dropped_anchor:
        print(
            f"      dropped {n_dropped_anchor} window(s) because the first forecast hour "
            f"was not in the allowed anchor set."
        )
    if n_dropped_past_dates:
        print(
            f"      dropped {n_dropped_past_dates} window(s) because the past block "
            f"spanned fewer than {min_past_dates} distinct calendar dates."
        )

    if not X_past:
        return (
            np.zeros((0, past_hours, len(past_features)), dtype=np.float32),
            np.zeros((0, future_hours, len(future_features)), dtype=np.float32),
            np.zeros((0, future_hours), dtype=np.float32),
        )
    return (
        np.array(X_past, dtype=np.float32),
        np.array(X_future, dtype=np.float32),
        np.array(y_future, dtype=np.float32),
    )


def _times_for_windows(
    df,
    *,
    past_hours: int,
    future_hours: int,
    anchor_hours: Optional[Sequence[int]],
    min_past_dates: int,
):
    """Return the future-timestamps array for the kept windows in ``df``.

    Mirrors :func:`_build_windows_in_memory` exactly so the two stay in
    sync row-for-row.
    """
    anchor_hours = _normalize_anchor_hours(anchor_hours)
    times: list = []
    n_total = max(len(df) - past_hours - future_hours + 1, 0)
    for i in range(n_total):
        past = df.iloc[i : i + past_hours]
        future = df.iloc[i + past_hours : i + past_hours + future_hours]
        if anchor_hours is not None:
            if int(future.iloc[0]["datetime"].hour) not in anchor_hours:
                continue
        if min_past_dates > 1:
            if past["datetime"].dt.date.nunique() < min_past_dates:
                continue
        times.append(future["datetime"].values)
    if not times:
        return np.zeros((0, future_hours), dtype="datetime64[ns]")
    return np.array(times, dtype="datetime64[ns]")
