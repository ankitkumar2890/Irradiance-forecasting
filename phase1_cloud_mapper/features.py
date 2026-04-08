# ==============================================================================
# features.py — Phase 1: CloudMapper v6 Feature Engineering
#
# Simple t / t-1 lag approach:
#   - 4 ERA5 cloud fractions at time t (primary signal)
#   - 4 ERA5 cloud fractions at time t-1 (trend signal)
#   - 4 cyclic time features (hour_sin, hour_cos, doy_sin, doy_cos)
#   - 3 station features (lat_norm, lon_norm, alt_norm)
#   = 15 total input features
#
# Why t/t-1 instead of rolling aggregates:
#   The network can implicitly learn delta = x(t) - x(t-1) if the trend
#   matters. One concrete lag is grounded — no bag of hand-engineered
#   summary statistics that may or may not capture what matters.
# ==============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ERA5_FRACTION_COLS, STATIONS


# ---------------------------------------------------------------------------
# Station metadata for spatial features
# ---------------------------------------------------------------------------
STATION_META = {
    s["id"]: {
        "lat_norm": s["lat"] / 90.0,
        "lon_norm": s["lon"] / 180.0,
        "alt_norm": s["alt_m"] / 1000.0,
    }
    for s in STATIONS
}

STATION_FEATURE_COLS = ["lat_norm", "lon_norm", "alt_norm"]
TIME_FEATURE_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
LAG_COLS = [f"{c}_lag1" for c in ERA5_FRACTION_COLS]


# ---------------------------------------------------------------------------
# 1. Time features — diurnal + seasonal cycles
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """
    Add cyclic time encodings.

    hour_sin/cos: captures the diurnal cycle (cloud formation peaks in afternoon
    from convection, clears at night from radiative cooling).

    doy_sin/cos: captures the seasonal cycle (monsoon vs dry season in South India).

    We use sin/cos pairs so that 23:00 and 01:00 are close to each other
    (unlike raw hour which has a discontinuity at midnight).
    """
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col], utc=True)
    hour_angle = 2.0 * np.pi * (dt.dt.hour + dt.dt.minute / 60.0) / 24.0
    day_angle = 2.0 * np.pi * dt.dt.dayofyear / 365.25
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df["doy_sin"] = np.sin(day_angle)
    df["doy_cos"] = np.cos(day_angle)
    return df


# ---------------------------------------------------------------------------
# 2. Station features — spatial identity
# ---------------------------------------------------------------------------
def add_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalised station coordinates as features.

    These let the model learn station-specific biases:
    - Chennai (coastal) has different cloud patterns than Coimbatore (inland, elevated)
    - Altitude affects boundary-layer cloud formation
    """
    df = df.copy()
    if "station_id" not in df.columns:
        return df
    for feat in STATION_FEATURE_COLS:
        df[feat] = df["station_id"].map(
            {sid: vals[feat] for sid, vals in STATION_META.items()}
        )
    return df


# ---------------------------------------------------------------------------
# 3. Lag features — t-1 for trend signal
# ---------------------------------------------------------------------------
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ERA5 cloud fractions from the previous hour (t-1).

    This gives the model the ability to detect:
    - Is total cloud increasing or decreasing?
    - Did low cloud just appear (fog forming)?
    - Did high cloud just disappear (convection ending)?

    The network can implicitly compute x(t) - x(t-1) if the delta
    matters, without us pre-computing rolling means, stds, etc.

    Computed per-station to avoid cross-station leakage
    (e.g., Chennai's midnight row shouldn't use Coimbatore's 23:00).
    """
    df = df.copy()

    if "station_id" in df.columns:
        # Per-station lag to avoid cross-station contamination
        for col in ERA5_FRACTION_COLS:
            lag_col = f"{col}_lag1"
            df[lag_col] = df.groupby("station_id")[col].shift(1)
    else:
        for col in ERA5_FRACTION_COLS:
            lag_col = f"{col}_lag1"
            df[lag_col] = df[col].shift(1)

    # Fill first row's NaN with the current value (no trend info for first sample)
    for col in LAG_COLS:
        df[col] = df[col].fillna(df[col.replace("_lag1", "")])

    return df


# ---------------------------------------------------------------------------
# 4. Full pipeline
# ---------------------------------------------------------------------------
def build_all_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply all feature engineering steps and return the ordered feature column list.

    Pipeline:
        1. Clip ERA5 fractions to [0, 1]
        2. Add time features (4 cols)
        3. Add station features (3 cols, if multi-station)
        4. Add t-1 lag features (4 cols)
        5. Assemble input column list

    Returns:
        (df_with_features, input_col_list)
    """
    df = clip_fraction_columns(df)
    df = add_time_features(df)
    df = add_station_features(df)
    df = add_lag_features(df)

    # Assemble input columns in order:
    #   [4 ERA5 at t] + [4 ERA5 at t-1] + [4 time] + [3 station]
    input_cols = list(ERA5_FRACTION_COLS)  # 4: cloud fractions at t
    input_cols += LAG_COLS                 # 4: cloud fractions at t-1
    input_cols += TIME_FEATURE_COLS        # 4: hour_sin, hour_cos, doy_sin, doy_cos

    # Station features (only if multi-station data)
    for col in STATION_FEATURE_COLS:
        if col in df.columns:
            input_cols.append(col)

    return df, input_cols


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def clip_fraction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clip ERA5 cloud fraction columns to physical bounds [0, 1]."""
    df = df.copy()
    for col in ERA5_FRACTION_COLS:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)
    return df
