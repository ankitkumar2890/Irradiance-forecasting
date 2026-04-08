"""
02_build_features.py — Merge GHI + synthetic ICON + clear-sky → CAF + temporal features.

For each year (2017–2019):
  1. Load ghi_{year}.csv (IST naive), icon_{year}.csv (UTC), clearsky_{year}.csv (IST naive)
  2. Convert ICON UTC → IST naive
  3. Inner-join all three on datetime
  4. Compute CAF = GHI / clear_sky_GHI  (clipped [0,1], night = 0)
  5. Add temporal encodings (hour_sin/cos, doy_sin/cos)

Output: dataset/processed_data_2017_2019.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DOWNLOADS_DIR, DATASET_DIR, YEARS


def load_and_merge_year(year):
    """Load and merge all 3 sources for a single year."""
    ghi = pd.read_csv(DOWNLOADS_DIR / f"ghi_{year}.csv")
    icon = pd.read_csv(DOWNLOADS_DIR / f"icon_{year}.csv")
    cs = pd.read_csv(DOWNLOADS_DIR / f"clearsky_{year}.csv")

    # Parse datetimes
    ghi["datetime"] = pd.to_datetime(ghi["datetime"])
    cs["datetime"] = pd.to_datetime(cs["datetime"])

    # ICON is UTC → convert to IST naive
    icon["datetime"] = (
        pd.to_datetime(icon["datetime"], utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )

    # Dedup
    for d in [ghi, icon]:
        d.drop_duplicates(subset=["datetime"], keep="first", inplace=True)

    # Interpolate clear-sky onto the GHI/ICON timestamp grid
    target_times = pd.Index(
        sorted(set(ghi["datetime"]).intersection(set(icon["datetime"])))
    )
    interp_idx = (
        pd.DatetimeIndex(cs["datetime"])
        .union(pd.DatetimeIndex(target_times))
        .sort_values()
    )
    cs = (
        cs.drop_duplicates(subset=["datetime"], keep="first")
        .set_index("datetime")
        .reindex(interp_idx)
        .interpolate(method="time", limit_direction="both")
        .reindex(target_times)
        .reset_index()
        .rename(columns={"index": "datetime"})
    )

    # Merge
    df = (
        icon.merge(cs, on="datetime", how="inner")
        .merge(ghi, on="datetime", how="left")
    )
    df["w_ghr"] = df["w_ghr"].fillna(0.0).clip(lower=0)
    df["zenith_angle"] = df["zenith_angle"].fillna(95.0)
    df["clear_sky_ghi"] = df["clear_sky_ghi"].fillna(0.0).clip(lower=0)

    print(f"  {year}: {len(df)} rows merged")
    return df


def main():
    print(f"=== Building features for {YEARS} ===\n")

    # Load and merge all years
    all_dfs = []
    for year in YEARS:
        df = load_and_merge_year(year)
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    print(f"\n  Combined: {len(df)} rows")

    # CAF = GHI / clear_sky_GHI
    df["CAF"] = np.where(
        df["clear_sky_ghi"] > 1.0,
        (df["w_ghr"] / df["clear_sky_ghi"]).clip(0.0, 1.0),
        0.0,
    )

    # Temporal encodings
    df["hour"] = df["datetime"].dt.hour
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # Output columns (note: only cloud_cover, no cloud_low/mid/high)
    cols = [
        "datetime", "CAF", "clear_sky_ghi", "cloud_cover",
        "zenith_angle", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    ]
    df = df[cols].dropna().sort_values("datetime").reset_index(drop=True)

    out = DATASET_DIR / "processed_data_2017_2019.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}  shape={df.shape}")
    print(f"  CAF: mean={df['CAF'].mean():.3f}  std={df['CAF'].std():.3f}")
    print(f"  Date range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
    print(f"  cloud_cover: mean={df['cloud_cover'].mean():.3f}")


if __name__ == "__main__":
    main()
