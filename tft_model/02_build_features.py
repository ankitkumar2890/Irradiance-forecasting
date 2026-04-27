"""Merge GHI + ERA5 + clear-sky into the TFT processed dataset."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASET_DIR, ERA5_FEATURE_COLUMNS, MULTI_STATION_DOWNLOADS_DIR, STATIONS, YEARS


def load_and_merge_station_year(station_id, year):
    """Load and merge one station-year from TFT-local downloads."""
    station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
    ghi = pd.read_csv(station_dir / f"ghi_{year}.csv")
    era5 = pd.read_csv(station_dir / f"era5_{year}.csv")
    cs = pd.read_csv(station_dir / f"clearsky_{year}.csv")

    ghi["datetime"] = pd.to_datetime(ghi["datetime"])
    era5["datetime"] = pd.to_datetime(era5["datetime"])
    cs["datetime"] = pd.to_datetime(cs["datetime"])

    # NREL often arrives on a :30 grid. Shift onto the :00 hourly grid first.
    ghi_minutes = sorted(ghi["datetime"].dt.minute.dropna().unique().tolist())
    if ghi_minutes == [30]:
        ghi["datetime"] = ghi["datetime"] - pd.Timedelta(minutes=30)

    for frame in [ghi, era5]:
        frame.drop_duplicates(subset=["datetime"], keep="first", inplace=True)

    target_times = pd.Index(sorted(set(ghi["datetime"]).intersection(set(era5["datetime"]))))
    interp_idx = pd.DatetimeIndex(cs["datetime"]).union(pd.DatetimeIndex(target_times)).sort_values()
    cs = (
        cs.drop_duplicates(subset=["datetime"], keep="first")
        .set_index("datetime")
        .reindex(interp_idx)
        .interpolate(method="time", limit_direction="both")
        .reindex(target_times)
        .reset_index()
        .rename(columns={"index": "datetime"})
    )

    df = era5.merge(cs, on="datetime", how="inner").merge(ghi, on="datetime", how="left")
    df["w_ghr"] = df["w_ghr"].fillna(0.0).clip(lower=0)
    df["clear_sky_ghi"] = df["clear_sky_ghi"].fillna(0.0).clip(lower=0)
    df["zenith_angle"] = df["zenith_angle"].fillna(95.0)

    for col in ["tcc", "lcc", "mcc", "hcc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0.0, 1.0)
    for col in ["u10", "v10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["station_id"] = station_id
    print(f"  {station_id} {year}: {len(df)} rows merged")
    return df


def main():
    print(f"=== Building TFT ERA5 features for {YEARS} ===\n")

    all_dfs = []
    for station in STATIONS:
        for year in YEARS:
            all_dfs.append(load_and_merge_station_year(station["id"], year))

    df = pd.concat(all_dfs, ignore_index=True).sort_values(["station_id", "datetime"]).reset_index(drop=True)
    print(f"\n  Combined: {len(df)} rows")

    df["CAF"] = np.where(
        df["clear_sky_ghi"] > 1.0,
        (df["w_ghr"] / df["clear_sky_ghi"]).clip(0.0, 1.0),
        0.0,
    )

    df["hour"] = df["datetime"].dt.hour
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    cols = [
        "station_id",
        "datetime",
        "CAF",
        "clear_sky_ghi",
        *ERA5_FEATURE_COLUMNS,
        "zenith_angle",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    df = df[cols].dropna().sort_values(["station_id", "datetime"]).reset_index(drop=True)

    out = DATASET_DIR / "processed_data_2017_2019.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out} shape={df.shape}")
    print(f"  Stations: {df['station_id'].nunique()} {sorted(df['station_id'].unique().tolist())}")
    print(f"  Range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  CAF mean/std: {df['CAF'].mean():.3f} / {df['CAF'].std():.3f}")
    for col in ERA5_FEATURE_COLUMNS:
        print(f"  {col}: mean={df[col].mean():.3f}")


if __name__ == "__main__":
    main()
