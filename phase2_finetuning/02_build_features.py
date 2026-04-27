"""Merge Phase 2 ERA5 + NSRDB GHI features into the fine-tuning dataset.

Reads from:  downloads/processed/<station>/ or downloads/multi_station/<station>/
Writes to:   dataset/processed_data_2017_2019.csv

Steps:
  1. Load hourly ERA5 plus NSRDB GHI features per station-year
  2. Align raw NSRDB data to hourly when processed files are stale
  3. Compute Cloud Attenuation Factor (CAF = GHI / clear_sky_ghi)
  4. Add cyclical time encodings (hour_sin/cos, doy_sin/cos)
  5. Concatenate all station-years → single processed CSV
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib.location import Location

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_DIR,
    ELEVATION_MAP,
    ERA5_FEATURE_COLUMNS,
    MULTI_STATION_DOWNLOADS_DIR,
    PROCESSED_DOWNLOADS_DIR,
    STATIONS,
    YEARS,
)


REQUIRED_GHI_COLUMNS = {"datetime", "w_ghr", "nsrdb_clearsky_ghi", "zenith_angle"}
STATION_META = {station["id"]: station for station in STATIONS}


def _finalize_hourly_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="first").reset_index(drop=True)


def _align_hourly_ghi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime").reset_index(drop=True)
    minutes = sorted(out["datetime"].dt.minute.dropna().unique().tolist())
    value_cols = [col for col in ["w_ghr", "nsrdb_clearsky_ghi", "zenith_angle"] if col in out.columns]

    if minutes == [0]:
        return _finalize_hourly_frame(out[["datetime", *value_cols]])

    indexed = out.set_index("datetime")[value_cols].apply(pd.to_numeric, errors="coerce")

    if minutes == [30]:
        # Legacy fallback for already-downloaded hourly :30 NSRDB data:
        # 6:00 uses 5:30 and 6:30
        timestamps = indexed.index.to_list()
        rows = []
        for prev_ts, next_ts in zip(timestamps[:-1], timestamps[1:]):
            midpoint = prev_ts + pd.Timedelta(minutes=30)
            if midpoint.minute != 0:
                continue
            pair = indexed.loc[[prev_ts, next_ts]]
            mean_vals = pair.mean(axis=0, skipna=True)
            row = {"datetime": midpoint}
            for col in value_cols:
                row[col] = mean_vals.get(col, np.nan)
            rows.append(row)
        return _finalize_hourly_frame(pd.DataFrame(rows))

    # Centered quarter-hour average:
    # 6:00 uses 5:45, 6:00, 6:15
    if set(minutes).issubset({0, 15, 30, 45}):
        target_idx = indexed.index[indexed.index.minute == 0]
        rows = []
        for ts in target_idx:
            vals = indexed.reindex(
                [ts - pd.Timedelta(minutes=15), ts, ts + pd.Timedelta(minutes=15)]
            )
            mean_vals = vals.mean(axis=0, skipna=True)
            row = {"datetime": ts}
            for col in value_cols:
                row[col] = mean_vals.get(col, np.nan)
            rows.append(row)
        return _finalize_hourly_frame(pd.DataFrame(rows))

    return _finalize_hourly_frame(indexed.reset_index())


def _align_hourly_era5(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime").reset_index(drop=True)
    minutes = sorted(out["datetime"].dt.minute.dropna().unique().tolist())
    value_cols = [col for col in out.columns if col != "datetime"]

    if minutes == [0]:
        return _finalize_hourly_frame(out)

    indexed = out.set_index("datetime")[value_cols].apply(pd.to_numeric, errors="coerce")

    if minutes == [30]:
        # 6:00 uses 5:30 and 6:30
        timestamps = indexed.index.to_list()
        rows = []
        for prev_ts, next_ts in zip(timestamps[:-1], timestamps[1:]):
            midpoint = prev_ts + pd.Timedelta(minutes=30)
            if midpoint.minute != 0:
                continue
            pair = indexed.loc[[prev_ts, next_ts]]
            mean_vals = pair.mean(axis=0, skipna=True)
            row = {"datetime": midpoint}
            for col in value_cols:
                row[col] = mean_vals.get(col, np.nan)
            rows.append(row)
        return _finalize_hourly_frame(pd.DataFrame(rows))

    return _finalize_hourly_frame(indexed.reset_index())


def _compute_solar_geometry(station_id: str, datetimes: pd.Series) -> pd.DataFrame:
    station = STATION_META[station_id]
    site = Location(
        station["lat"],
        station["lon"],
        tz="Asia/Kolkata",
        altitude=station["alt_m"],
    )
    # Use a plain numpy array for the datetime column so the returned frame has
    # a clean RangeIndex. Otherwise pvlib's DatetimeIndex (name="datetime") on
    # `solpos` propagates via alignment and makes `datetime` both an index
    # level and a column → merge ambiguity.
    naive = pd.to_datetime(datetimes).to_numpy()
    times = pd.DatetimeIndex(naive).tz_localize("Asia/Kolkata")
    solpos = site.get_solarposition(times)
    return pd.DataFrame(
        {
            "datetime": naive,
            "zenith_angle_pvlib": pd.to_numeric(
                solpos["apparent_zenith"].to_numpy(), errors="coerce"
            ),
            "azimuth_angle": pd.to_numeric(
                solpos["azimuth"].to_numpy(), errors="coerce"
            ),
        }
    )


def load_hourly_ghi(station_id: str, year: int) -> pd.DataFrame:
    processed_path = PROCESSED_DOWNLOADS_DIR / station_id / f"ghi_{year}.csv"
    raw_path = MULTI_STATION_DOWNLOADS_DIR / station_id / f"ghi_{year}.csv"

    if processed_path.exists():
        processed = pd.read_csv(processed_path, parse_dates=["datetime"])
        missing = sorted(REQUIRED_GHI_COLUMNS.difference(processed.columns))
        if not missing:
            processed = _finalize_hourly_frame(processed[["datetime", "w_ghr", "nsrdb_clearsky_ghi", "zenith_angle"]])
            processed_minutes = sorted(processed["datetime"].dt.minute.dropna().unique().tolist())
            if processed_minutes == [0]:
                return processed
            print(f"  {station_id} {year}: processed ghi grid {processed_minutes} is not hourly :00; using raw NSRDB ghi file.")
        else:
            print(f"  {station_id} {year}: processed ghi missing {missing}; using raw NSRDB ghi file.")

    raw = pd.read_csv(raw_path, parse_dates=["datetime"])
    missing = sorted({"datetime", "w_ghr", "nsrdb_clearsky_ghi"}.difference(raw.columns))
    if missing:
        raise ValueError(
            f"{raw_path} is missing required NSRDB GHI columns: {missing}. "
            "Re-run phase2_finetuning/01_fetch_data.py."
        )

    hourly = _align_hourly_ghi(raw)
    if "zenith_angle" not in hourly.columns:
        legacy_cs = PROCESSED_DOWNLOADS_DIR / station_id / f"clearsky_{year}.csv"
        if legacy_cs.exists():
            zenith = pd.read_csv(legacy_cs, parse_dates=["datetime"])[["datetime", "zenith_angle"]]
            zenith = _align_hourly_ghi(
                zenith.assign(w_ghr=np.nan, nsrdb_clearsky_ghi=np.nan)[["datetime", "w_ghr", "nsrdb_clearsky_ghi", "zenith_angle"]]
            )[["datetime", "zenith_angle"]]
            hourly = hourly.merge(zenith, on="datetime", how="left")
            print(f"  {station_id} {year}: zenith_angle taken from legacy clearsky file.")
        else:
            raise ValueError(
                f"{raw_path} is missing zenith_angle. Re-run phase2_finetuning/01_fetch_data.py "
                "after the NSRDB zenith update."
            )
    return hourly


def load_and_merge_station_year(station_id: str, year: int) -> pd.DataFrame:
    """Load and merge one station-year from the hourly downloads."""
    processed_era5_path = PROCESSED_DOWNLOADS_DIR / station_id / f"era5_{year}.csv"
    raw_era5_path = MULTI_STATION_DOWNLOADS_DIR / station_id / f"era5_{year}.csv"
    era5_path = processed_era5_path if processed_era5_path.exists() else raw_era5_path
    era5 = pd.read_csv(era5_path, parse_dates=["datetime"])
    era5 = _align_hourly_era5(era5)
    ghi = load_hourly_ghi(station_id, year)

    df = era5.merge(ghi, on="datetime", how="inner")
    solar = _compute_solar_geometry(station_id, df["datetime"])
    df = df.merge(solar, on="datetime", how="left")

    df["w_ghr"] = pd.to_numeric(df["w_ghr"], errors="coerce").fillna(0.0).clip(lower=0)
    df["clear_sky_ghi"] = (
        pd.to_numeric(df["nsrdb_clearsky_ghi"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0)
    )
    df["zenith_angle"] = (
        pd.to_numeric(df["zenith_angle"], errors="coerce")
        .fillna(pd.to_numeric(df["zenith_angle_pvlib"], errors="coerce"))
        .fillna(95.0)
    )
    df["azimuth_angle"] = pd.to_numeric(df["azimuth_angle"], errors="coerce").fillna(180.0)

    for col in ["tcc", "lcc", "mcc", "hcc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0.0, 1.0)
    for col in ["u10", "v10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    df["wind_direction"] = np.arctan2(df["v10"], df["u10"])
    df["elevation_m"] = ELEVATION_MAP.get(station_id, 0.0)

    df["station_id"] = station_id
    print(f"  {station_id} {year}: {len(df)} rows merged")
    return df


def main() -> None:
    print(f"=== Building fine-tuning features for {YEARS} ===\n")

    all_dfs = []
    for station in STATIONS:
        for year in YEARS:
            all_dfs.append(load_and_merge_station_year(station["id"], year))

    df = pd.concat(all_dfs, ignore_index=True).sort_values(["station_id", "datetime"]).reset_index(drop=True)
    print(f"\n  Combined: {len(df)} rows")

    # Cloud Attenuation Factor: ratio of actual GHI to clear-sky GHI
    df["CAF"] = np.where(
        df["clear_sky_ghi"] > 1.0,
        (df["w_ghr"] / df["clear_sky_ghi"]).clip(0.0, 1.0),
        0.0,
    )

    # Cyclical time encodings
    df["hour"] = df["datetime"].dt.hour
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # Select final columns
    cols = [
        "station_id",
        "datetime",
        "w_ghr",
        "CAF",
        "clear_sky_ghi",
        "zenith_angle",
        "azimuth_angle",
        "tcc",
        "lcc",
        "mcc",
        "hcc",
        "wind_speed",
        "wind_direction",
        "elevation_m",
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
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.3f}")
        else:
            print(f"  {col}: (not in final CSV — replaced by derived feature)")


if __name__ == "__main__":
    main()
