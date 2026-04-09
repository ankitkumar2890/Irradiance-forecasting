"""Merge GHI + direct ERA5 + clear-sky into the ERA5-only fine-tuning dataset."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DOWNLOADS_DIR, DATASET_DIR, YEARS


def load_and_merge_year(year):
    """Load and merge GHI, direct ERA5, and clear-sky for a single year."""
    ghi = pd.read_csv(DOWNLOADS_DIR / f"ghi_{year}.csv")
    era5 = pd.read_csv(DOWNLOADS_DIR / f"era5_{year}.csv")
    cs = pd.read_csv(DOWNLOADS_DIR / f"clearsky_{year}.csv")

    # Parse datetimes
    ghi["datetime"] = pd.to_datetime(ghi["datetime"])
    cs["datetime"] = pd.to_datetime(cs["datetime"])

    # NREL GHI is currently arriving on a :30 hourly grid.
    # Shift it onto the top-of-hour grid so it aligns with direct ERA5 and clear-sky.
    ghi_minutes = sorted(ghi["datetime"].dt.minute.dropna().unique().tolist())
    if ghi_minutes == [30]:
        ghi["datetime"] = ghi["datetime"] - pd.Timedelta(minutes=30)

    era5["datetime"] = pd.to_datetime(era5["datetime"])

    # Dedup
    for d in [ghi, era5]:
        d.drop_duplicates(subset=["datetime"], keep="first", inplace=True)

    # Interpolate clear-sky onto the GHI/ERA5 timestamp grid
    target_times = pd.Index(
        sorted(set(ghi["datetime"]).intersection(set(era5["datetime"])))
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
        era5.merge(cs, on="datetime", how="inner")
        .merge(ghi, on="datetime", how="left")
    )
    df["w_ghr"] = df["w_ghr"].fillna(0.0).clip(lower=0)
    df["zenith_angle"] = df["zenith_angle"].fillna(95.0)
    df["clear_sky_ghi"] = df["clear_sky_ghi"].fillna(0.0).clip(lower=0)
    for col in ["total_cloud_cover", "low_cloud_cover", "medium_cloud_cover", "high_cloud_cover"]:
        df[col] = df[col].clip(0.0, 1.0)
    for col in ["cloud_liquid_water", "cloud_ice_water", "water_vapour"]:
        df[col] = df[col].clip(lower=0.0)

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

    cols = [
        "datetime", "CAF", "clear_sky_ghi",
        "total_cloud_cover", "low_cloud_cover", "medium_cloud_cover", "high_cloud_cover",
        "cloud_liquid_water", "cloud_ice_water", "water_vapour",
        "zenith_angle", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    ]
    df = df[cols].dropna().sort_values("datetime").reset_index(drop=True)

    out = DATASET_DIR / "processed_data_2017_2019.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}  shape={df.shape}")
    print(f"  CAF: mean={df['CAF'].mean():.3f}  std={df['CAF'].std():.3f}")
    print(f"  Date range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")
    print(f"  total_cloud_cover: mean={df['total_cloud_cover'].mean():.3f}")


if __name__ == "__main__":
    main()
