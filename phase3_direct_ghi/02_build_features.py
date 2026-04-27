"""Merge GHI + gridded weather + clear-sky features into the Direct GHI dataset.

Reads from:  downloads/multi_station/<station>/  (all hourly, IST)
Writes to:   dataset/processed_data_2017_2019.csv

Key differences from Phase 2:
  - Target = raw GHI (w_ghr) in W/m², NOT CAF
  - Derives wind, cloud-transition, and humidity/temperature features
  - Includes PVLib solar geometry + clear-sky GHI
  - Adds simple neighbor-context means from the 3x3 grid
  - Injects elevation_m as a static spatial feature per station
"""
import sys
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_DIR,
    ELEVATION_MAP,
    GRID_NEIGHBOR_COUNT,
    MULTI_STATION_DOWNLOADS_DIR,
    PROCESSED_DOWNLOADS_DIR,
    STATIONS,
    YEARS,
)


CLOUD_COLUMNS = ["tcc", "lcc", "mcc", "hcc"]
NSRDB_FLUX_COLUMNS = [
    "w_ghr",
    "nsrdb_clearsky_ghi",
    "nsrdb_clearsky_dni",
    "nsrdb_clearsky_dhi",
]
NSRDB_POINT_COLUMNS = [
    "nsrdb_zenith_angle",
    "nsrdb_temperature",
    "nsrdb_relative_humidity",
    "nsrdb_dew_point",
    "nsrdb_surface_pressure",
    "nsrdb_wind_speed",
    "nsrdb_wind_direction",
    "nsrdb_cloud_type",
]
ERA5_ALIGN_COLUMNS = [
    *CLOUD_COLUMNS,
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_gusts_10m",
    "u10",
    "v10",
]
CLEARSKY_ALIGN_COLUMNS = [
    "zenith_angle",
    "azimuth_angle",
    "clearsky_ghi",
    "clearsky_dni",
    "clearsky_dhi",
]
WEATHER_DEFAULTS = {
    "temperature_2m": 30.0,
    "relative_humidity_2m": 50.0,
    "dew_point_2m": 20.0,
    "surface_pressure": 1013.25,
    "wind_gusts_10m": 0.0,
}
DELTA_COLUMNS = [
    *CLOUD_COLUMNS,
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed",
]
NEIGHBOR_MEAN_COLUMNS = [
    "tcc",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed",
]
SOLAR_POINT_COLUMNS = ["zenith_angle", "azimuth_angle"]
SOLAR_FLUX_COLUMNS = ["clearsky_ghi", "clearsky_dni", "clearsky_dhi"]


def _finalize_hourly_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly.copy()
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])
    return hourly.sort_values("datetime").reset_index(drop=True)


def align_point_to_hourly_grid(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Align point-in-time values to HH:00 without averaging across adjacent hours."""
    out = df.sort_values("datetime").reset_index(drop=True).copy()
    minutes = sorted(out["datetime"].dt.minute.dropna().unique().tolist())

    if minutes == [0]:
        return _finalize_hourly_frame(out[["datetime", *[c for c in value_cols if c in out.columns]]])

    # Quarter-hour point values: sample the exact top-of-hour instant.
    if set(minutes).issubset({0, 15, 30, 45}):
        hourly = out[out["datetime"].dt.minute == 0][["datetime", *[c for c in value_cols if c in out.columns]]]
        return _finalize_hourly_frame(hourly)

    # Hourly values stamped at :30: shift them back to the corresponding HH:00.
    if minutes == [30]:
        shifted = out[["datetime", *[c for c in value_cols if c in out.columns]]].copy()
        shifted["datetime"] = shifted["datetime"] - pd.Timedelta(minutes=30)
        return _finalize_hourly_frame(shifted)

    return _finalize_hourly_frame(out[["datetime", *[c for c in value_cols if c in out.columns]]])


def align_mean_to_hourly_grid(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Aggregate sub-hourly flux-like values to hourly means on the HH:00 grid."""
    out = df.sort_values("datetime").reset_index(drop=True).copy()
    minutes = sorted(out["datetime"].dt.minute.dropna().unique().tolist())

    if minutes == [0]:
        return _finalize_hourly_frame(out[["datetime", *[c for c in value_cols if c in out.columns]]])

    cols = [c for c in value_cols if c in out.columns]
    indexed = out.set_index("datetime")[cols].apply(pd.to_numeric, errors="coerce")

    if set(minutes).issubset({0, 15, 30, 45}):
        hourly = indexed.resample("1h", label="left", closed="left").mean().reset_index()
        return _finalize_hourly_frame(hourly)

    if minutes == [30]:
        shifted = indexed.copy()
        shifted.index = shifted.index - pd.Timedelta(minutes=30)
        return _finalize_hourly_frame(shifted.reset_index())

    return _finalize_hourly_frame(indexed.reset_index())


def align_nsrdb_interval_to_hourly_grid(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Aggregate NSRDB 15-minute interval data to hourly, preserving hour-end labels."""
    out = df.sort_values("datetime").reset_index(drop=True).copy()
    cols = [c for c in value_cols if c in out.columns]
    if not cols:
        return _finalize_hourly_frame(out[["datetime"]].drop_duplicates())

    minutes = sorted(out["datetime"].dt.minute.dropna().unique().tolist())
    indexed = out.set_index("datetime")[cols].apply(pd.to_numeric, errors="coerce")

    if set(minutes).issubset({0, 15, 30, 45}):
        hourly = indexed.resample("1h", label="right", closed="right").mean().reset_index()
        return _finalize_hourly_frame(hourly)

    if minutes == [0]:
        return _finalize_hourly_frame(indexed.reset_index())

    return _finalize_hourly_frame(indexed.reset_index())


def _ensure_clearsky_columns(cs: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """Backfill missing PVLib solar geometry / clear-sky columns for old caches."""
    required = {"zenith_angle", "azimuth_angle", "clearsky_ghi", "clearsky_dni", "clearsky_dhi"}
    missing = sorted(required.difference(cs.columns))
    if not missing:
        return cs

    print(f"    WARNING: {station_id} clear-sky file missing {missing}; computing on-the-fly.")
    from pvlib.location import Location

    station_data = next(s for s in STATIONS if s["id"] == station_id)
    site = Location(
        station_data["lat"],
        station_data["lon"],
        tz="Asia/Kolkata",
        altitude=station_data["alt_m"],
    )
    times = pd.DatetimeIndex(cs["datetime"]).tz_localize("Asia/Kolkata")
    solpos = site.get_solarposition(times)
    clearsky = site.get_clearsky(times, model="ineichen")

    cs = cs.copy()
    def _coerce_or_fill(column_name: str, fallback: pd.Series) -> pd.Series:
        base = cs[column_name] if column_name in cs.columns else pd.Series(fallback.values, index=cs.index)
        coerced = pd.to_numeric(base, errors="coerce")
        missing_mask = coerced.isna()
        if missing_mask.any():
            coerced.loc[missing_mask] = fallback.values[missing_mask]
        return coerced

    cs["zenith_angle"] = _coerce_or_fill("zenith_angle", solpos["apparent_zenith"])
    cs["azimuth_angle"] = _coerce_or_fill("azimuth_angle", solpos["azimuth"])
    cs["clearsky_ghi"] = _coerce_or_fill("clearsky_ghi", clearsky["ghi"])
    cs["clearsky_dni"] = _coerce_or_fill("clearsky_dni", clearsky["dni"])
    cs["clearsky_dhi"] = _coerce_or_fill("clearsky_dhi", clearsky["dhi"])
    return cs


def _build_neighbor_map() -> dict[str, list[str]]:
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in km between two station centers."""
        earth_radius_km = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = (
            sin(dlat / 2) ** 2
            + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        )
        return 2 * earth_radius_km * asin(sqrt(a))

    station_meta = {station["id"]: station for station in STATIONS}
    neighbor_map: dict[str, list[str]] = {}
    for station_id, station in station_meta.items():
        ranked_neighbors = sorted(
            (
                (
                    other_id,
                    _haversine_km(station["lat"], station["lon"], other_station["lat"], other_station["lon"]),
                )
                for other_id, other_station in station_meta.items()
                if other_id != station_id
            ),
            key=lambda item: item[1],
        )
        neighbor_map[station_id] = [other_id for other_id, _ in ranked_neighbors[:GRID_NEIGHBOR_COUNT]]
    return neighbor_map


def load_and_merge_station_year(station_id: str, year: int) -> pd.DataFrame:
    """Load and merge one station-year from the multi-station downloads."""
    station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id

    # Try processed dir first, fall back to multi_station dir
    for base_dir in [PROCESSED_DOWNLOADS_DIR / station_id, station_dir]:
        ghi_path = base_dir / f"ghi_{year}.csv"
        era5_path = base_dir / f"era5_{year}.csv"
        cs_path = base_dir / f"clearsky_{year}.csv"
        if ghi_path.exists() and era5_path.exists() and cs_path.exists():
            break
    else:
        # Final fallback — just use multi_station
        ghi_path = station_dir / f"ghi_{year}.csv"
        era5_path = station_dir / f"era5_{year}.csv"
        cs_path = station_dir / f"clearsky_{year}.csv"

    ghi = pd.read_csv(ghi_path, parse_dates=["datetime"])
    era5 = pd.read_csv(era5_path, parse_dates=["datetime"])
    cs = pd.read_csv(cs_path, parse_dates=["datetime"])
    cs = _ensure_clearsky_columns(cs, station_id)
    if "nsrdb_clearsky_ghi" not in ghi.columns:
        print(
            f"    WARNING: {station_id} {year} ghi file lacks NSRDB clear-sky columns; "
            "falling back to PVLib clear-sky, which is not source-aligned."
        )

    # Source-aware hourly alignment:
    # - NSRDB interval data is aggregated with hour-end labels
    # - PVLib azimuth remains a point-in-time geometry fallback
    # - hourly weather stays on its native HH:00 grid
    ghi = align_nsrdb_interval_to_hourly_grid(ghi, [*NSRDB_FLUX_COLUMNS, *NSRDB_POINT_COLUMNS])
    era5 = align_point_to_hourly_grid(era5, ERA5_ALIGN_COLUMNS)
    cs_point = align_point_to_hourly_grid(cs, SOLAR_POINT_COLUMNS)
    cs_flux = align_nsrdb_interval_to_hourly_grid(cs, SOLAR_FLUX_COLUMNS)
    cs = cs_point.merge(cs_flux, on="datetime", how="inner")

    # Merge the source-aligned pieces.
    df = (
        era5.merge(cs, on="datetime", how="inner")
            .merge(ghi, on="datetime", how="inner")
    )

    # Clean & clip GHI
    df["w_ghr"] = pd.to_numeric(df["w_ghr"], errors="coerce").fillna(0.0).clip(lower=0)

    # Prefer NSRDB clear-sky and meteorology when available because they share
    # the same quarter-hour source and interval semantics as the target.
    df["zenith_angle"] = pd.to_numeric(
        df.get("nsrdb_zenith_angle", df["zenith_angle"]),
        errors="coerce",
    ).fillna(pd.to_numeric(df["zenith_angle"], errors="coerce")).fillna(95.0)
    df["azimuth_angle"] = pd.to_numeric(df["azimuth_angle"], errors="coerce").fillna(180.0)

    for target_col, source_col, fallback_col in [
        ("clearsky_ghi", "nsrdb_clearsky_ghi", "clearsky_ghi"),
        ("clearsky_dni", "nsrdb_clearsky_dni", "clearsky_dni"),
        ("clearsky_dhi", "nsrdb_clearsky_dhi", "clearsky_dhi"),
    ]:
        fallback = pd.to_numeric(df.get(fallback_col, 0.0), errors="coerce")
        source = pd.to_numeric(df.get(source_col, fallback), errors="coerce")
        df[target_col] = source.fillna(fallback).fillna(0.0).clip(lower=0.0)

    for target_col, nsrdb_col, era5_col, default in [
        ("temperature_2m", "nsrdb_temperature", "temperature_2m", WEATHER_DEFAULTS["temperature_2m"]),
        ("relative_humidity_2m", "nsrdb_relative_humidity", "relative_humidity_2m", WEATHER_DEFAULTS["relative_humidity_2m"]),
        ("dew_point_2m", "nsrdb_dew_point", "dew_point_2m", WEATHER_DEFAULTS["dew_point_2m"]),
        ("surface_pressure", "nsrdb_surface_pressure", "surface_pressure", WEATHER_DEFAULTS["surface_pressure"]),
    ]:
        era5_vals = pd.to_numeric(df.get(era5_col, default), errors="coerce")
        nsrdb_vals = pd.to_numeric(df.get(nsrdb_col, era5_vals), errors="coerce")
        df[target_col] = nsrdb_vals.fillna(era5_vals).fillna(default)

    # Cloud cover fractions
    for col in CLOUD_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    if "wind_gusts_10m" not in df.columns:
        print(f"    WARNING: {station_id} {year} missing wind_gusts_10m; filling with default 0.0.")
        df["wind_gusts_10m"] = WEATHER_DEFAULTS["wind_gusts_10m"]
    df["wind_gusts_10m"] = pd.to_numeric(df["wind_gusts_10m"], errors="coerce").fillna(WEATHER_DEFAULTS["wind_gusts_10m"])

    # Wind: prefer NSRDB speed/direction when available; otherwise derive from Open-Meteo u/v.
    df["u10"] = pd.to_numeric(df.get("u10", 0.0), errors="coerce").fillna(0.0)
    df["v10"] = pd.to_numeric(df.get("v10", 0.0), errors="coerce").fillna(0.0)
    fallback_wind_speed = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    fallback_wind_direction = np.arctan2(df["v10"], df["u10"])
    df["wind_speed"] = pd.to_numeric(
        df.get("nsrdb_wind_speed", fallback_wind_speed),
        errors="coerce",
    ).fillna(fallback_wind_speed)
    nsrdb_wd = pd.to_numeric(
        df.get("nsrdb_wind_direction", pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    nsrdb_wd_rad = np.deg2rad((270.0 - nsrdb_wd) % 360.0)
    df["wind_direction"] = nsrdb_wd_rad.fillna(fallback_wind_direction)

    # Static spatial feature: elevation
    df["elevation_m"] = ELEVATION_MAP.get(station_id, 0.0)

    df["station_id"] = station_id
    print(f"  {station_id} {year}: {len(df)} rows merged")
    return df


def main() -> None:
    print(f"=== Phase 3: Building Direct GHI features for {YEARS} ===\n")

    all_dfs = []
    for station in STATIONS:
        for year in YEARS:
            all_dfs.append(load_and_merge_station_year(station["id"], year))

    df = pd.concat(all_dfs, ignore_index=True).sort_values(["station_id", "datetime"]).reset_index(drop=True)
    print(f"\n  Combined: {len(df)} rows")

    # Derived transition features on the full station timeline.
    clear_sky_index = np.where(
        df["clearsky_ghi"] > 20.0,
        df["w_ghr"] / df["clearsky_ghi"].clip(lower=20.0),
        0.0,
    )
    df["clear_sky_index"] = pd.Series(clear_sky_index, index=df.index).clip(lower=0.0)

    for col in DELTA_COLUMNS:
        df[f"{col}_delta_1h"] = (
            df.groupby("station_id", sort=False)[col]
            .diff()
            .fillna(0.0)
        )

    # Neighbor context uses the immediate surrounding cells of each box.
    neighbor_map = _build_neighbor_map()
    for feature in NEIGHBOR_MEAN_COLUMNS:
        pivot = df.pivot(index="datetime", columns="station_id", values=feature)
        out_col = f"neighbor_{feature}_mean"
        df[out_col] = np.nan
        for station_id, neighbors in neighbor_map.items():
            if not neighbors:
                continue
            neighbor_mean = pivot[neighbors].mean(axis=1)
            station_mask = df["station_id"] == station_id
            df.loc[station_mask, out_col] = (
                df.loc[station_mask, "datetime"]
                .map(neighbor_mean)
                .to_numpy()
            )

    # Cyclical time encodings
    df["hour"] = df["datetime"].dt.hour
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # Select final columns. clearsky_ghi is included as a known driver, while
    # clear_sky_index remains past-only via config to avoid target leakage.
    cols = [
        "station_id",
        "datetime",
        "w_ghr",             # TARGET: direct GHI in W/m²
        "zenith_angle",
        "azimuth_angle",
        "clearsky_ghi",
        "clear_sky_index",
        "tcc", "lcc", "mcc", "hcc",
        "tcc_delta_1h", "lcc_delta_1h", "mcc_delta_1h", "hcc_delta_1h",
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "surface_pressure",
        "temperature_2m_delta_1h",
        "relative_humidity_2m_delta_1h",
        "surface_pressure_delta_1h",
        "wind_speed",
        "wind_direction",
        "wind_gusts_10m",
        "wind_speed_delta_1h",
        "neighbor_tcc_mean",
        "neighbor_temperature_2m_mean",
        "neighbor_relative_humidity_2m_mean",
        "neighbor_wind_speed_mean",
        "elevation_m",       # Static spatial feature (alt_m from station config)
        "hour_sin", "hour_cos",
        "doy_sin", "doy_cos",
    ]
    df = df[cols].dropna().sort_values(["station_id", "datetime"]).reset_index(drop=True)

    out = DATASET_DIR / "processed_data_2017_2019.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out} shape={df.shape}")
    print(f"  Stations: {df['station_id'].nunique()} {sorted(df['station_id'].unique().tolist())}")
    print(f"  Range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  GHI w_ghr mean/std: {df['w_ghr'].mean():.1f} / {df['w_ghr'].std():.1f}")
    print(f"  clearsky_ghi mean: {df['clearsky_ghi'].mean():.1f} W/m²")
    print(f"  clear_sky_index mean: {df['clear_sky_index'].mean():.3f}")
    print(f"  wind_speed mean: {df['wind_speed'].mean():.2f} m/s")
    print(f"  temperature_2m mean: {df['temperature_2m'].mean():.2f} °C")
    print(f"  relative_humidity_2m mean: {df['relative_humidity_2m'].mean():.2f} %")
    print(f"  elevation_m unique: {sorted(df['elevation_m'].unique().tolist())}")
    for col in CLOUD_COLUMNS:
        print(f"  {col}: mean={df[col].mean():.3f}")


if __name__ == "__main__":
    main()
