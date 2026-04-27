"""Fetch NSRDB solar data, gridded weather, and solar geometry files for Phase 3.

This version uses the simpler direct-download logic from fetch_downloads_only.py:
  - NREL NSRDB for GHI + source-aligned clear-sky / met fields
  - PVLib for solar azimuth fallback / geometry support
  - Open-Meteo archive for hourly cloud cover + transition-weather features
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    MULTI_STATION_DOWNLOADS_DIR,
    NREL_API_KEY,
    NREL_EMAIL,
    STATIONS,
    YEARS,
)

LOCAL_TIMEZONE = "Asia/Kolkata"
MAX_RETRIES = 5
RETRY_BACKOFF = 10

OPEN_METEO_HOURLY_VARS = [
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

ERA5_REQUIRED_COLUMNS = {
    "datetime",
    "tcc",
    "lcc",
    "mcc",
    "hcc",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "u10",
    "v10",
    "wind_gusts_10m",
}

CLEARSKY_REQUIRED_COLUMNS = {
    "datetime",
    "zenith_angle",
    "azimuth_angle",
    "clearsky_ghi",
    "clearsky_dni",
    "clearsky_dhi",
}

NSRDB_ATTRIBUTE_COLUMNS = {
    "datetime",
    "w_ghr",
    "nsrdb_clearsky_ghi",
    "nsrdb_clearsky_dni",
    "nsrdb_clearsky_dhi",
    "nsrdb_zenith_angle",
    "nsrdb_temperature",
    "nsrdb_relative_humidity",
    "nsrdb_dew_point",
    "nsrdb_surface_pressure",
    "nsrdb_wind_speed",
    "nsrdb_wind_direction",
    "nsrdb_cloud_type",
}

NSRDB_ATTRIBUTES = ",".join([
    "ghi",
    "clearsky_ghi",
    "clearsky_dni",
    "clearsky_dhi",
    "solar_zenith_angle",
    "air_temperature",
    "relative_humidity",
    "dew_point",
    "surface_pressure",
    "wind_speed",
    "wind_direction",
    "cloud_type",
])

NSRDB_RENAME_MAP = {
    "GHI": "w_ghr",
    "Clearsky GHI": "nsrdb_clearsky_ghi",
    "Clearsky DNI": "nsrdb_clearsky_dni",
    "Clearsky DHI": "nsrdb_clearsky_dhi",
    "Solar Zenith Angle": "nsrdb_zenith_angle",
    "Temperature": "nsrdb_temperature",
    "Relative Humidity": "nsrdb_relative_humidity",
    "Dew Point": "nsrdb_dew_point",
    "Pressure": "nsrdb_surface_pressure",
    "Wind Speed": "nsrdb_wind_speed",
    "Wind Direction": "nsrdb_wind_direction",
    "Cloud Type": "nsrdb_cloud_type",
}


def fetch_nrel_ghi(year: int, lat: float, lon: float, station_id: str) -> None:
    """Download raw 15-minute NSRDB solar + meteorology fields."""
    assert NREL_API_KEY, "Set NREL_API_KEY env var."

    station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
    station_dir.mkdir(parents=True, exist_ok=True)
    out = station_dir / f"ghi_{year}.csv"
    if out.exists():
        existing = pd.read_csv(out, nrows=2)
        missing = sorted(NSRDB_ATTRIBUTE_COLUMNS.difference(existing.columns))
        if not missing:
            print(f"  Skip {station_id}/{out.name} (cached)")
            return
        print(f"  Regenerating {station_id}/{out.name} (missing {missing})")

    print(f"  NREL GHI timezone: NSRDB local time because utc=false ({station_id} {year})")
    url = "https://developer.nrel.gov/api/nsrdb/v2/solar/msg-iodc-download.csv"
    payload = {
        "api_key": NREL_API_KEY,
        "full_name": "Research User",
        "email": NREL_EMAIL,
        "affiliation": "Personal/Academic",
        "reason": "Research",
        "wkt": f"POINT({lon} {lat})",
        "names": str(year),
        "attributes": NSRDB_ATTRIBUTES,
        "interval": "15",
        "utc": "false",
        "leap_day": "false",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=payload, timeout=180)
            if response.status_code == 200 and "Error" not in response.text[:200]:
                df = pd.read_csv(io.StringIO(response.text), skiprows=2)
                df.columns = [col.strip() for col in df.columns]
                df["datetime"] = pd.to_datetime(
                    df[["Year", "Month", "Day", "Hour", "Minute"]].rename(
                        columns={
                            "Year": "year",
                            "Month": "month",
                            "Day": "day",
                            "Hour": "hour",
                            "Minute": "minute",
                        }
                    )
                )
                keep_cols = ["datetime", *[col for col in NSRDB_RENAME_MAP if col in df.columns]]
                df = df[keep_cols].rename(columns=NSRDB_RENAME_MAP)
                for col in df.columns:
                    if col == "datetime":
                        continue
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                if "w_ghr" in df.columns:
                    df["w_ghr"] = df["w_ghr"].clip(lower=0)
                for col in ["nsrdb_clearsky_ghi", "nsrdb_clearsky_dni", "nsrdb_clearsky_dhi"]:
                    if col in df.columns:
                        df[col] = df[col].clip(lower=0)
                df.to_csv(out, index=False)
                print(f"    Saved {out.name} shape={df.shape} resolution=15min")
                return

            print(f"    NREL failed: station={station_id} year={year} status={response.status_code}")
            print(response.text[:500])
            raise RuntimeError(f"HTTP {response.status_code}")
        except Exception as exc:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"    Retry {attempt}/{MAX_RETRIES} after {wait}s — {exc}")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"NREL GHI download failed for {station_id} {year} after {MAX_RETRIES} retries"
                ) from exc


def generate_clearsky(year: int, lat: float, lon: float, alt_m: float, station_id: str) -> None:
    """Generate 15-minute solar position and clear-sky irradiance via PVLib."""
    from pvlib.location import Location

    station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
    station_dir.mkdir(parents=True, exist_ok=True)
    out = station_dir / f"clearsky_{year}.csv"

    if out.exists():
        existing = pd.read_csv(out, nrows=2)
        missing = sorted(CLEARSKY_REQUIRED_COLUMNS.difference(existing.columns))
        if not missing:
            print(f"  Skip {station_id}/{out.name} (cached)")
            return
        print(f"  Regenerating {station_id}/{out.name} (missing {missing})")

    times = pd.date_range(
        f"{year}-01-01",
        f"{year}-12-31 23:45",
        freq="15min",
        tz=LOCAL_TIMEZONE,
    )
    print(f"  Solar position timezone: {LOCAL_TIMEZONE} ({station_id} {year})")
    site = Location(lat, lon, tz=LOCAL_TIMEZONE, altitude=alt_m)
    solpos = site.get_solarposition(times)
    clearsky = site.get_clearsky(times, model="ineichen")

    df = pd.DataFrame(
        {
            "datetime": times.tz_localize(None),
            "zenith_angle": solpos["apparent_zenith"].values,
            "azimuth_angle": solpos["azimuth"].values,
            "clearsky_ghi": clearsky["ghi"].values,
            "clearsky_dni": clearsky["dni"].values,
            "clearsky_dhi": clearsky["dhi"].values,
        }
    )
    df.to_csv(out, index=False)
    print(f"    Saved {out.name} shape={df.shape} resolution=15min")


def download_era5_station_year(year: int, lat: float, lon: float, station_id: str) -> None:
    """Fetch hourly cloud cover + transition-weather features from Open-Meteo."""
    station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
    station_dir.mkdir(parents=True, exist_ok=True)
    out = station_dir / f"era5_{year}.csv"

    if out.exists():
        existing = pd.read_csv(out, nrows=2)
        missing = sorted(ERA5_REQUIRED_COLUMNS.difference(existing.columns))
        if not missing:
            print(f"  Skip {station_id}/{out.name} (cached)")
            return
        print(f"  Regenerating {station_id}/{out.name} (missing {missing})")

    print(f"  ERA5 (Open-Meteo) timezone: {LOCAL_TIMEZONE} ({station_id} {year})")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": ",".join(OPEN_METEO_HOURLY_VARS),
        "timezone": LOCAL_TIMEZONE,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
            hourly = data["hourly"]

            df = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(hourly["time"]),
                    "tcc": np.array(hourly["cloud_cover"], dtype=float) / 100.0,
                    "lcc": np.array(hourly["cloud_cover_low"], dtype=float) / 100.0,
                    "mcc": np.array(hourly["cloud_cover_mid"], dtype=float) / 100.0,
                    "hcc": np.array(hourly["cloud_cover_high"], dtype=float) / 100.0,
                    "temperature_2m": np.array(hourly["temperature_2m"], dtype=float),
                    "relative_humidity_2m": np.array(hourly["relative_humidity_2m"], dtype=float),
                    "dew_point_2m": np.array(hourly["dew_point_2m"], dtype=float),
                    "surface_pressure": np.array(hourly["surface_pressure"], dtype=float),
                    "wind_gusts_10m": np.array(hourly["wind_gusts_10m"], dtype=float),
                }
            )

            ws = np.array(hourly["wind_speed_10m"], dtype=float)
            wd_rad = np.deg2rad(np.array(hourly["wind_direction_10m"], dtype=float))
            df["u10"] = -ws * np.sin(wd_rad)
            df["v10"] = -ws * np.cos(wd_rad)
            df = df.ffill().bfill()

            df.to_csv(out, index=False)
            print(f"    Saved {out.name} shape={df.shape} resolution=60min")
            return
        except Exception as exc:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"    Retry {attempt}/{MAX_RETRIES} after {wait}s — {exc}")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"ERA5 download failed for {station_id} {year} after {MAX_RETRIES} retries"
                ) from exc


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Phase 3: Direct GHI — Data acquisition for years {YEARS}")
    print(f"  Cluster: {', '.join(station['id'] for station in STATIONS)}")
    print("=" * 60)

    print("\n=== GHI (NREL NSRDB) ===")
    for station in STATIONS:
        for year in YEARS:
            fetch_nrel_ghi(year, station["lat"], station["lon"], station["id"])

    print("\n=== ERA5 (Open-Meteo) ===")
    for station in STATIONS:
        for year in YEARS:
            download_era5_station_year(year, station["lat"], station["lon"], station["id"])

    print("\n=== Clear-sky + Azimuth (PVLib) ===")
    for station in STATIONS:
        for year in YEARS:
            generate_clearsky(year, station["lat"], station["lon"], station["alt_m"], station["id"])

    print("\n=== Verification ===")
    for station in STATIONS:
        for year in YEARS:
            for filename in [f"ghi_{year}.csv", f"era5_{year}.csv", f"clearsky_{year}.csv"]:
                path = MULTI_STATION_DOWNLOADS_DIR / station["id"] / filename
                if path.exists():
                    df = pd.read_csv(path, nrows=2)
                    print(f"  OK {station['id']}/{filename:<18} cols={list(df.columns)}")
                else:
                    print(f"  MISSING {station['id']}/{filename}")

    print("\nDone. Next: python 02_build_features.py")
