"""Fetch multi-station NSRDB GHI and ERA5 (Open-Meteo) inputs for Phase 2.

ERA5 is pulled from the Open-Meteo archive (same source as Phase 3), so no
CDS API key or NetCDF dependency is required. GHI continues to come from
NREL NSRDB at 15-min resolution.

Outputs (per station, per year):
  downloads/multi_station/<station>/ghi_<year>.csv
  downloads/multi_station/<station>/era5_<year>.csv
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    DOWNLOADS_DIR,
    MULTI_STATION_DOWNLOADS_DIR,
    NREL_API_KEY,
    NREL_EMAIL,
    STATION_ELEVATIONS_CACHE,
    STATIONS,
    YEARS,
)

LOCAL_TIMEZONE = "Asia/Kolkata"
MAX_RETRIES = 5
RETRY_BACKOFF = 10  # seconds; doubles each retry

# Open-Meteo ERA5 archive variables (same set as Phase 3 for parity)
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

# Columns Phase 2's downstream pipeline requires from the ERA5 CSV
ERA5_REQUIRED_COLUMNS = {"datetime", "tcc", "lcc", "mcc", "hcc", "u10", "v10"}

NSRDB_REQUIRED_COLUMNS = {"datetime", "w_ghr", "nsrdb_clearsky_ghi", "zenith_angle"}
NSRDB_RENAME_MAP = {
    "GHI": "w_ghr",
    "Clearsky GHI": "nsrdb_clearsky_ghi",
    "Solar Zenith Angle": "zenith_angle",
}


def fetch_nrel_ghi(year: int, lat: float, lon: float, station_id: str) -> None:
    """Download NSRDB GHI, clear-sky GHI, and zenith for one station-year."""
    assert NREL_API_KEY, "Set NREL_API_KEY env var."
    station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
    station_dir.mkdir(parents=True, exist_ok=True)
    out = station_dir / f"ghi_{year}.csv"

    if out.exists():
        existing = pd.read_csv(out, nrows=2)
        missing = sorted(NSRDB_REQUIRED_COLUMNS.difference(existing.columns))
        if not missing:
            print(f"  Skip {station_id}/{out.name} (cached)")
            return
        print(f"  Regenerating {station_id}/{out.name} (missing {missing})")

    print(f"  NREL GHI {year} for {station_id} ({lat}, {lon})...")
    url = "https://developer.nrel.gov/api/nsrdb/v2/solar/msg-iodc-download.csv"
    payload = {
        "api_key": NREL_API_KEY,
        "full_name": "Research User",
        "email": NREL_EMAIL,
        "affiliation": "Research",
        "reason": "Academic",
        "wkt": f"POINT({lon} {lat})",
        "names": str(year),
        "attributes": "ghi,clearsky_ghi,solar_zenith_angle",
        "interval": "15",
        "utc": "false",
        "leap_day": "false",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=payload, timeout=180)
            if response.status_code == 200 and "Error" not in response.text[:200]:
                df = pd.read_csv(io.StringIO(response.text), skiprows=2)
                df.columns = [c.strip() for c in df.columns]
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
                missing_nsrdb_cols = sorted(NSRDB_REQUIRED_COLUMNS.difference(df.columns))
                if missing_nsrdb_cols:
                    raise RuntimeError(
                        f"NSRDB response for {station_id} {year} did not include required columns: {missing_nsrdb_cols}"
                    )
                for col in df.columns:
                    if col == "datetime":
                        continue
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["w_ghr"] = df["w_ghr"].clip(lower=0)
                df["nsrdb_clearsky_ghi"] = df["nsrdb_clearsky_ghi"].clip(lower=0)
                df.to_csv(out, index=False)
                print(f"    Saved {out.name} shape={df.shape}")
                return

            print(f"    NSRDB fetch failed for {station_id} {year} status={response.status_code}")
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


def download_era5_station_year(year: int, lat: float, lon: float, station_id: str) -> None:
    """Fetch ERA5 cloud cover + weather fields for one station-year from Open-Meteo."""
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
                    # Open-Meteo returns cloud cover as percent (0–100); convert to 0–1 fraction.
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

            # Convert wind speed + direction → u10, v10 components (meteorological convention)
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


def fetch_station_elevations() -> None:
    """Resolve per-station SRTM elevations from Open-Meteo and cache them.

    Grid-mode station lists in config.py default every box to the cluster
    center elevation, which makes `elevation_m` a constant column and useless
    as a spatial differentiator for the model. This one-shot lookup replaces
    those defaults with real per-site SRTM elevations, cached to
    `STATION_ELEVATIONS_CACHE` so subsequent runs skip the API call.
    """
    STATION_ELEVATIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, float] = {}
    if STATION_ELEVATIONS_CACHE.exists():
        try:
            with open(STATION_ELEVATIONS_CACHE, "r", encoding="utf-8") as fh:
                existing = {str(k): float(v) for k, v in json.load(fh).items()}
        except (OSError, ValueError):
            existing = {}

    missing = [s for s in STATIONS if s["id"] not in existing]
    if not missing:
        print(f"  Skip station elevations (cache complete: {STATION_ELEVATIONS_CACHE.name})")
        return

    print(f"  Fetching SRTM elevations for {len(missing)} station(s) via Open-Meteo...")
    lats = ",".join(f"{s['lat']}" for s in missing)
    lons = ",".join(f"{s['lon']}" for s in missing)
    url = "https://api.open-meteo.com/v1/elevation"
    params = {"latitude": lats, "longitude": lons}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            elevations = payload.get("elevation")
            if not isinstance(elevations, list) or len(elevations) != len(missing):
                raise RuntimeError(
                    f"Unexpected elevation response: expected {len(missing)} values, "
                    f"got {elevations!r}"
                )
            break
        except Exception as exc:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"    Retry {attempt}/{MAX_RETRIES} after {wait}s — {exc}")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Station elevation lookup failed after {MAX_RETRIES} retries"
                ) from exc

    for station, elev in zip(missing, elevations):
        existing[station["id"]] = float(elev)
        print(f"    {station['id']:<22} lat={station['lat']:.4f}  lon={station['lon']:.4f}  elev={float(elev):.1f} m")

    with open(STATION_ELEVATIONS_CACHE, "w", encoding="utf-8") as fh:
        json.dump(dict(sorted(existing.items())), fh, indent=2)
    print(f"    Saved {STATION_ELEVATIONS_CACHE}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Phase 2 fine-tuning data acquisition for years {YEARS}")
    print(f"  Stations: {', '.join(station['id'] for station in STATIONS)}")
    print("=" * 60)

    print("\n=== Station elevations (Open-Meteo SRTM) ===")
    fetch_station_elevations()

    print("\n=== GHI (NREL NSRDB) ===")
    for station in STATIONS:
        for year in YEARS:
            fetch_nrel_ghi(year, station["lat"], station["lon"], station["id"])

    print("\n=== ERA5 (Open-Meteo archive) ===")
    for station in STATIONS:
        for year in YEARS:
            download_era5_station_year(year, station["lat"], station["lon"], station["id"])

    print("\n=== Verification ===")
    for station in STATIONS:
        for year in YEARS:
            for filename in [f"ghi_{year}.csv", f"era5_{year}.csv"]:
                path = MULTI_STATION_DOWNLOADS_DIR / station["id"] / filename
                if path.exists():
                    df = pd.read_csv(path, nrows=2)
                    print(f"  OK {station['id']}/{filename:<18} cols={list(df.columns)}")
                else:
                    print(f"  MISSING {station['id']}/{filename}")

    print("\nDone. Next: python phase2_finetuning/02_build_features.py")
