"""Download NSRDB GHI and ERA5 files for all phase 2 stations."""
import io
import sys
import time
import numpy as np
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MULTI_STATION_DOWNLOADS_DIR,
    NREL_API_KEY,
    NREL_EMAIL,
    STATIONS,
    YEARS,
)

LOCAL_TIMEZONE = "Asia/Kolkata"
NSRDB_REQUIRED_COLUMNS = {"datetime", "w_ghr", "nsrdb_clearsky_ghi", "zenith_angle"}
NSRDB_RENAME_MAP = {
    "GHI": "w_ghr",
    "Clearsky GHI": "nsrdb_clearsky_ghi",
    "Solar Zenith Angle": "zenith_angle",
}

MAX_RETRIES = 5
RETRY_BACKOFF = 10  # seconds; doubles each retry

# Open-Meteo ERA5 archive variables to fetch
OPEN_METEO_HOURLY_VARS = [
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
]


# ──────────────────────────────────────────────
#  GHI (NREL NSRDB)
# ──────────────────────────────────────────────
def download_ghi_file(year: int, lat: float, lon: float, station_id: str) -> None:
    station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
    station_dir.mkdir(parents=True, exist_ok=True)

    out = station_dir / f"ghi_{year}.csv"
    if out.exists():
        existing = pd.read_csv(out, nrows=2)
        missing = sorted(NSRDB_REQUIRED_COLUMNS.difference(existing.columns))
        if not missing:
            print(f"Already exists, skipping: {out}")
            return
        print(f"Regenerating {out} because it is missing {missing}")

    print(f"NREL GHI source timezone: NSRDB local time because utc=false ({station_id} {year})")
    url = "https://developer.nrel.gov/api/nsrdb/v2/solar/msg-iodc-download.csv"
    payload = {
        "api_key": NREL_API_KEY,
        "full_name": "Research User",
        "email": NREL_EMAIL,
        "affiliation": "Personal/Academic",
        "reason": "Research",
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
                print(f"Saved {out}")
                return

            print(f"NREL GHI failed: station={station_id} year={year}")
            print(f"HTTP status: {response.status_code}")
            print(response.text[:500])
            raise RuntimeError(f"HTTP {response.status_code}")

        except Exception as e:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"  Retry {attempt}/{MAX_RETRIES} after {wait}s — {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"NREL GHI download failed for {station_id} {year} after {MAX_RETRIES} retries"
                ) from e


def download_ghi() -> None:
    for station in STATIONS:
        for year in YEARS:
            download_ghi_file(year, station["lat"], station["lon"], station["id"])


# ──────────────────────────────────────────────
#  ERA5 via Open-Meteo archive API (free, no key)
# ──────────────────────────────────────────────
def download_era5_station_year(year: int, lat: float, lon: float, station_id: str) -> None:
    """Fetch ERA5 cloud cover + wind for one station-year from Open-Meteo."""
    station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
    station_dir.mkdir(parents=True, exist_ok=True)

    out = station_dir / f"era5_{year}.csv"
    if out.exists():
        print(f"Already exists, skipping: {out}")
        return

    print(f"ERA5 (Open-Meteo) timezone: {LOCAL_TIMEZONE} ({station_id} {year})")
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
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            hourly = data["hourly"]
            df = pd.DataFrame({
                "datetime": pd.to_datetime(hourly["time"]),
                # Cloud cover: Open-Meteo returns % (0-100), convert to fraction (0-1)
                "tcc": np.array(hourly["cloud_cover"], dtype=float) / 100.0,
                "lcc": np.array(hourly["cloud_cover_low"], dtype=float) / 100.0,
                "mcc": np.array(hourly["cloud_cover_mid"], dtype=float) / 100.0,
                "hcc": np.array(hourly["cloud_cover_high"], dtype=float) / 100.0,
            })

            # Convert wind speed + direction → u10, v10 components
            ws = np.array(hourly["wind_speed_10m"], dtype=float)
            wd_rad = np.deg2rad(np.array(hourly["wind_direction_10m"], dtype=float))
            df["u10"] = -ws * np.sin(wd_rad)
            df["v10"] = -ws * np.cos(wd_rad)

            # Fill any missing values
            df = df.ffill().bfill()

            df.to_csv(out, index=False)
            print(f"Saved {out}  ({len(df)} rows)")
            return

        except Exception as e:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"  Retry {attempt}/{MAX_RETRIES} after {wait}s — {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"ERA5 download failed for {station_id} {year} after {MAX_RETRIES} retries"
                ) from e


def download_era5() -> None:
    for station in STATIONS:
        for year in YEARS:
            download_era5_station_year(
                year, station["lat"], station["lon"], station["id"]
            )


# ──────────────────────────────────────────────
if __name__ == "__main__":
    download_ghi()
    download_era5()
