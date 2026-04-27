"""Download GHI, clear-sky + azimuth, and ERA5 files for all Phase 3 stations.

Standalone version that uses Open-Meteo (free, no CDS key needed) for ERA5 data.
Phase 3 addition: also outputs solar azimuth_angle in clear-sky files.
"""
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
        print(f"Already exists, skipping: {out}")
        return

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
        "attributes": "ghi",
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
                ghi_col = next(col for col in df.columns if col.upper() == "GHI")
                df = df[["datetime", ghi_col]].rename(columns={ghi_col: "w_ghr"})
                df["w_ghr"] = pd.to_numeric(df["w_ghr"], errors="coerce").clip(lower=0)

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
#  Clear-sky + Azimuth (pvlib)
# ──────────────────────────────────────────────
def download_clearsky_file(year: int, lat: float, lon: float, alt_m: float, station_id: str) -> None:
    """Generate solar zenith angle and azimuth angle via PVLib."""
    from pvlib.location import Location

    station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
    station_dir.mkdir(parents=True, exist_ok=True)

    out = station_dir / f"clearsky_{year}.csv"
    if out.exists():
        # Check if existing file already has azimuth_angle
        existing = pd.read_csv(out, nrows=2)
        if "azimuth_angle" in existing.columns and "clear_sky_ghi" not in existing.columns:
            print(f"Already exists with azimuth, skipping: {out}")
            return
        else:
            print(f"Regenerating {out} (removing clear_sky_ghi, adding azimuth)...")

    # Generate times directly in local timezone (matching GHI which uses utc=false)
    times = pd.date_range(
        f"{year}-01-01",
        f"{year}-12-31 23:45",
        freq="15min",
        tz=LOCAL_TIMEZONE,
    )
    print(f"Solar position timezone: {LOCAL_TIMEZONE} ({station_id} {year})")
    site = Location(lat, lon, tz=LOCAL_TIMEZONE, altitude=alt_m)
    solpos = site.get_solarposition(times)

    df = pd.DataFrame(
        {
            "datetime": times.tz_localize(None),
            "zenith_angle": solpos["apparent_zenith"].values,
            "azimuth_angle": solpos["azimuth"].values,
        }
    )
    df.to_csv(out, index=False)
    print(f"Saved {out}")


def download_clearsky() -> None:
    for station in STATIONS:
        for year in YEARS:
            download_clearsky_file(
                year,
                station["lat"],
                station["lon"],
                station["alt_m"],
                station["id"],
            )


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
    download_clearsky()
    download_era5()
