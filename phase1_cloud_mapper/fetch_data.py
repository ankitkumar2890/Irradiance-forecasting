# ==============================================================================
# fetch_data.py — Download data required for Phase 1 CloudMapper
#
# Downloads and saves:
#
#   downloads/era5_2022_2023.csv       ← ERA5 cloud columns (CDS API)
#   downloads/era5_2017_2019.csv       ← ERA5 cloud columns (CDS API)
#   downloads/icon_2022.csv            ← ICON cloud cover (Open-Meteo)
#   downloads/icon_2023.csv            ← ICON cloud cover (Open-Meteo)
#   downloads/icon_2024.csv            ← ICON cloud cover (Open-Meteo)
#
# Prerequisites:
#   pip install cdsapi xarray netcdf4 requests pandas numpy
#
#   export CDSAPI_KEY="your-cds-personal-api-key"
# Run:
#   python fetch_data.py
#   python fetch_data.py --skip-era5      # if ERA5 already downloaded
#   python fetch_data.py --only-era5      # just ERA5 (slow, run first)
#   python fetch_data.py --extract-era5-only  # rebuild ERA5 CSVs from cached NetCDF
# ==============================================================================
import argparse
import sys
import calendar
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DOWNLOADS_DIR, ERA5_DIR,
    STATIONS,
    CDSAPI_KEY,
    SAMPLE_ONE_STATION_PER_TIMESTEP,
)


# ==============================================================================
# CELL 3+4: ERA5 — Download NetCDF + Extract to CSV
# ==============================================================================
ERA5_VARIABLES = [
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "total_column_cloud_liquid_water",
    "total_column_cloud_ice_water",
    "total_column_water_vapour",
]
ERA5_OUTPUT_COLS = [
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "cloud_liquid_water",
    "cloud_ice_water",
    "water_vapour",
]
AREA = [17.0, 72.5, 8.0, 81.5]  # [N, W, S, E] — South India bbox
GRID = [0.25, 0.25]
TARGET_HOURS = [f"{h:02d}:00" for h in range(24)]
MAX_FETCH_WORKERS = min(5, len(STATIONS))


def sample_one_station_per_timestep(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one station per timestamp using the configured station rotation."""
    if "station_id" not in df.columns:
        return df.sort_values("datetime").reset_index(drop=True)
    if not SAMPLE_ONE_STATION_PER_TIMESTEP:
        return df.sort_values(["datetime", "station_id"]).reset_index(drop=True)

    configured_stations = [station["id"] for station in STATIONS]
    available_stations = set(df["station_id"].dropna().unique())
    stations = [station_id for station_id in configured_stations if station_id in available_stations]
    stations.extend(sorted(available_stations - set(stations)))
    if len(stations) <= 1:
        return df.sort_values(["datetime", "station_id"]).reset_index(drop=True)

    selected = []
    for idx, (_, group) in enumerate(df.sort_values(["datetime", "station_id"]).groupby("datetime")):
        preferred_station = stations[idx % len(stations)]
        match = group[group["station_id"] == preferred_station]
        selected.append(match.iloc[0] if not match.empty else group.iloc[0])

    return pd.DataFrame(selected).reset_index(drop=True)


def download_era5_year(year: int) -> None:
    """Download ERA5 single-levels month-by-month for one year via CDS API v2."""
    import cdsapi 

    # Write .cdsapirc for this session
    cdsapirc = Path.home() / ".cdsapirc"
    cdsapirc.write_text(
        f"url: https://cds.climate.copernicus.eu/api\nkey: {CDSAPI_KEY}\n"
    )

    client = cdsapi.Client()
    year_dir = ERA5_DIR / str(year)
    year_dir.mkdir(exist_ok=True)

    for month in range(1, 13):
        out_file = year_dir / f"{year}_{month:02d}.nc"
        if out_file.exists():
            print(f"    Skip {year}-{month:02d} (cached)")
            continue

        ndays = calendar.monthrange(year, month)[1]
        request = {
            "product_type": "reanalysis",
            "variable":     ERA5_VARIABLES,
            "year":         str(year),
            "month":        f"{month:02d}",
            "day":          [f"{d:02d}" for d in range(1, ndays + 1)],
            "time":         TARGET_HOURS,
            "area":         AREA,
            "grid":         GRID,
            "data_format":  "netcdf",
        }
        print(f"    Downloading ERA5 {year}-{month:02d}...")
        client.retrieve("reanalysis-era5-single-levels", request, str(out_file))
        print(f"      Done: {out_file.name}")


def extract_era5_to_csv(years: list[int], output_filename: str) -> None:
    """Extract ERA5 point data from NetCDF files → single CSV."""
    import xarray as xr

    print(f"  Extracting ERA5 {years} → {output_filename}")
    all_dfs: list[pd.DataFrame] = []

    for year in years:
        year_dir = ERA5_DIR / str(year)
        for month in range(1, 13):
            nc_file = year_dir / f"{year}_{month:02d}.nc"
            if not nc_file.exists():
                print(f"    Missing: {nc_file}")
                continue

            ds = xr.open_dataset(nc_file)
            lat_d = "latitude" if "latitude" in ds.dims else "lat"
            lon_d = "longitude" if "longitude" in ds.dims else "lon"

            for station in STATIONS:
                ds_pt = ds.sel(
                    {lat_d: station["lat"], lon_d: station["lon"]},
                    method="nearest",
                ).squeeze(drop=True)
                df = ds_pt.to_dataframe().reset_index()
                df["station_id"] = station["id"]

                # Normalise time dimension name
                for tc in ("valid_time", "time", "forecast_time"):
                    if tc in df.columns:
                        df = df.rename(columns={tc: "datetime"})
                        break
                else:
                    print(f"    No time column in {nc_file.name} — skipping")
                    continue

                # Drop leftover coordinate columns
                drop = {lat_d, lon_d, "level", "number", "step", "surface",
                        "expver", "pressure_level"}
                df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

                # Canonical column names
                df = df.rename(columns={
                    "tcc": "total_cloud_cover",
                    "lcc": "low_cloud_cover",
                    "mcc": "medium_cloud_cover",
                    "hcc": "high_cloud_cover",
                    "tclw": "cloud_liquid_water",
                    "tciw": "cloud_ice_water",
                    "tcwv": "water_vapour",
                    "total_cloud_cover": "total_cloud_cover",
                    "low_cloud_cover": "low_cloud_cover",
                    "medium_cloud_cover": "medium_cloud_cover",
                    "high_cloud_cover": "high_cloud_cover",
                    "total_column_cloud_liquid_water": "cloud_liquid_water",
                    "total_column_cloud_ice_water": "cloud_ice_water",
                    "total_column_water_vapour": "water_vapour",
                })
                all_dfs.append(df)
            ds.close()

    if not all_dfs:
        raise RuntimeError(f"No ERA5 data found for years={years}")

    out_df = pd.concat(all_dfs, ignore_index=True).assign(
        datetime=lambda d: pd.to_datetime(d["datetime"], utc=True)
    )
    output_cols = ["datetime", "station_id"] + [col for col in ERA5_OUTPUT_COLS if col in out_df.columns]
    out_df = (
        out_df[output_cols]
        .dropna(subset=["datetime"])
        .sort_values(["datetime", "station_id"])
        .reset_index(drop=True)
    )
    out_df = sample_one_station_per_timestep(out_df)
    path = DOWNLOADS_DIR / output_filename
    out_df.to_csv(path, index=False)
    print(f"    {output_filename}  shape={out_df.shape}  "
          f"range={out_df['datetime'].iloc[0]} → {out_df['datetime'].iloc[-1]}")
    print(f"      sampled stations={out_df['station_id'].value_counts().to_dict()}")


def fetch_all_era5() -> None:
    """Download and extract ERA5 for all needed years."""
    assert CDSAPI_KEY, (
        "CDSAPI_KEY not set. Run:\n  export CDSAPI_KEY='your-key'\n"
        "Get your key from: https://cds.climate.copernicus.eu/profile"
    )

    print("\n=== ERA5 Downloads (CDS API) ===")
    print("  2022-2023 (CloudMapper overlap)...")
    download_era5_year(2022)
    download_era5_year(2023)
    print("  2017-2019 (TFT training)...")
    for yr in range(2017, 2020):
        download_era5_year(yr)

    print("\n=== ERA5 Extraction (NetCDF → CSV) ===")
    extract_era5_to_csv([2022, 2023], "era5_2022_2023.csv")
    extract_era5_to_csv([2017, 2018, 2019], "era5_2017_2019.csv")


# ==============================================================================
# CELL 5: ICON Cloud Cover (Open-Meteo archive)
# ==============================================================================
def fetch_openmeteo_icon_station(start_date: str, end_date: str, station: dict) -> pd.DataFrame:
    """Fetch ICON cloud cover from Open-Meteo historical archive for one station."""
    print(f"    {station['id']} ({station['lat']}, {station['lon']})...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    hourly_vars = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]

    # icon_global has gaps at this latitude for early 2022.
    # best_match backfills with ERA5 reanalysis cloud cover — full year.
    # We try icon_global first; if >10% NaN, fall back to best_match.
    chosen_model = None
    chosen_payload = None
    for model in ("icon_global", "best_match"):
        params = {
            "latitude":   station["lat"],
            "longitude":  station["lon"],
            "start_date": start_date,
            "end_date":   end_date,
            "hourly":     ",".join(hourly_vars),
            "models":     model,
            "timezone":   "UTC",
        }
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 200:
            payload = r.json()
            if "error" not in payload:
                # Check NaN coverage
                cc = payload["hourly"].get("cloud_cover", [])
                n_null = sum(1 for v in cc if v is None)
                pct_null = n_null / max(len(cc), 1) * 100
                print(f"    {model}: {len(cc)} rows, {n_null} NaN ({pct_null:.0f}%)")
                if pct_null < 10:
                    chosen_model = model
                    chosen_payload = payload
                    break
                else:
                    print(f"    {model} has too many gaps — trying next...")
                    chosen_payload = payload
                    continue
        print(f"    {model} failed (HTTP {r.status_code}), trying next...")
    else:
        # best_match was the last attempt; use it even with gaps
        chosen_model = "best_match"
        if chosen_payload is None:
            raise RuntimeError(f"Open-Meteo fetch failed for station {station['id']}")

    print(f"      {station['id']} using model: {chosen_model}")

    hourly = chosen_payload["hourly"]
    df = pd.DataFrame({"datetime": pd.to_datetime(hourly["time"], utc=True)})
    df["station_id"] = station["id"]
    for var in hourly_vars:
        raw = hourly.get(var)
        df[var] = np.array(raw, dtype=float) / 100.0 if raw else np.nan

    df = df.rename(columns={
        "cloud_cover_low":  "cloud_low",
        "cloud_cover_mid":  "cloud_mid",
        "cloud_cover_high": "cloud_high",
    })

    n_valid = df["cloud_cover"].notna().sum()
    print(f"      {station['id']} rows={len(df)} valid={n_valid}")
    return df


def fetch_openmeteo_icon_archive(
    start_date: str,
    end_date:   str,
    output_filename: str,
) -> None:
    """Fetch ICON cloud cover for all stations, then sample one station per timestamp."""
    print(f"  ICON archive {start_date} → {end_date} ({MAX_FETCH_WORKERS} parallel workers)...")

    station_frames = []
    with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as executor:
        futures = [
            executor.submit(fetch_openmeteo_icon_station, start_date, end_date, station)
            for station in STATIONS
        ]
        for future in as_completed(futures):
            station_frames.append(future.result())

    df = (
        pd.concat(station_frames, ignore_index=True)
        .sort_values(["datetime", "station_id"])
        .reset_index(drop=True)
    )
    df = sample_one_station_per_timestep(df)

    path = DOWNLOADS_DIR / output_filename
    df.to_csv(path, index=False)
    n_valid = df["cloud_cover"].notna().sum()
    print(f"    {output_filename}  shape={df.shape}  valid={n_valid}")
    print(f"      sampled stations={df['station_id'].value_counts().to_dict()}")


def fetch_all_icon() -> None:
    print("\n=== ICON Cloud Cover (Open-Meteo archive) ===")
    # 2022: icon_global has gaps for this lat, best_match backfills with ERA5
    fetch_openmeteo_icon_archive("2022-01-01", "2022-12-31", "icon_2022.csv")
    # 2023: full icon_global coverage — extra training data
    fetch_openmeteo_icon_archive("2023-01-01", "2023-12-31", "icon_2023.csv")
    # 2024: test set
    fetch_openmeteo_icon_archive("2024-01-01", "2024-12-31", "icon_2024.csv")


# ==============================================================================
# Verification — check that all expected files exist
# ==============================================================================
EXPECTED_FILES = [
    "era5_2022_2023.csv",
    "era5_2017_2019.csv",
    "icon_2022.csv",
    "icon_2023.csv",
    "icon_2024.csv",
]


def verify_downloads() -> None:
    print("\n=== Verification ===")
    all_ok = True
    for f in EXPECTED_FILES:
        p = DOWNLOADS_DIR / f
        if p.exists():
            df = pd.read_csv(p, nrows=2)
            print(f"  OK  {f:<35}  cols={list(df.columns)}")
        else:
            print(f"  MISSING  {f}")
            all_ok = False

    if all_ok:
        print("\n  All Phase 1 files present. Ready for:")
        print("    python train_mapper.py")
        print("    python generate_synthetic.py")
    else:
        print("\n  Some files missing — check errors above.")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fetch Phase 1 data for the CloudMapper")
    parser.add_argument("--skip-era5", action="store_true",
                        help="Skip ERA5 download (if already cached)")
    parser.add_argument("--only-era5", action="store_true",
                        help="Only download ERA5 (slowest step)")
    parser.add_argument("--extract-era5-only", action="store_true",
                        help="Only rebuild ERA5 CSVs from cached NetCDF files")
    parser.add_argument("--verify", action="store_true",
                        help="Only check which files exist")
    args = parser.parse_args()

    if args.verify:
        verify_downloads()
        return

    if args.only_era5:
        fetch_all_era5()
        verify_downloads()
        return

    if args.extract_era5_only:
        extract_era5_to_csv([2022, 2023], "era5_2022_2023.csv")
        extract_era5_to_csv([2017, 2018, 2019], "era5_2017_2019.csv")
        verify_downloads()
        return

    # ---- Full fetch ----
    # ERA5 is by far the slowest (CDS queue). Run it first.
    if not args.skip_era5:
        fetch_all_era5()
    else:
        print("\n  Skipping ERA5 (--skip-era5)")

    # ICON fetch is fast (direct REST API)
    fetch_all_icon()

    verify_downloads()


if __name__ == "__main__":
    print("=" * 55)
    print("  Phase 1 CloudMapper — Data Fetch")
    print("=" * 55)
    main()
