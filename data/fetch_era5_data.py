"""
ERA5 Downloader (Single + Pressure Levels)
Native resolution (0.25 deg)
South India subset
No interpolation, no static variables.

Downloads:
- Single levels: tclw, tciw, tcwv
- Pressure levels:
  - RH: 850, 700 hPa
  - Temperature: 850, 700 hPa
  - U/V wind: 850 hPa
  - Vertical velocity: 700 hPa

Output:
- Monthly merged NetCDF: data_0.nc
"""

import calendar
import time
import argparse
from pathlib import Path

import cdsapi
import xarray as xr

# =========================
# CDS API AUTH (IN FILE)
# =========================
CDSAPI_URL = "https://cds.climate.copernicus.eu/api"
CDSAPI_KEY = "12786cbc-c2fb-4c4e-a380-40e490520f5b"  # your token

# =========================
# CONFIG
# =========================
DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2019

BASE_OUTPUT_DIR = Path("/Users/IRFAN/Library/CloudStorage/GoogleDrive-irfan.a@atriauniversity.edu.in/My Drive/Irradiance-forecasting/new_input_era5")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# South India domain (9 deg x 9 deg)
# ERA5 area format: [North, West, South, East]
SOUTH_INDIA_NORTH = 17.0
SOUTH_INDIA_WEST = 72.5
SOUTH_INDIA_SOUTH = 8.0
SOUTH_INDIA_EAST = 81.5

AREA = [
    SOUTH_INDIA_NORTH,
    SOUTH_INDIA_WEST,
    SOUTH_INDIA_SOUTH,
    SOUTH_INDIA_EAST,
]

# ERA5 native grid (~0.25 deg)
GRID = [0.25, 0.25]

TARGET_HOURS = [f"{h:02d}:00" for h in range(24)]

# =========================
# VARIABLES
# =========================
SINGLE_LEVEL_VARIABLES = [
    "total_column_cloud_liquid_water",
    "total_column_cloud_ice_water",
    "total_column_water_vapour",
]

PRESSURE_LEVEL_DATASET = "reanalysis-era5-pressure-levels"
SINGLE_LEVEL_DATASET = "reanalysis-era5-single-levels"

PRESSURE_LEVEL_REQUESTS = [
    {
        "name": "rh_temp_700_850",
        "variables": [
            "relative_humidity",
            "temperature",
        ],
        "pressure_levels": ["700", "850"],
    },
    {
        "name": "uv_850",
        "variables": [
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        "pressure_levels": ["850"],
    },
    {
        "name": "w_700",
        "variables": ["vertical_velocity"],
        "pressure_levels": ["700"],
    },
]

# =========================
# RETRY LOGIC
# =========================
def download_with_retry(client, dataset, request, target, max_retries=5):
    for attempt in range(max_retries):
        try:
            client.retrieve(dataset, request, target)
            return True
        except Exception as e:
            wait = 30 * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Failed permanently: {target}")
                return False


def build_common_request(year, month, days):
    return {
        "product_type": "reanalysis",
        "year": str(year),
        "month": f"{month:02d}",
        "day": days,
        "time": TARGET_HOURS,
        "area": AREA,
        "grid": GRID,
        "format": "netcdf",
    }


def merge_files(single_level_file, pressure_level_files, out_file):
    datasets = []
    try:
        datasets.append(xr.open_dataset(single_level_file))
        for path in pressure_level_files:
            datasets.append(xr.open_dataset(path))

        merged = xr.merge(datasets, compat="override")
        merged.to_netcdf(out_file)
        merged.close()
    finally:
        for ds in datasets:
            ds.close()


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 in monthly files. Use --start-year/--end-year for batch runs."
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    args = parser.parse_args()

    start_year = int(args.start_year)
    end_year = int(args.end_year)
    if end_year < start_year:
        raise ValueError(f"Invalid year range: start={start_year}, end={end_year}")

    url = CDSAPI_URL.strip()
    key = CDSAPI_KEY.strip()

    if not url or not key:
        raise ValueError("CDSAPI_URL and CDSAPI_KEY must be non-empty.")

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Token-only key is valid with new CDS credentials
    client = cdsapi.Client(url=url, key=key)

    print(f"Downloading years: {start_year}..{end_year}")
    for year in range(start_year, end_year + 1):
        year_dir = BASE_OUTPUT_DIR / str(year)
        year_dir.mkdir(exist_ok=True)

        for month in range(1, 13):
            month_dir = year_dir / f"{year}_{month:02d}"
            month_dir.mkdir(exist_ok=True)

            out_file = month_dir / "data_0.nc"
            if out_file.exists():
                print(f"Skipping {year}-{month:02d}, already exists.")
                continue

            ndays = calendar.monthrange(year, month)[1]
            days = [f"{d:02d}" for d in range(1, ndays + 1)]

            common_request = build_common_request(year, month, days)
            single_level_file = month_dir / "single_levels_tmp.nc"
            pressure_level_tmp_files = []

            single_level_request = dict(common_request)
            single_level_request["variable"] = SINGLE_LEVEL_VARIABLES

            print(f"Downloading {year}-{month:02d} single-level fields...")
            ok_single = download_with_retry(
                client,
                SINGLE_LEVEL_DATASET,
                single_level_request,
                str(single_level_file),
            )
            if not ok_single:
                if single_level_file.exists():
                    single_level_file.unlink()
                continue

            all_pressure_ok = True
            for request_spec in PRESSURE_LEVEL_REQUESTS:
                pressure_file = month_dir / f"{request_spec['name']}_tmp.nc"
                pressure_request = dict(common_request)
                pressure_request["variable"] = request_spec["variables"]
                pressure_request["pressure_level"] = request_spec["pressure_levels"]
                pressure_level_tmp_files.append(pressure_file)

                print(
                    f"Downloading {year}-{month:02d} pressure fields: "
                    f"{request_spec['name']}..."
                )
                ok_pressure = download_with_retry(
                    client,
                    PRESSURE_LEVEL_DATASET,
                    pressure_request,
                    str(pressure_file),
                )
                if not ok_pressure:
                    all_pressure_ok = False
                    break

            if not all_pressure_ok:
                print(f"Skipping merge for {year}-{month:02d} due to failed download(s).")
                if single_level_file.exists():
                    single_level_file.unlink()
                for tmp_file in pressure_level_tmp_files:
                    if tmp_file.exists():
                        tmp_file.unlink()
                continue

            print(f"Merging monthly files for {year}-{month:02d}...")
            merge_files(single_level_file, pressure_level_tmp_files, out_file)

            if single_level_file.exists():
                single_level_file.unlink()
            for tmp_file in pressure_level_tmp_files:
                if tmp_file.exists():
                    tmp_file.unlink()

    print("\nERA5 0.25 deg dataset ready.")
    print("Location:", BASE_OUTPUT_DIR)

if __name__ == "__main__":
    main()
