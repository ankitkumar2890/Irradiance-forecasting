"""
ERA5 (Single Levels) Downloader
Native resolution (~0.25 deg ~= 25 km)
South India subset
No interpolation, no static variables

Dataset: reanalysis-era5-single-levels
"""

import cdsapi
from pathlib import Path
import calendar
import time

# =========================
# CDS API AUTH (IN FILE)
# =========================
CDSAPI_URL = "https://cds.climate.copernicus.eu/api"
CDSAPI_KEY = "12786cbc-c2fb-4c4e-a380-40e490520f5b"  # your token

# =========================
# CONFIG
# =========================
START_YEAR = 2019
END_YEAR = 2019

BASE_OUTPUT_DIR = Path("/content/drive/MyDrive/Irradiance-forecasting/ERA5_Input")
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

# ERA5 native grid (~25 km)
GRID = [0.25, 0.25]

TARGET_HOURS = [f"{h:02d}:00" for h in range(24)]

# =========================
# VARIABLES (Cloud Relevant)
# =========================
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_column_water_vapour",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
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

# =========================
# MAIN
# =========================
def main():
    url = CDSAPI_URL.strip()
    key = CDSAPI_KEY.strip()

    if not url or not key:
        raise ValueError("CDSAPI_URL and CDSAPI_KEY must be non-empty.")

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Token-only key is valid with new CDS credentials
    client = cdsapi.Client(url=url, key=key)

    for year in range(START_YEAR, END_YEAR + 1):
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

            request = {
                "product_type": "reanalysis",
                "variable": VARIABLES,
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": TARGET_HOURS,
                "area": AREA,
                "grid": GRID,
                "format": "netcdf",
            }

            print(f"Downloading {year}-{month:02d} (native ~25 km, South India)...")
            download_with_retry(
                client,
                "reanalysis-era5-single-levels",
                request,
                str(out_file),
            )

    print("\nERA5 native (~25 km) dataset ready.")
    print("Location:", BASE_OUTPUT_DIR)

if __name__ == "__main__":
    main()
