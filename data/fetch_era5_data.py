"""
ERA5 (Single Levels) Downloader
Native resolution (~0.25° ≈ 25 km)
Small box around plant site
No interpolation, no static variables

Dataset: reanalysis-era5-single-levels
"""

import cdsapi
from pathlib import Path
import calendar
import time

# =========================
# CONFIG
# =========================

START_YEAR = 2005
END_YEAR   = 2005

BASE_OUTPUT_DIR = Path("data_store/ERA5_Input")

# Plant site
SITE_LAT = 9.14
SITE_LON = 77.92

# Box size around site for context
# 1.0° ≈ ~110 km → gives ~9x9 ERA5 pixels at 0.25°
BUFFER_DEG = 1.0

AREA = [
    SITE_LAT + BUFFER_DEG,  # North
    SITE_LON - BUFFER_DEG,  # West
    SITE_LAT - BUFFER_DEG,  # South
    SITE_LON + BUFFER_DEG,  # East
]

# ERA5 native grid (≈25 km). You can also remove this line; default is the same.
GRID = [0.25, 0.25]

TARGET_HOURS = [f"{h:02d}:00" for h in range(24)]

# =========================
# VARIABLES (Cloud Relevant)
# =========================

VARIABLES = [
    # Temperature
    "2m_temperature",
    "2m_dewpoint_temperature",

    # Moisture
    "total_column_water_vapour",

    # Wind
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",

    # Pressure
    "surface_pressure",

    # Cloud predictors
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
            print(f"⚠ Attempt {attempt+1} failed: {e}")

            if attempt < max_retries - 1:
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"✗ Failed permanently: {target}")
                return False

# =========================
# MAIN
# =========================

def main():

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

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
                "area": AREA,        # <-- spatial subset around site
                "grid": GRID,        # <-- native 0.25° (~25 km), no interpolation
                "format": "netcdf",
            }

            print(f"Downloading {year}-{month:02d} (native ~25 km, site box)...")

            download_with_retry(
                client,
                "reanalysis-era5-single-levels",
                request,
                str(out_file),
            )

    print("\n✅ ERA5 native (~25 km) dataset ready.")
    print("Location:", BASE_OUTPUT_DIR)


if __name__ == "__main__":
    main()
