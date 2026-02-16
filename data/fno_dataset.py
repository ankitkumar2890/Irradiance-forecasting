"""
Build FNO Dataset (One Month)

Merges:
- ERA5 monthly NetCDF (9x9 grid)
- MODIS COT GeoTIFFs (~1 km grid)

Outputs:
X: [N, C, H, W]
Y: [N, 1, H, W]

Ready for FNO training.
"""

import os
import re
import numpy as np
import xarray as xr
import rasterio
from pathlib import Path
from datetime import datetime

# =========================
# CONFIG
# =========================

YEAR = 2005
MONTH = 1

ERA5_PATH = Path(f"data_store/ERA5_Input/2005/2005_01/data_0.nc")
MODIS_DIR = Path("data_store/MOD06L2_COT")

OUTPUT_FILE = Path("data_store/fno_dataset_month1.npz")

ERA5_VARIABLES = [
    "t2m",
    "d2m",
    "tcwv",
    "u10",
    "v10",
    "sp",
    "lcc",
    "mcc",
    "hcc",
]

# =========================
# LOAD ERA5
# =========================

print("Loading ERA5...")
ds = xr.open_dataset(ERA5_PATH)

time_dim = "time" if "time" in ds.dims else "valid_time"

era5_times = ds[time_dim].values

# =========================
# HELPER: match ERA5 time
# =========================

def get_nearest_era5_index(target_time):
    diffs = np.abs(era5_times - np.datetime64(target_time))
    return int(np.argmin(diffs))

# =========================
# PROCESS MODIS FILES
# =========================

X_list = []
Y_list = []

modis_files = sorted([f for f in MODIS_DIR.glob("*.tif")])

print(f"Found {len(modis_files)} MODIS files")

for tif_path in modis_files:

    # Extract timestamp from filename
    # Format: 2005-01-01_10-30_COT.tif
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})", tif_path.name)
    if not match:
        continue

    date_str, hour_str, minute_str = match.groups()
    modis_time = datetime.strptime(
        f"{date_str} {hour_str}:{minute_str}",
        "%Y-%m-%d %H:%M"
    )

    # Only keep files in selected month
    if modis_time.year != YEAR or modis_time.month != MONTH:
        continue

    # Load MODIS COT
    with rasterio.open(tif_path) as src:
        cot = src.read(1)
        transform = src.transform
        height = src.height
        width = src.width

    if np.all(np.isnan(cot)):
        continue

    # Log-transform COT
    cot = np.log1p(cot)

    # =========================
    # Match ERA5 time
    # =========================

    era5_idx = get_nearest_era5_index(modis_time)

    # =========================
    # Interpolate ERA5 → MODIS grid
    # =========================

    # Build lat/lon grid for MODIS
    rows = np.arange(height)
    cols = np.arange(width)

    xs, ys = np.meshgrid(cols, rows)
    lon, lat = rasterio.transform.xy(transform, ys, xs)
    lon = np.array(lon)
    lat = np.array(lat)

    # Interpolate ERA5 variables
    era5_interp_channels = []

    for var in ERA5_VARIABLES:
        era5_slice = ds[var].isel({time_dim: era5_idx})

        interp = era5_slice.interp(
            latitude=(("y", "x"), lat),
            longitude=(("y", "x"), lon),
            method="linear"
        )

        era5_interp_channels.append(interp.values)

    era5_stack = np.stack(era5_interp_channels, axis=0)

    # Remove samples with too many NaNs
    # Allow partial swath coverage
    if np.isnan(cot).mean() > 0.95:
        print("  Skipping (almost empty swath)")
        continue

    # ERA5 interpolation should never be mostly NaN
    if np.isnan(era5_stack).mean() > 0.1:
        print("  Warning: ERA5 interpolation contains NaNs")


    X_list.append(era5_stack.astype(np.float32))
    Y_list.append(cot[np.newaxis, :, :].astype(np.float32))

    print("✓ Paired:", tif_path.name)

# =========================
# FINALIZE DATASET
# =========================

X = np.stack(X_list)
Y = np.stack(Y_list)

print("\nFinal tensor shapes:")
print("X:", X.shape)
print("Y:", Y.shape)

np.savez_compressed(OUTPUT_FILE, X=X, Y=Y)

print("\nSaved dataset to:", OUTPUT_FILE)
