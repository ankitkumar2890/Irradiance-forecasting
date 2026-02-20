"""
Build FNO Dataset (Full Year)

Merges:
- ERA5 monthly NetCDF files (9x9 grid) — all 12 months
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

ERA5_DIR = Path(f"data_store/ERA5_Input/{YEAR}")
MODIS_DIR = Path("data_store/MOD06L2_COT")

OUTPUT_FILE = Path(f"data_store/fno_dataset_{YEAR}1.npz")

# Maximum allowed MODIS↔ERA5 pairing gap (minutes).
# Samples outside this window are discarded to avoid wrong temporal pairing.
MAX_TIME_DIFF_MINUTES = 60

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
# LOAD ERA5 (all months)
# =========================

print("Loading ERA5 (all months)...")

era5_files = sorted(ERA5_DIR.glob("*/data_0.nc"))
print(f"Found {len(era5_files)} ERA5 monthly files")

if not era5_files:
    raise FileNotFoundError(f"No ERA5 monthly files found under: {ERA5_DIR}")

ds = xr.open_mfdataset(era5_files, combine="by_coords")

time_dim = "time" if "time" in ds.dims else "valid_time"

era5_times = ds[time_dim].values

print(f"ERA5 time range: {era5_times[0]} to {era5_times[-1]}")

# =========================
# HELPER: match ERA5 time
# =========================

def get_nearest_era5_match(target_time):
    diffs = np.abs(era5_times - np.datetime64(target_time))
    idx = int(np.argmin(diffs))
    diff_minutes = int(diffs[idx] / np.timedelta64(1, "m"))
    return idx, diff_minutes

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

    # Only keep files in selected year
    if modis_time.year != YEAR:
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

    era5_idx, time_diff_minutes = get_nearest_era5_match(modis_time)
    if time_diff_minutes > MAX_TIME_DIFF_MINUTES:
        print(
            f"  Skipping {tif_path.name}: nearest ERA5 is "
            f"{time_diff_minutes} min away (> {MAX_TIME_DIFF_MINUTES})"
        )
        continue

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

if not X_list or not Y_list:
    raise RuntimeError(
        "No valid MODIS↔ERA5 pairs were produced. "
        "Check time alignment, date range, and filtering thresholds."
    )

X = np.stack(X_list)
Y = np.stack(Y_list)

print(f"\nFinal tensor shapes:")
print("X:", X.shape)
print("Y:", Y.shape)
print(f"Total samples: {X.shape[0]}")

np.savez_compressed(OUTPUT_FILE, X=X, Y=Y)

print("\nSaved dataset to:", OUTPUT_FILE)
