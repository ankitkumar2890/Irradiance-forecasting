"""
Build tiled FNO dataset for years 2010 and 2011 only.

Each sample is one tile:
- X: ERA5 channels on 9x9 tile
- Y: MODIS COT crop for same tile/time
"""

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.windows import from_bounds

# =========================
# CONFIG
# =========================

DATA_ROOT = Path(os.getenv("DATA_ROOT", "data_store")).expanduser()
ERA5_ROOT = DATA_ROOT / "ERA5_Input"
MODIS_DIR = DATA_ROOT / "MOD06L2_COT"

YEARS = [2010, 2011]

OUT_FILE = DATA_ROOT / "fno_dataset_2010_2011.npz"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

ERA5_VARIABLES = [
    "t2m", "d2m", "tcwv", "u10", "v10", "sp", "lcc", "mcc", "hcc"
]

BOX_SIZE_ERA5 = 9
BOXES_PER_SIDE = 4
GRID_SIZE_ERA5 = BOX_SIZE_ERA5 * BOXES_PER_SIDE  # 36

MAX_TIME_DIFF_MINUTES = 60
MIN_VALID_COT_FRACTION = 0.50


def parse_modis_timestamp(name):
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})", name)
    if not match:
        return None
    date_str, hour_str, minute_str = match.groups()
    return datetime.strptime(f"{date_str} {hour_str}:{minute_str}", "%Y-%m-%d %H:%M")


def nearest_era5_match(era5_times, target_time):
    diffs = np.abs(era5_times - np.datetime64(target_time))
    idx = int(np.argmin(diffs))
    diff_minutes = int(diffs[idx] / np.timedelta64(1, "m"))
    return idx, diff_minutes


def build_tiles(lat_crop, lon_crop):
    lat_step = float(np.abs(lat_crop[1] - lat_crop[0]))
    lon_step = float(np.abs(lon_crop[1] - lon_crop[0]))
    tiles = []
    for tr in range(BOXES_PER_SIDE):
        for tc in range(BOXES_PER_SIDE):
            r0 = tr * BOX_SIZE_ERA5
            r1 = (tr + 1) * BOX_SIZE_ERA5
            c0 = tc * BOX_SIZE_ERA5
            c1 = (tc + 1) * BOX_SIZE_ERA5

            lat_block = lat_crop[r0:r1]
            lon_block = lon_crop[c0:c1]
            north = float(np.max(lat_block))
            south = float(np.min(lat_block))
            west = float(np.min(lon_block))
            east = float(np.max(lon_block))
            bounds = (
                west - 0.5 * lon_step,
                south - 0.5 * lat_step,
                east + 0.5 * lon_step,
                north + 0.5 * lat_step,
            )
            tiles.append(
                {
                    "tile_id": tr * BOXES_PER_SIDE + tc,
                    "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                    "bounds": bounds,
                }
            )
    return tiles


def append_sample(store, x, y, tile_id, ts):
    store["X"].append(x)
    store["Y"].append(y)
    store["tile_id"].append(tile_id)
    store["timestamp"].append(ts)


def process_year(year, store, expected_shape):
    era5_dir = ERA5_ROOT / str(year)
    era5_files = sorted(era5_dir.glob("*/data_0.nc"))
    if not era5_files:
        raise FileNotFoundError(f"No ERA5 monthly files found under: {era5_dir}")

    print(f"\n=== Year {year} ===")
    print(f"Loading ERA5 files: {len(era5_files)}")
    ds = xr.open_mfdataset(era5_files, combine="by_coords")
    time_dim = "time" if "time" in ds.dims else "valid_time"

    lat_all = ds["latitude"].values
    lon_all = ds["longitude"].values
    if lat_all.size < GRID_SIZE_ERA5 or lon_all.size < GRID_SIZE_ERA5:
        raise RuntimeError(
            f"ERA5 grid too small for {GRID_SIZE_ERA5}x{GRID_SIZE_ERA5} crop "
            f"in year {year}. Got lat={lat_all.size}, lon={lon_all.size}."
        )

    lat_idx = slice(0, GRID_SIZE_ERA5)
    lon_idx = slice(0, GRID_SIZE_ERA5)
    ds_crop = ds.isel(latitude=lat_idx, longitude=lon_idx)
    lat_crop = lat_all[lat_idx]
    lon_crop = lon_all[lon_idx]
    era5_times = ds_crop[time_dim].values
    tiles = build_tiles(lat_crop, lon_crop)

    modis_files = sorted(MODIS_DIR.glob(f"{year}-*.tif"))
    print(f"MODIS files found for {year}: {len(modis_files)}")

    year_added = 0
    for tif_path in modis_files:
        modis_time = parse_modis_timestamp(tif_path.name)
        if modis_time is None:
            continue

        era5_idx, dt_min = nearest_era5_match(era5_times, modis_time)
        if dt_min > MAX_TIME_DIFF_MINUTES:
            continue

        with rasterio.open(tif_path) as src:
            for tile in tiles:
                window = from_bounds(*tile["bounds"], transform=src.transform)
                window = window.round_offsets().round_lengths()
                cot = src.read(1, window=window, boundless=True, fill_value=np.nan)
                if cot.size == 0:
                    continue

                valid_fraction = 1.0 - float(np.isnan(cot).mean())
                if valid_fraction < MIN_VALID_COT_FRACTION:
                    continue

                if expected_shape[0] is None:
                    expected_shape[0] = cot.shape
                    print(f"Locked MODIS tile shape: {expected_shape[0]}")
                elif cot.shape != expected_shape[0]:
                    continue

                cot = np.log1p(cot).astype(np.float32)

                era5_channels = []
                for var in ERA5_VARIABLES:
                    arr = ds_crop[var].isel({time_dim: era5_idx}).values
                    box = arr[tile["r0"]:tile["r1"], tile["c0"]:tile["c1"]]
                    era5_channels.append(box)
                era5_stack = np.stack(era5_channels, axis=0).astype(np.float32)

                append_sample(
                    store,
                    era5_stack,
                    cot[np.newaxis, :, :],
                    tile["tile_id"],
                    modis_time.strftime("%Y-%m-%d %H:%M"),
                )
                year_added += 1

    ds.close()
    print(f"Accepted samples from {year}: {year_added}")


def finalize_and_save(store, out_file):
    if not store["X"]:
        raise RuntimeError("No samples produced.")

    X = np.stack(store["X"])
    Y = np.stack(store["Y"])
    tile_id = np.array(store["tile_id"], dtype=np.int32)
    timestamp = np.array(store["timestamp"])

    np.savez_compressed(out_file, X=X, Y=Y, tile_id=tile_id, timestamp=timestamp)

    print(f"\nDataset saved: {out_file}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Number of instances made: {X.shape[0]}")


def main():
    print(f"Building dataset for years: {YEARS}")
    print(f"Data root: {DATA_ROOT}")

    store = {"X": [], "Y": [], "tile_id": [], "timestamp": []}
    expected_shape = [None]

    for year in YEARS:
        process_year(year, store, expected_shape)

    finalize_and_save(store, OUT_FILE)


if __name__ == "__main__":
    main()
