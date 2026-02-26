"""
Build FNO dataset shards (2-year chunks) to reduce runtime risk and memory pressure.

Output structure:
- /content/drive/MyDrive/Irradiance-forecasting/fno_dataset_10/shards/train_2010_2011.npz
- /content/drive/MyDrive/Irradiance-forecasting/fno_dataset_10/shards/train_2012_2013.npz
- /content/drive/MyDrive/Irradiance-forecasting/fno_dataset_10/shards/train_2014_2015.npz
- /content/drive/MyDrive/Irradiance-forecasting/fno_dataset_10/shards/train_2016_2017.npz
- /content/drive/MyDrive/Irradiance-forecasting/fno_dataset_10/shards/val_2018_2019.npz
"""

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.windows import Window, bounds as window_bounds

# =========================
# CONFIG
# =========================

ERA5_ROOT = Path("/content/drive/MyDrive/Irradiance-forecasting/ERA5_Input")
MODIS_DIR = Path("/content/drive/MyDrive/Irradiance-forecasting/MOD06L2_COT")
OUT_DIR = Path("/content/drive/MyDrive/Irradiance-forecasting/fno_dataset_10")
SHARD_DIR = OUT_DIR / "shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_GROUPS = [
    [2010, 2011],
    [2012, 2013],
    [2014, 2015],
    [2016, 2017],
]
VAL_GROUPS = [[2018, 2019]]

ERA5_VARIABLES = ["lcc"]

MAX_TIME_DIFF_MINUTES = 30
MIN_VALID_COT_FRACTION = 0.50

# Tile and resolution design.
MODIS_TILE_SIZE = 108       # 108x108 pixels, each ~1 km
MODIS_PIXEL_KM = 1
ERA5_PIXEL_KM = 9           # target ERA5 spacing for each low-res pixel
ERA5_TILE_SIZE = MODIS_TILE_SIZE * MODIS_PIXEL_KM // ERA5_PIXEL_KM  # 12
UPSAMPLE_FACTOR = ERA5_PIXEL_KM // MODIS_PIXEL_KM                    # 9

# Shared domain check.
DOMAIN_NORTH = 17.0
DOMAIN_SOUTH = 8.0
DOMAIN_WEST = 72.5
DOMAIN_EAST = 81.5
DOMAIN_BOUNDS = (DOMAIN_WEST, DOMAIN_SOUTH, DOMAIN_EAST, DOMAIN_NORTH)
DOMAIN_TOL_DEG = 0.03

if MODIS_TILE_SIZE % ERA5_PIXEL_KM != 0:
    raise ValueError("MODIS_TILE_SIZE must be divisible by ERA5_PIXEL_KM.")


def build_modis_tile_windows(height, width):
    tiles_h = height // MODIS_TILE_SIZE
    tiles_w = width // MODIS_TILE_SIZE
    windows = []
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            windows.append(
                {
                    "tile_id": tr * tiles_w + tc,
                    "window": Window(
                        col_off=tc * MODIS_TILE_SIZE,
                        row_off=tr * MODIS_TILE_SIZE,
                        width=MODIS_TILE_SIZE,
                        height=MODIS_TILE_SIZE,
                    ),
                }
            )
    return windows, tiles_h, tiles_w


def tile_centers_from_bounds(bounds, out_h, out_w):
    west, south, east, north = bounds
    lon_step = (east - west) / out_w
    lat_step = (north - south) / out_h
    lats = north - (np.arange(out_h) + 0.5) * lat_step
    lons = west + (np.arange(out_w) + 0.5) * lon_step
    return lats, lons


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


def check_era5_covers_domain(ds_interp):
    lat = ds_interp["latitude"].values
    lon = ds_interp["longitude"].values
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))

    if lat_min > DOMAIN_SOUTH or lat_max < DOMAIN_NORTH:
        raise RuntimeError(
            f"ERA5 latitude coverage [{lat_min}, {lat_max}] does not cover "
            f"domain [{DOMAIN_SOUTH}, {DOMAIN_NORTH}]."
        )
    if lon_min > DOMAIN_WEST or lon_max < DOMAIN_EAST:
        raise RuntimeError(
            f"ERA5 longitude coverage [{lon_min}, {lon_max}] does not cover "
            f"domain [{DOMAIN_WEST}, {DOMAIN_EAST}]."
        )


def check_modis_matches_domain(src):
    b = src.bounds
    src_bounds = (float(b.left), float(b.bottom), float(b.right), float(b.top))
    diffs = [abs(a - e) for a, e in zip(src_bounds, DOMAIN_BOUNDS)]
    if any(d > DOMAIN_TOL_DEG for d in diffs):
        raise RuntimeError(
            f"MODIS bounds {src_bounds} differ from expected shared domain "
            f"{DOMAIN_BOUNDS} by {diffs} degrees."
        )


def process_year(year, store):
    era5_dir = ERA5_ROOT / str(year)
    era5_files = sorted(era5_dir.glob("*/data_0.nc"))
    if not era5_files:
        raise FileNotFoundError(f"No ERA5 monthly files found under: {era5_dir}")

    print(f"\n=== Year {year} ===")
    print(f"Loading ERA5 files: {len(era5_files)}")
    ds = xr.open_mfdataset(era5_files, combine="by_coords", chunks={"time": 24})
    time_dim = "time" if "time" in ds.dims else "valid_time"
    ds_interp = ds.sortby("latitude", ascending=False).sortby("longitude", ascending=True)
    era5_times = ds_interp[time_dim].values
    check_era5_covers_domain(ds_interp)

    modis_files = sorted(MODIS_DIR.glob(f"{year}-*.tif"))
    print(f"MODIS files found for {year}: {len(modis_files)}")

    printed_tiling_stats = False
    year_added = 0
    for tif_path in modis_files:
        modis_time = parse_modis_timestamp(tif_path.name)
        if modis_time is None:
            continue

        era5_idx, dt_min = nearest_era5_match(era5_times, modis_time)
        if dt_min > MAX_TIME_DIFF_MINUTES:
            continue

        with rasterio.open(tif_path) as src:
            check_modis_matches_domain(src)
            tiles, tiles_h, tiles_w = build_modis_tile_windows(src.height, src.width)

            if not printed_tiling_stats:
                total_px = src.height * src.width
                used_px = tiles_h * tiles_w * MODIS_TILE_SIZE * MODIS_TILE_SIZE
                wasted_px = total_px - used_px
                wasted_frac = wasted_px / max(total_px, 1)
                print(
                    f"Tiling stats ({year}): raster={src.height}x{src.width}, "
                    f"tiles={tiles_h}x{tiles_w}={tiles_h*tiles_w}, "
                    f"used={used_px} px, wasted={wasted_px} px ({100*wasted_frac:.2f}%)."
                )
                printed_tiling_stats = True

            for tile in tiles:
                cot = src.read(1, window=tile["window"], boundless=False)
                if cot.shape != (MODIS_TILE_SIZE, MODIS_TILE_SIZE):
                    continue

                valid_fraction = 1.0 - float(np.isnan(cot).mean())
                if valid_fraction < MIN_VALID_COT_FRACTION:
                    continue

                cot = np.log1p(cot).astype(np.float32)
                bounds = window_bounds(tile["window"], src.transform)
                lats, lons = tile_centers_from_bounds(bounds, ERA5_TILE_SIZE, ERA5_TILE_SIZE)

                era5_channels = []
                for var in ERA5_VARIABLES:
                    field = ds_interp[var].isel({time_dim: era5_idx})
                    coarse_linear = field.interp(
                        latitude=xr.DataArray(lats, dims="latitude"),
                        longitude=xr.DataArray(lons, dims="longitude"),
                        method="linear",
                    )
                    coarse_nearest = field.interp(
                        latitude=xr.DataArray(lats, dims="latitude"),
                        longitude=xr.DataArray(lons, dims="longitude"),
                        method="nearest",
                    )
                    coarse = coarse_linear.where(np.isfinite(coarse_linear), coarse_nearest)
                    box = coarse.values.astype(np.float32)
                    if box.shape != (ERA5_TILE_SIZE, ERA5_TILE_SIZE):
                        continue
                    box_up = np.repeat(np.repeat(box, UPSAMPLE_FACTOR, axis=0), UPSAMPLE_FACTOR, axis=1)
                    era5_channels.append(box_up)

                if len(era5_channels) != len(ERA5_VARIABLES):
                    continue

                era5_stack = np.stack(era5_channels, axis=0).astype(np.float32)
                store["X"].append(era5_stack)
                store["Y"].append(cot[np.newaxis, :, :])
                store["tile_id"].append(tile["tile_id"])
                store["timestamp"].append(modis_time.strftime("%Y-%m-%d %H:%M"))
                store["year"].append(year)
                year_added += 1

    ds.close()
    print(f"Accepted samples from {year}: {year_added}")


def finalize_and_save(store, out_file):
    if not store["X"]:
        raise RuntimeError(f"No samples produced for {out_file}.")

    X = np.stack(store["X"])
    Y = np.stack(store["Y"])
    tile_id = np.array(store["tile_id"], dtype=np.int32)
    timestamp = np.array(store["timestamp"])
    year = np.array(store["year"], dtype=np.int16)

    np.savez_compressed(out_file, X=X, Y=Y, tile_id=tile_id, timestamp=timestamp, year=year)
    print(f"\nShard saved: {out_file}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"samples: {X.shape[0]}")


def build_group(split_name, years):
    store = {"X": [], "Y": [], "tile_id": [], "timestamp": [], "year": []}
    for year in years:
        process_year(year, store)

    out_file = SHARD_DIR / f"{split_name}_{years[0]}_{years[-1]}.npz"
    finalize_and_save(store, out_file)


def main():
    print(f"ERA5 root: {ERA5_ROOT}")
    print(f"MODIS dir: {MODIS_DIR}")
    print(f"Shard dir: {SHARD_DIR}")
    print(
        f"Tile setup: MODIS {MODIS_TILE_SIZE}x{MODIS_TILE_SIZE} at {MODIS_PIXEL_KM} km, "
        f"ERA5 coarse {ERA5_TILE_SIZE}x{ERA5_TILE_SIZE} at {ERA5_PIXEL_KM} km "
        f"upsampled by {UPSAMPLE_FACTOR}x."
    )

    for years in TRAIN_GROUPS:
        build_group("train", years)

    for years in VAL_GROUPS:
        build_group("val", years)

    print("\nDone. Next step: run merge script to create final train/validate files.")


if __name__ == "__main__":
    main()

