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

try:
    from data.prithvi_utils import MIN_PRITHVI_YEAR, PrithviDynamicProvider
except ImportError:
    from prithvi_utils import MIN_PRITHVI_YEAR, PrithviDynamicProvider

try:
    from data.hls_landsat_utils import MIN_HLS_YEAR, LandsatHLSDynamicProvider
except ImportError:
    from hls_landsat_utils import MIN_HLS_YEAR, LandsatHLSDynamicProvider

# =========================
# CONFIG
# =========================

ERA5_ROOT = Path("/content/drive/MyDrive/Irradiance-forecasting/new_input_era5")
MODIS_DIR = Path("/content/drive/MyDrive/Irradiance-forecasting/MOD06L2_COT")
OUT_DIR = Path("/content/drive/MyDrive/Irradiance-forecasting/fno_dataset_10")
SHARD_DIR = OUT_DIR / "shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_GROUPS = [
    [2010, 2011],
    [2012, 2013],
    [2014, 2015],
]
VAL_GROUPS = [[2016, 2017]]
TEST_GROUPS = [[2018, 2019]]

# ERA5 feature spec aligned with data/fetch_era5_data.py output.
# `name` is the NetCDF variable key and `level` is optional (for pressure-level variables).
ERA5_FEATURE_SPECS = [
    {"name": "tclw"},              # total_column_cloud_liquid_water
    {"name": "tciw"},              # total_column_cloud_ice_water
    {"name": "tcwv"},              # total_column_water_vapour
    {"name": "r", "level": 700},   # relative_humidity @ 700 hPa
    {"name": "r", "level": 850},   # relative_humidity @ 850 hPa
    {"name": "t", "level": 700},   # temperature @ 700 hPa
    {"name": "t", "level": 850},   # temperature @ 850 hPa
    {"name": "u", "level": 850},   # u wind @ 850 hPa
    {"name": "v", "level": 850},   # v wind @ 850 hPa
    {"name": "w", "level": 700},   # vertical_velocity @ 700 hPa
]

ERA5_MAX_TIME_DIFF_MINUTES = 30
MIN_VALID_COT_FRACTION = 0.50

# Prithvi dynamic features (PCA-projected to 8 channels)
USE_PRITHVI = False
PRITHVI_SOURCE_PATH = Path("/content/drive/MyDrive/Irradiance-forecasting/PRITHVI/tiles")
# Supports either a single NetCDF/Zarr source or a directory of per-ERA5-tile NetCDF files.
PRITHVI_PCA_MODEL_PATH = Path("/content/drive/MyDrive/Irradiance-forecasting/PRITHVI/prithvi_pca_model.npz")
PRITHVI_VARIABLE_NAME = None        # set string if source has multiple data variables
# Prithvi embeddings capture land features (vegetation, urban texture, soil variation)
# which change slowly (seasonally). Since optical satellite imagery (HLS) is only 
# available every few days and often obscured by clouds, it is standard practice to
# use a much larger matching window (e.g., 15 days) to find the nearest clear-sky image.
PRITHVI_MAX_TIME_DIFF_MINUTES = 15 * 24 * 60  # Allow a 15-day gap (21,600 minutes)
PRITHVI_TIME_DIM = None             # override if needed
PRITHVI_CHANNEL_DIM = None          # override if needed
PRITHVI_LAT_DIM = None              # override if needed
PRITHVI_LON_DIM = None              # override if needed
PRITHVI_TILE_CACHE_SIZE = 128

# Landsat-8 HLS dynamic features replacing Prithvi when enabled.
# HLS provides 7 channels: NDMI, NDVI, NDWI, MNDWI, SWIR1, brightness_temp, elevation
# Plus 1 binary mask channel = 8 additional channels total
USE_LANDSAT_HLS = True
LANDSAT_HLS_SOURCE_PATH = Path("/content/drive/MyDrive/Irradiance-forecasting/hls_landsat_tiles")
LANDSAT_HLS_VARIABLE_NAME = None
LANDSAT_HLS_LAG_DAYS = 8
LANDSAT_HLS_MAX_STALENESS_DAYS = 45
LANDSAT_HLS_TIME_DIM = None
LANDSAT_HLS_CHANNEL_DIM = None
LANDSAT_HLS_LAT_DIM = None
LANDSAT_HLS_LON_DIM = None
LANDSAT_HLS_TILE_CACHE_SIZE = 32
LANDSAT_HLS_COORD_TOL_DEG = 1e-6

# Tile and resolution design.
# For ERA5 single-levels at native 0.25 deg, effective spacing is ~27 km
# over the South India domain in this project.
MODIS_TILE_SIZE = 108       # 108x108 pixels, each ~1 km
MODIS_PIXEL_KM = 1
ERA5_PIXEL_KM = 27          # approximate native ERA5 spacing at project latitude
ERA5_TILE_SIZE = MODIS_TILE_SIZE * MODIS_PIXEL_KM // ERA5_PIXEL_KM  # 4
UPSAMPLE_FACTOR = ERA5_PIXEL_KM // MODIS_PIXEL_KM                    # 27

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


def _pick_level_dim(field: xr.DataArray):
    for dim in ("pressure_level", "level", "isobaricInhPa"):
        if dim in field.dims:
            return dim
    return None


def validate_era5_features(ds_interp):
    missing = []
    for spec in ERA5_FEATURE_SPECS:
        var_name = spec["name"]
        if var_name not in ds_interp.data_vars:
            missing.append(f"{var_name} (variable not found)")
            continue
        if "level" in spec:
            level_value = int(spec["level"])
            field = ds_interp[var_name]
            level_dim = _pick_level_dim(field)
            if level_dim is None:
                missing.append(f"{var_name}@{level_value} (no pressure level dimension)")
                continue
            levels = np.asarray(field[level_dim].values)
            if levels.size == 0 or np.min(np.abs(levels - level_value)) > 1e-6:
                missing.append(f"{var_name}@{level_value} (level missing; found {levels.tolist()})")
    if missing:
        joined = "; ".join(missing)
        raise KeyError(f"ERA5 feature spec mismatch: {joined}")


def select_era5_feature(ds_interp, time_dim, era5_idx, spec):
    var_name = spec["name"]
    field = ds_interp[var_name].isel({time_dim: era5_idx})
    if "level" in spec:
        level_value = int(spec["level"])
        level_dim = _pick_level_dim(field)
        if level_dim is None:
            raise KeyError(f"{var_name}@{level_value}: pressure level dimension not found.")
        field = field.sel({level_dim: level_value}, method="nearest")
    return field


def process_year(year, store, dynamic_provider=None, dynamic_feature_channels=None, expected_x_channels=None):
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
    validate_era5_features(ds_interp)

    modis_files = sorted(MODIS_DIR.glob(f"{year}-*.tif"))
    print(f"MODIS files found for {year}: {len(modis_files)}")

    printed_tiling_stats = False
    year_added = 0
    rejected_era5_time = 0
    rejected_dynamic_time = 0
    accepted_dynamic_time_diffs = []
    _seen_dynamic_errors = set()  # log first occurrence of each error type
    for tif_path in modis_files:
        modis_time = parse_modis_timestamp(tif_path.name)
        if modis_time is None:
            continue

        era5_idx, dt_min = nearest_era5_match(era5_times, modis_time)
        if dt_min > ERA5_MAX_TIME_DIFF_MINUTES:
            rejected_era5_time += 1
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
                target_lats, target_lons = tile_centers_from_bounds(
                    bounds, MODIS_TILE_SIZE, MODIS_TILE_SIZE
                )

                era5_channels = []
                for spec in ERA5_FEATURE_SPECS:
                    field = select_era5_feature(ds_interp, time_dim, era5_idx, spec)
                    coarse_linear = field.interp(
                        latitude=xr.DataArray(target_lats, dims="latitude"),
                        longitude=xr.DataArray(target_lons, dims="longitude"),
                        method="linear",
                    )
                    coarse_nearest = field.interp(
                        latitude=xr.DataArray(target_lats, dims="latitude"),
                        longitude=xr.DataArray(target_lons, dims="longitude"),
                        method="nearest",
                    )
                    coarse = coarse_linear.where(np.isfinite(coarse_linear), coarse_nearest)
                    box = coarse.values.astype(np.float32)
                    if box.shape != (MODIS_TILE_SIZE, MODIS_TILE_SIZE):
                        continue
                    era5_channels.append(box)

                if len(era5_channels) != len(ERA5_FEATURE_SPECS):
                    continue

                era5_stack = np.stack(era5_channels, axis=0).astype(np.float32)
                x_stack = era5_stack

                if dynamic_provider is not None:
                    valid_dynamic = 0
                    dynamic_stack = np.zeros(
                        (int(dynamic_feature_channels), MODIS_TILE_SIZE, MODIS_TILE_SIZE),
                        dtype=np.float32,
                    )

                    min_dynamic_year = MIN_PRITHVI_YEAR if USE_PRITHVI else MIN_HLS_YEAR
                    if modis_time.year >= min_dynamic_year:
                        try:
                            sample, dynamic_dt_min = dynamic_provider.sample_tile_for_time(
                                target_time=modis_time,
                                tile_id=int(tile["tile_id"]),
                                lats=target_lats,
                                lons=target_lons,
                            )
                            if sample is None:
                                rejected_dynamic_time += 1
                            elif sample.shape[1:] == (MODIS_TILE_SIZE, MODIS_TILE_SIZE) and np.isfinite(sample).all():
                                dynamic_stack = sample.astype(np.float32)
                                valid_dynamic = 1
                                accepted_dynamic_time_diffs.append(int(dynamic_dt_min))
                            else:
                                rejected_dynamic_time += 1
                        except Exception as exc:
                            err_key = type(exc).__name__
                            if err_key not in _seen_dynamic_errors:
                                _seen_dynamic_errors.add(err_key)
                                print(
                                    f"  WARNING: dynamic provider error ({err_key}): {exc} "
                                    f"[further {err_key} errors suppressed]"
                                )
                            rejected_dynamic_time += 1

                    mask = np.full((1, MODIS_TILE_SIZE, MODIS_TILE_SIZE), valid_dynamic, dtype=np.float32)
                    x_stack = np.concatenate([x_stack, dynamic_stack, mask], axis=0)

                if expected_x_channels is not None and x_stack.shape[0] != int(expected_x_channels):
                    raise RuntimeError(
                        f"Unexpected input channel count for {tif_path.name}: "
                        f"got {x_stack.shape[0]}, expected {expected_x_channels}."
                    )

                # Neural networks cannot train on NaNs. Drop any sample where
                # ERA5 or the selected dynamic feature source left gaps.
                if not np.isfinite(x_stack).all():
                    continue

                store["X"].append(x_stack)
                store["Y"].append(cot[np.newaxis, :, :])
                store["tile_id"].append(tile["tile_id"])
                store["timestamp"].append(modis_time.strftime("%Y-%m-%d %H:%M"))
                store["year"].append(year)
                year_added += 1

    ds.close()
    print(f"Accepted samples from {year}: {year_added}")
    print(
        f"Skipped MODIS files due to time mismatch: ERA5={rejected_era5_time}, "
        f"dynamic={rejected_dynamic_time}"
    )
    if dynamic_provider is not None and accepted_dynamic_time_diffs:
        diffs = np.asarray(accepted_dynamic_time_diffs, dtype=np.int32)
        print(
            "Accepted dynamic-feature↔MODIS pairing time-diff (minutes): "
            f"mean={float(np.mean(diffs)):.2f}, p95={float(np.percentile(diffs, 95)):.1f}, "
            f"max={int(np.max(diffs))}"
        )


def finalize_and_save(store, out_file, expected_x_channels=None):
    if not store["X"]:
        raise RuntimeError(f"No samples produced for {out_file}.")

    X = np.stack(store["X"])
    Y = np.stack(store["Y"])
    tile_id = np.array(store["tile_id"], dtype=np.int32)
    timestamp = np.array(store["timestamp"])
    year = np.array(store["year"], dtype=np.int16)

    if expected_x_channels is not None and X.shape[1] != int(expected_x_channels):
        raise RuntimeError(
            f"Shard {out_file} has X channels={X.shape[1]}, expected {expected_x_channels}."
        )

    np.savez_compressed(out_file, X=X, Y=Y, tile_id=tile_id, timestamp=timestamp, year=year)
    print(f"\nShard saved: {out_file}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"samples: {X.shape[0]}")


def build_group(split_name, years, dynamic_provider=None, dynamic_feature_channels=None, expected_x_channels=None):
    store = {"X": [], "Y": [], "tile_id": [], "timestamp": [], "year": []}
    for year in years:
        process_year(
            year,
            store,
            dynamic_provider=dynamic_provider,
            dynamic_feature_channels=dynamic_feature_channels,
            expected_x_channels=expected_x_channels,
        )

    out_file = SHARD_DIR / f"{split_name}_{years[0]}_{years[-1]}.npz"
    finalize_and_save(store, out_file, expected_x_channels=expected_x_channels)


def main():
    print(f"ERA5 root: {ERA5_ROOT}")
    print(f"MODIS dir: {MODIS_DIR}")
    print(f"Shard dir: {SHARD_DIR}")
    print(
        f"Tile setup: MODIS {MODIS_TILE_SIZE}x{MODIS_TILE_SIZE} at {MODIS_PIXEL_KM} km, "
        f"ERA5 coarse {ERA5_TILE_SIZE}x{ERA5_TILE_SIZE} at {ERA5_PIXEL_KM} km "
        f"upsampled by {UPSAMPLE_FACTOR}x."
    )
    print(f"ERA5 feature channels configured: {len(ERA5_FEATURE_SPECS)}")

    dynamic_provider = None
    dynamic_feature_channels = None
    expected_x_channels = len(ERA5_FEATURE_SPECS)
    if USE_PRITHVI and USE_LANDSAT_HLS:
        raise ValueError("Enable only one dynamic feature source: Prithvi or Landsat HLS.")

    if USE_PRITHVI:
        print("Prithvi dynamic features: ENABLED")
        print(f"  source: {PRITHVI_SOURCE_PATH}")
        print(f"  pca model: {PRITHVI_PCA_MODEL_PATH}")
        print(f"  max time gap (minutes): {PRITHVI_MAX_TIME_DIFF_MINUTES}")
        dynamic_provider = PrithviDynamicProvider(
            source_path=PRITHVI_SOURCE_PATH,
            pca_model_path=PRITHVI_PCA_MODEL_PATH,
            variable_name=PRITHVI_VARIABLE_NAME,
            max_time_diff_minutes=PRITHVI_MAX_TIME_DIFF_MINUTES,
            time_dim=PRITHVI_TIME_DIM,
            channel_dim=PRITHVI_CHANNEL_DIM,
            lat_dim=PRITHVI_LAT_DIM,
            lon_dim=PRITHVI_LON_DIM,
            tile_cache_size=PRITHVI_TILE_CACHE_SIZE,
        )
        print(
            f"  Prithvi channels after PCA: {dynamic_provider.n_components} "
            f"(raw channels: {dynamic_provider.da.sizes['channel']})"
        )
        dynamic_feature_channels = int(dynamic_provider.n_components)
        expected_x_channels += dynamic_feature_channels + 1  # +1 for binary mask
    elif USE_LANDSAT_HLS:
        print("Landsat-8 HLS dynamic features: ENABLED")
        print(f"  source: {LANDSAT_HLS_SOURCE_PATH}")
        print(f"  lag days: {LANDSAT_HLS_LAG_DAYS}")
        print(f"  max staleness days: {LANDSAT_HLS_MAX_STALENESS_DAYS}")
        dynamic_provider = LandsatHLSDynamicProvider(
            source_path=LANDSAT_HLS_SOURCE_PATH,
            variable_name=LANDSAT_HLS_VARIABLE_NAME,
            lag_days=LANDSAT_HLS_LAG_DAYS,
            max_staleness_days=LANDSAT_HLS_MAX_STALENESS_DAYS,
            time_dim=LANDSAT_HLS_TIME_DIM,
            channel_dim=LANDSAT_HLS_CHANNEL_DIM,
            lat_dim=LANDSAT_HLS_LAT_DIM,
            lon_dim=LANDSAT_HLS_LON_DIM,
            tile_cache_size=LANDSAT_HLS_TILE_CACHE_SIZE,
            coord_tolerance_deg=LANDSAT_HLS_COORD_TOL_DEG,
        )
        print(f"  Landsat/HLS channels: {dynamic_provider.n_channels}")
        dynamic_feature_channels = int(dynamic_provider.n_channels)
        expected_x_channels += dynamic_feature_channels + 1  # +1 for binary mask
    else:
        print("Dynamic feature source: DISABLED")
    print(f"Expected input channels in generated shards: {expected_x_channels}")

    for years in TRAIN_GROUPS:
        build_group(
            "train",
            years,
            dynamic_provider=dynamic_provider,
            dynamic_feature_channels=dynamic_feature_channels,
            expected_x_channels=expected_x_channels,
        )

    for years in VAL_GROUPS:
        build_group(
            "val",
            years,
            dynamic_provider=dynamic_provider,
            dynamic_feature_channels=dynamic_feature_channels,
            expected_x_channels=expected_x_channels,
        )

    for years in TEST_GROUPS:
        build_group(
            "test",
            years,
            dynamic_provider=dynamic_provider,
            dynamic_feature_channels=dynamic_feature_channels,
            expected_x_channels=expected_x_channels,
        )

    print("\nDone. Next step: run merge script to create final train/validate/test files.")


if __name__ == "__main__":
    main()
