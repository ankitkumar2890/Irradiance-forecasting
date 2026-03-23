"""
Fetch Landsat-8 HLS-derived features for the FNO pipeline.

Features produced per 1 km FNO tile:
  - NDMI
  - NDVI
  - NDWI
  - MNDWI
  - SWIR1 (B06)
  - Surface temperature proxy (B10 brightness temperature)
  - Elevation (Copernicus DEM GLO-30)

Key design choices:
  - Uses HLSL30 (Landsat-8/9 HLS) only.
  - Aggregates onto the same 108x108 ~1 km tile grid used by data/fno_dataset.py.
  - Uses rasterio reprojection with average resampling for continuous HLS bands and
    bilinear resampling for DEM.
  - Stores per-tile, per-year NetCDF files that can be consumed by
    data/hls_landsat_utils.py.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.windows import Window, bounds as window_bounds
import xarray as xr
from dotenv import load_dotenv


DEFAULT_START_YEAR = 2013
DEFAULT_END_YEAR = 2013
DEFAULT_MAX_CLOUD_COVER = 80.0
DEFAULT_MAX_SCENES_PER_TILE_YEAR = 12
REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
OUTPUT_DIR = Path("/Users/IRFAN/Library/CloudStorage/GoogleDrive-irfan.a@atriauniversity.edu.in/My Drive/Irradiance-forecasting/hls_landsat_tiles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EARTHDATA_TOKEN_ENV_VARS = ("LAADS_TOKEN", "EARTHDATA_TOKEN", "EDL_TOKEN", "LPDAAC_TOKEN")
ASSET_DOWNLOAD_RETRIES = 4
ASSET_DOWNLOAD_BACKOFF_SECONDS = 3
STAC_SEARCH_RETRIES = 4
STAC_SEARCH_BACKOFF_SECONDS = 5

# Shared domain and tile design from MODIS/FNO pipeline.
DOMAIN_NORTH = 17.0
DOMAIN_SOUTH = 8.0
DOMAIN_WEST = 72.5
DOMAIN_EAST = 81.5
DOMAIN_CENTER_LAT = 0.5 * (DOMAIN_NORTH + DOMAIN_SOUTH)
MODIS_TILE_SIZE = 108

# Approximate 1 km geographic grid used in MODIS preprocessing.
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * math.cos(math.radians(DOMAIN_CENTER_LAT))
PIXEL_LAT_DEG = 1.0 / KM_PER_DEG_LAT
PIXEL_LON_DEG = 1.0 / KM_PER_DEG_LON
GLOBAL_GRID_H = int((DOMAIN_NORTH - DOMAIN_SOUTH) / PIXEL_LAT_DEG)
GLOBAL_GRID_W = int((DOMAIN_EAST - DOMAIN_WEST) / PIXEL_LON_DEG)
TILES_H = GLOBAL_GRID_H // MODIS_TILE_SIZE
TILES_W = GLOBAL_GRID_W // MODIS_TILE_SIZE

HLS_STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
HLS_COLLECTION = "HLSL30_2.0"
DEM_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEM_COLLECTION = "cop-dem-glo-30"
HLS_ASSETS = {
    "green": "B03",
    "red": "B04",
    "nir": "B05",
    "swir1": "B06",
    "b10": "B10",
}
FEATURE_NAMES = [
    "ndmi",
    "ndvi",
    "ndwi",
    "mndwi",
    "swir1_b06",
    "brightness_temp_b10",
    "elevation_dem",
]


def build_tile_windows():
    windows = []
    full_transform = from_bounds(
        DOMAIN_WEST, DOMAIN_SOUTH, DOMAIN_EAST, DOMAIN_NORTH, GLOBAL_GRID_W, GLOBAL_GRID_H
    )
    for tr in range(TILES_H):
        for tc in range(TILES_W):
            window = Window(
                col_off=tc * MODIS_TILE_SIZE,
                row_off=tr * MODIS_TILE_SIZE,
                width=MODIS_TILE_SIZE,
                height=MODIS_TILE_SIZE,
            )
            bounds = window_bounds(window, full_transform)
            tile_id = tr * TILES_W + tc
            tile_transform = from_bounds(*bounds, MODIS_TILE_SIZE, MODIS_TILE_SIZE)
            lat_step = (bounds[3] - bounds[1]) / MODIS_TILE_SIZE
            lon_step = (bounds[2] - bounds[0]) / MODIS_TILE_SIZE
            lat = bounds[3] - (np.arange(MODIS_TILE_SIZE, dtype=np.float32) + 0.5) * lat_step
            lon = bounds[0] + (np.arange(MODIS_TILE_SIZE, dtype=np.float32) + 0.5) * lon_step
            windows.append(
                {
                    "tile_id": tile_id,
                    "bounds": bounds,
                    "transform": tile_transform,
                    "latitude": lat,
                    "longitude": lon,
                }
            )
    return windows


def _asset_scale_offset(asset) -> tuple[float, float]:
    bands = asset.extra_fields.get("raster:bands", [])
    if bands:
        band0 = bands[0]
        scale = float(band0.get("scale", 1.0))
        offset = float(band0.get("offset", 0.0))
        return scale, offset
    return 1.0, 0.0


def _signed_pc_href(asset) -> str:
    import planetary_computer

    href = asset.href
    return planetary_computer.sign(href)


def _get_earthdata_token() -> str:
    for env_name in EARTHDATA_TOKEN_ENV_VARS:
        token = os.getenv(env_name)
        if token:
            return token
    raise RuntimeError(
        "Missing Earthdata bearer token. Set one of: "
        + ", ".join(EARTHDATA_TOKEN_ENV_VARS)
    )


def _download_protected_asset(asset) -> str:
    import requests

    token = _get_earthdata_token()
    href = asset.href
    suffix = Path(href).suffix or ".tif"
    headers = {"Authorization": f"Bearer {token}"}
    last_error = None

    for attempt in range(1, ASSET_DOWNLOAD_RETRIES + 1):
        fd, tmp_path = tempfile.mkstemp(prefix="hls_asset_", suffix=suffix)
        os.close(fd)
        try:
            with requests.get(href, headers=headers, stream=True, timeout=120) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if "html" in content_type:
                    raise RuntimeError(
                        f"Authenticated request for {href} returned HTML instead of raster data."
                    )
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return tmp_path
        except Exception as exc:
            last_error = exc
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            if attempt < ASSET_DOWNLOAD_RETRIES:
                sleep_s = ASSET_DOWNLOAD_BACKOFF_SECONDS * attempt
                print(
                    f"    asset download retry {attempt}/{ASSET_DOWNLOAD_RETRIES - 1} "
                    f"for {Path(href).name} after error: {exc}"
                )
                time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download asset after retries: {href}") from last_error


def read_asset_to_tile(asset, *, tile_transform, out_shape, resampling) -> np.ndarray:
    local_path = _download_protected_asset(asset)
    scale, offset = _asset_scale_offset(asset)

    try:
        with rasterio.open(local_path) as src:
            arr = src.read(1, masked=True).astype(np.float32)
            filled = arr.filled(np.nan)
            valid = np.isfinite(filled)
            filled[valid] = filled[valid] * scale + offset

            src_fill = np.float32(-9999.0)
            filled = np.where(np.isfinite(filled), filled, src_fill).astype(np.float32)
            dst = np.full(out_shape, np.float32(-9999.0), dtype=np.float32)

            reproject(
                source=filled,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_fill,
                dst_transform=tile_transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.float32(-9999.0),
                resampling=resampling,
            )
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass

    dst = np.where(dst == np.float32(-9999.0), np.nan, dst)
    return dst.astype(np.float32)


def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float32)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-6)
    out[mask] = num[mask] / den[mask]
    return out


def compute_feature_stack(green, red, nir, swir1, b10, dem) -> np.ndarray:
    ndvi = safe_ratio(nir - red, nir + red)
    ndmi = safe_ratio(nir - swir1, nir + swir1)
    ndwi = safe_ratio(green - nir, green + nir)
    mndwi = safe_ratio(green - swir1, green + swir1)
    return np.stack(
        [
            ndmi,
            ndvi,
            ndwi,
            mndwi,
            swir1.astype(np.float32),
            b10.astype(np.float32),
            dem.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def pick_dem_asset(item):
    for key in ("data", "dem", "elevation"):
        if key in item.assets:
            return item.assets[key]
    for asset in item.assets.values():
        if asset.media_type and "tiff" in asset.media_type.lower():
            return asset
    raise KeyError(f"Could not find DEM asset in item {item.id}")


def fetch_dem_tile(catalog, tile_info) -> np.ndarray:
    bbox = list(tile_info["bounds"])
    search = catalog.search(collections=[DEM_COLLECTION], bbox=bbox)
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No DEM items found for tile_id={tile_info['tile_id']}")

    sources = [rasterio.open(_signed_pc_href(pick_dem_asset(item))) for item in items]
    try:
        src_crs = sources[0].crs
        mosaic, transform = rio_merge(sources, bounds=bbox)
    finally:
        for src in sources:
            src.close()

    src_fill = np.float32(-9999.0)
    src_arr = mosaic[0].astype(np.float32)
    src_arr = np.where(np.isfinite(src_arr), src_arr, src_fill)
    dst = np.full((MODIS_TILE_SIZE, MODIS_TILE_SIZE), src_fill, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=transform,
        src_crs=src_crs,
        src_nodata=src_fill,
        dst_transform=tile_info["transform"],
        dst_crs="EPSG:4326",
        dst_nodata=src_fill,
        resampling=Resampling.bilinear,
    )
    return np.where(dst == src_fill, np.nan, dst).astype(np.float32)


def search_hls_items(catalog, year: int, bbox, max_cloud_cover: float):
    last_error = None
    for attempt in range(1, STAC_SEARCH_RETRIES + 1):
        try:
            search = catalog.search(
                collections=[HLS_COLLECTION],
                bbox=list(bbox),
                datetime=f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
                query={"eo:cloud_cover": {"lt": float(max_cloud_cover)}},
            )
            items = list(search.items())
            items.sort(
                key=lambda item: (
                    float(item.properties.get("eo:cloud_cover", 1000.0)),
                    item.datetime,
                )
            )
            return items
        except Exception as exc:
            last_error = exc
            if attempt < STAC_SEARCH_RETRIES:
                sleep_s = STAC_SEARCH_BACKOFF_SECONDS * attempt
                print(
                    f"  STAC search retry {attempt}/{STAC_SEARCH_RETRIES - 1} "
                    f"for year={year}, bbox={tuple(round(x, 4) for x in bbox)} after error: {exc}"
                )
                time.sleep(sleep_s)
    raise RuntimeError(
        f"Failed HLS STAC search after retries for year={year}, bbox={tuple(round(x, 4) for x in bbox)}"
    ) from last_error


def item_has_required_assets(item) -> bool:
    return all(asset_name in item.assets for asset_name in HLS_ASSETS.values())


def dedupe_and_sort_scene_stack(
    out_time: list[np.datetime64],
    out_stack: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Keep the first scene for each timestamp and write the final cube in time order.

    Candidate HLS items are sorted upstream by cloud cover and then datetime, so
    the first duplicate timestamp is the best-ranked scene for that time.
    """
    first_by_time: dict[np.datetime64, np.ndarray] = {}
    dropped_duplicates = 0

    for ts, feat in zip(out_time, out_stack):
        if ts in first_by_time:
            dropped_duplicates += 1
            continue
        first_by_time[ts] = feat

    ordered_times = sorted(first_by_time)
    ordered_stack = [first_by_time[ts] for ts in ordered_times]
    return (
        np.asarray(ordered_times),
        np.stack(ordered_stack, axis=0).astype(np.float32),
        dropped_duplicates,
    )


def process_tile_year(
    hls_catalog,
    dem_catalog,
    *,
    tile_info,
    year: int,
    max_cloud_cover: float,
    max_scenes_per_tile_year: int | None,
    output_dir: Path,
    dem_cache: dict[int, np.ndarray],
    skip_existing: bool,
):
    out_path = output_dir / f"hls_landsat_tile{tile_info['tile_id']:03d}_year{year}.nc"
    if skip_existing and out_path.exists():
        print(f"[tile {tile_info['tile_id']:03d}] year={year} skipping existing {out_path.name}")
        return

    items = [
        item
        for item in search_hls_items(hls_catalog, year, tile_info["bounds"], max_cloud_cover)
        if item_has_required_assets(item)
    ]
    total_items = len(items)
    if max_scenes_per_tile_year is not None and max_scenes_per_tile_year > 0:
        items = items[:max_scenes_per_tile_year]
    print(
        f"[tile {tile_info['tile_id']:03d}] year={year} candidate HLS scenes={total_items} "
        f"using={len(items)} "
        f"bbox={tuple(round(x, 4) for x in tile_info['bounds'])}"
    )
    if not items:
        return

    tile_id = int(tile_info["tile_id"])
    if tile_id not in dem_cache:
        dem_cache[tile_id] = fetch_dem_tile(dem_catalog, tile_info)
    dem = dem_cache[tile_id]
    out_time = []
    out_stack = []

    for item in items:
        try:
            green = read_asset_to_tile(
                item.assets[HLS_ASSETS["green"]],
                tile_transform=tile_info["transform"],
                out_shape=(MODIS_TILE_SIZE, MODIS_TILE_SIZE),
                resampling=Resampling.average,
            )
            red = read_asset_to_tile(
                item.assets[HLS_ASSETS["red"]],
                tile_transform=tile_info["transform"],
                out_shape=(MODIS_TILE_SIZE, MODIS_TILE_SIZE),
                resampling=Resampling.average,
            )
            nir = read_asset_to_tile(
                item.assets[HLS_ASSETS["nir"]],
                tile_transform=tile_info["transform"],
                out_shape=(MODIS_TILE_SIZE, MODIS_TILE_SIZE),
                resampling=Resampling.average,
            )
            swir1 = read_asset_to_tile(
                item.assets[HLS_ASSETS["swir1"]],
                tile_transform=tile_info["transform"],
                out_shape=(MODIS_TILE_SIZE, MODIS_TILE_SIZE),
                resampling=Resampling.average,
            )
            b10 = read_asset_to_tile(
                item.assets[HLS_ASSETS["b10"]],
                tile_transform=tile_info["transform"],
                out_shape=(MODIS_TILE_SIZE, MODIS_TILE_SIZE),
                resampling=Resampling.average,
            )
        except Exception as exc:
            print(f"  skipping scene {item.id}: {exc}")
            continue

        feat = compute_feature_stack(green, red, nir, swir1, b10, dem)
        if not np.isfinite(feat).any():
            continue

        out_time.append(np.datetime64(item.datetime.replace(tzinfo=None)))
        out_stack.append(feat.astype(np.float32))

    if not out_stack:
        print(f"  no usable HLS scenes for tile={tile_info['tile_id']:03d}, year={year}")
        return

    ordered_time, ordered_stack, dropped_duplicates = dedupe_and_sort_scene_stack(out_time, out_stack)
    if dropped_duplicates:
        print(
            f"  deduped {dropped_duplicates} duplicate scene timestamps for "
            f"tile={tile_info['tile_id']:03d}, year={year}"
        )

    da = xr.DataArray(
        ordered_stack,
        dims=["time", "channel", "latitude", "longitude"],
        coords={
            "time": ordered_time,
            "channel": FEATURE_NAMES,
            "latitude": tile_info["latitude"],
            "longitude": tile_info["longitude"],
        },
        name="landsat_hls_features",
        attrs={
            "collection": HLS_COLLECTION,
            "dem_collection": DEM_COLLECTION,
            "native_hls_resolution_m": 30,
            "native_dem_resolution_m": 30,
            "target_resolution_km": 1,
            "n_raw_scenes": len(out_time),
            "n_unique_timestamps": int(len(ordered_time)),
            "n_dropped_duplicate_timestamps": int(dropped_duplicates),
            "notes": (
                "B10 is HLS Landsat brightness temperature, used here as a surface-temperature proxy. "
                "Provider-side historical matching applies an 8-day lag."
            ),
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(out_path)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Landsat-8 HLS feature cubes for FNO.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--tile-id", type=int, default=None, help="Optional single tile id to process.")
    parser.add_argument("--max-cloud-cover", type=float, default=DEFAULT_MAX_CLOUD_COVER)
    parser.add_argument(
        "--max-scenes-per-tile-year",
        type=int,
        default=DEFAULT_MAX_SCENES_PER_TILE_YEAR,
        help="Limit the number of HLS scenes used per tile/year after sorting by cloud cover.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rebuild NetCDF outputs even if they already exist.",
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    hls_catalog = pystac_client.Client.open(HLS_STAC_URL)
    dem_catalog = pystac_client.Client.open(
        DEM_STAC_URL,
        modifier=__import__("planetary_computer").sign_inplace,
    )

    tiles = build_tile_windows()
    if args.tile_id is not None:
        tiles = [tile for tile in tiles if tile["tile_id"] == int(args.tile_id)]
        if not tiles:
            raise ValueError(f"Unknown tile_id={args.tile_id}")

    print(
        f"Domain grid: {GLOBAL_GRID_H}x{GLOBAL_GRID_W}, usable tiles={TILES_H}x{TILES_W}={TILES_H*TILES_W}, "
        f"tile_size={MODIS_TILE_SIZE}x{MODIS_TILE_SIZE}"
    )
    print("Native source resolutions: HLS L30 ~= 30 m grid, DEM ~= 30 m, output = 1 km")
    print(
        f"Fetch policy: max_cloud_cover<{float(args.max_cloud_cover)}, "
        f"max_scenes_per_tile_year={args.max_scenes_per_tile_year}, "
        f"skip_existing={not args.no_skip_existing}"
    )

    dem_cache: dict[int, np.ndarray] = {}
    for year in range(args.start_year, args.end_year + 1):
        for tile in tiles:
            try:
                process_tile_year(
                    hls_catalog,
                    dem_catalog,
                    tile_info=tile,
                    year=year,
                    max_cloud_cover=float(args.max_cloud_cover),
                    max_scenes_per_tile_year=args.max_scenes_per_tile_year,
                    output_dir=output_dir,
                    dem_cache=dem_cache,
                    skip_existing=not args.no_skip_existing,
                )
            except Exception as exc:
                print(
                    f"[tile {tile['tile_id']:03d}] year={year} failed after retries: {exc}"
                )


if __name__ == "__main__":
    main()
