#!/usr/bin/env python3
"""
Validate HLS/Landsat tile NetCDF outputs used by the FNO pipeline.

This checks:
  - filename pattern and tile/year extraction
  - NetCDF readability
  - exactly one data variable (unless --variable-name is passed)
  - standardizable dimensions: time x channel x latitude x longitude
  - expected shape: (*, 7, 108, 108)
  - monotonic / unique time, latitude, longitude coordinates
  - channel labels match the fetch script's FEATURE_NAMES when present
  - timestamps belong to the filename year
  - at least some finite data exist
  - exact lat/lon alignment against build_tile_windows() from fetch_hls_landsat_features.py

It can give high confidence on structural integrity and area alignment, but it cannot
provide a 100% scientific guarantee that the source scenes themselves are "correct".
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.hls_landsat_utils import TILE_FILE_RE, _standardize_dataarray


@dataclass
class ValidationResult:
    path: str
    tile_id: int
    year: int
    ok: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, object]


def load_expected_tile_specs() -> tuple[dict[int, dict[str, np.ndarray]], list[str], int]:
    from data.fetch_hls_landsat_features import FEATURE_NAMES, MODIS_TILE_SIZE, build_tile_windows

    tile_specs = {}
    for tile in build_tile_windows():
        tile_specs[int(tile["tile_id"])] = {
            "latitude": np.asarray(tile["latitude"], dtype=np.float64),
            "longitude": np.asarray(tile["longitude"], dtype=np.float64),
        }
    return tile_specs, list(FEATURE_NAMES), int(MODIS_TILE_SIZE)


def choose_variable(ds: xr.Dataset, variable_name: Optional[str]) -> xr.DataArray:
    if variable_name:
        if variable_name not in ds.data_vars:
            raise ValueError(
                f"Requested variable_name={variable_name!r} not found. "
                f"Available variables: {list(ds.data_vars)}"
            )
        return ds[variable_name]

    if len(ds.data_vars) != 1:
        raise ValueError(
            "Expected exactly one data variable unless --variable-name is provided. "
            f"Available variables: {list(ds.data_vars)}"
        )
    return ds[list(ds.data_vars)[0]]


def validate_file(
    path: Path,
    *,
    tile_specs: dict[int, dict[str, np.ndarray]],
    expected_feature_names: list[str],
    expected_tile_size: int,
    variable_name: Optional[str],
    coord_atol: float,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, object] = {}

    match = TILE_FILE_RE.match(path.name)
    if not match:
        return ValidationResult(
            path=str(path),
            tile_id=-1,
            year=-1,
            ok=False,
            errors=[f"Filename does not match expected pattern: {path.name}"],
            warnings=[],
            stats={},
        )

    tile_id = int(match.group("tile"))
    year = int(match.group("year"))
    stats["filename"] = path.name

    if tile_id not in tile_specs:
        errors.append(f"tile_id={tile_id} is not part of the expected tile grid.")
        return ValidationResult(
            path=str(path), tile_id=tile_id, year=year, ok=False, errors=errors, warnings=warnings, stats=stats
        )

    try:
        with xr.open_dataset(path) as ds:
            da = choose_variable(ds, variable_name).load()
    except Exception as exc:
        errors.append(f"Failed to open/read NetCDF: {exc}")
        return ValidationResult(
            path=str(path), tile_id=tile_id, year=year, ok=False, errors=errors, warnings=warnings, stats=stats
        )

    try:
        da = _standardize_dataarray(da)
    except Exception as exc:
        errors.append(f"Could not standardize dims to time/channel/latitude/longitude: {exc}")
        return ValidationResult(
            path=str(path), tile_id=tile_id, year=year, ok=False, errors=errors, warnings=warnings, stats=stats
        )

    stats["dims"] = {k: int(v) for k, v in da.sizes.items()}
    stats["n_rows"] = int(da.sizes["time"])

    if tuple(da.dims) != ("time", "channel", "latitude", "longitude"):
        errors.append(f"Unexpected dimension order after standardization: {da.dims}")

    if int(da.sizes["channel"]) != len(expected_feature_names):
        errors.append(
            f"Expected {len(expected_feature_names)} channels, found {int(da.sizes['channel'])}."
        )
    if int(da.sizes["latitude"]) != expected_tile_size or int(da.sizes["longitude"]) != expected_tile_size:
        errors.append(
            "Expected spatial shape "
            f"{expected_tile_size}x{expected_tile_size}, found "
            f"{int(da.sizes['latitude'])}x{int(da.sizes['longitude'])}."
        )
    if int(da.sizes["time"]) <= 0:
        errors.append("No time slices found.")

    time_values = np.asarray(da["time"].values)
    lat_values = np.asarray(da["latitude"].values, dtype=np.float64)
    lon_values = np.asarray(da["longitude"].values, dtype=np.float64)

    if time_values.size:
        stats["time_start"] = str(np.min(time_values))
        stats["time_end"] = str(np.max(time_values))
        if np.unique(time_values).size != time_values.size:
            errors.append("Duplicate timestamps found.")
        if np.any(np.diff(time_values) <= np.timedelta64(0, "ns")):
            errors.append("Timestamps are not strictly increasing.")
        years = np.unique(time_values.astype("datetime64[Y]").astype(int) + 1970)
        if years.size != 1 or int(years[0]) != year:
            errors.append(f"Filename year={year} does not match time coordinate years={years.tolist()}.")

    if lat_values.size:
        if not np.all(np.isfinite(lat_values)):
            errors.append("Latitude coordinate contains non-finite values.")
        if np.any(np.diff(lat_values) >= 0):
            errors.append("Latitude coordinate is not strictly descending.")

    if lon_values.size:
        if not np.all(np.isfinite(lon_values)):
            errors.append("Longitude coordinate contains non-finite values.")
        if np.any(np.diff(lon_values) <= 0):
            errors.append("Longitude coordinate is not strictly ascending.")

    expected_lat = tile_specs[tile_id]["latitude"]
    expected_lon = tile_specs[tile_id]["longitude"]
    if lat_values.shape != expected_lat.shape or not np.allclose(lat_values, expected_lat, atol=coord_atol, rtol=0.0):
        errors.append(
            f"Latitude grid does not align with expected tile geometry for tile_id={tile_id}."
        )
    if lon_values.shape != expected_lon.shape or not np.allclose(lon_values, expected_lon, atol=coord_atol, rtol=0.0):
        errors.append(
            f"Longitude grid does not align with expected tile geometry for tile_id={tile_id}."
        )

    if "channel" in da.coords:
        channel_values = [str(x) for x in da["channel"].values.tolist()]
        stats["channel_names"] = channel_values
        if channel_values != expected_feature_names:
            errors.append(
                f"Channel labels do not match expected features. Found={channel_values}, "
                f"expected={expected_feature_names}"
            )
    else:
        warnings.append("No channel coordinate labels found; channel order could not be verified.")

    values = np.asarray(da.values, dtype=np.float32)
    finite_mask = np.isfinite(values)
    finite_fraction = float(finite_mask.mean()) if finite_mask.size else 0.0
    stats["finite_fraction"] = finite_fraction
    if finite_fraction == 0.0:
        errors.append("All values are non-finite.")
    elif finite_fraction < 0.01:
        warnings.append(f"Very low finite-data fraction: {finite_fraction:.4f}")

    if values.shape[0] > 0:
        per_time_finite = finite_mask.reshape(values.shape[0], -1).any(axis=1)
        empty_times = int((~per_time_finite).sum())
        stats["empty_time_slices"] = empty_times
        if empty_times > 0:
            warnings.append(f"{empty_times} time slices contain no finite values.")

        first_idx = 0
        first_time = str(time_values[first_idx])
        first_slice = values[first_idx]
        example_means = {}
        channel_names = stats.get("channel_names", expected_feature_names)
        for channel_idx, channel_name in enumerate(channel_names):
            channel_slice = first_slice[channel_idx]
            example_means[str(channel_name)] = float(np.nanmean(channel_slice))
        center_y = first_slice.shape[1] // 2
        center_x = first_slice.shape[2] // 2
        center_pixel = {
            str(channel_names[channel_idx]): float(first_slice[channel_idx, center_y, center_x])
            for channel_idx in range(first_slice.shape[0])
        }
        stats["example_instance"] = {
            "timestamp": first_time,
            "channel_means": example_means,
            "center_pixel": center_pixel,
        }

    return ValidationResult(
        path=str(path),
        tile_id=tile_id,
        year=year,
        ok=not errors,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HLS/Landsat tile NetCDF files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing hls_landsat_tile*.nc")
    parser.add_argument("--variable-name", type=str, default=None, help="Optional data variable name override.")
    parser.add_argument("--start-year", type=int, default=None, help="Optional minimum filename year filter.")
    parser.add_argument("--end-year", type=int, default=None, help="Optional maximum filename year filter.")
    parser.add_argument("--tile-id", type=int, default=None, help="Optional tile id filter.")
    parser.add_argument("--coord-atol", type=float, default=1e-6, help="Absolute tolerance for coord alignment.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON report path.")
    return parser.parse_args()


def should_include(path: Path, start_year: Optional[int], end_year: Optional[int], tile_id: Optional[int]) -> bool:
    match = TILE_FILE_RE.match(path.name)
    if not match:
        return False
    path_tile = int(match.group("tile"))
    path_year = int(match.group("year"))
    if start_year is not None and path_year < start_year:
        return False
    if end_year is not None and path_year > end_year:
        return False
    if tile_id is not None and path_tile != tile_id:
        return False
    return True


def compute_missing_files(
    *,
    files: list[Path],
    tile_specs: dict[int, dict[str, np.ndarray]],
    start_year: Optional[int],
    end_year: Optional[int],
    tile_id: Optional[int],
) -> list[dict[str, object]]:
    if start_year is None or end_year is None:
        return []

    expected_tiles = [int(tile_id)] if tile_id is not None else sorted(tile_specs)
    existing = set()
    for path in files:
        match = TILE_FILE_RE.match(path.name)
        if match:
            existing.add((int(match.group("tile")), int(match.group("year"))))

    missing = []
    for year in range(int(start_year), int(end_year) + 1):
        for expected_tile in expected_tiles:
            if (expected_tile, year) not in existing:
                missing.append({"tile_id": expected_tile, "year": year})
    return missing


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    tile_specs, expected_feature_names, expected_tile_size = load_expected_tile_specs()
    files = sorted(path for path in input_dir.glob("*.nc") if should_include(path, args.start_year, args.end_year, args.tile_id))

    if not files:
        raise FileNotFoundError(f"No matching HLS/Landsat NetCDF files found under: {input_dir}")

    results = [
        validate_file(
            path,
            tile_specs=tile_specs,
            expected_feature_names=expected_feature_names,
            expected_tile_size=expected_tile_size,
            variable_name=args.variable_name,
            coord_atol=float(args.coord_atol),
        )
        for path in files
    ]
    missing_files = compute_missing_files(
        files=files,
        tile_specs=tile_specs,
        start_year=args.start_year,
        end_year=args.end_year,
        tile_id=args.tile_id,
    )

    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count

    report = {
        "input_dir": str(input_dir.resolve()),
        "checked_files": len(results),
        "ok_files": ok_count,
        "failed_files": fail_count,
        "missing_files": missing_files,
        "results": [
            {
                "path": r.path,
                "tile_id": r.tile_id,
                "year": r.year,
                "ok": r.ok,
                "errors": r.errors,
                "warnings": r.warnings,
                "stats": r.stats,
            }
            for r in results
        ],
    }

    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status}] tile={r.tile_id:03d} year={r.year} file={Path(r.path).name}")
        if r.stats:
            print(
                f"  rows={r.stats.get('n_rows')} "
                f"time_range=[{r.stats.get('time_start')}, {r.stats.get('time_end')}] "
                f"finite_fraction={r.stats.get('finite_fraction')}"
            )
            example = r.stats.get("example_instance")
            if isinstance(example, dict):
                print(f"  example timestamp: {example.get('timestamp')}")
                print(f"  example channel means: {example.get('channel_means')}")
        for msg in r.errors:
            print(f"  error: {msg}")
        for msg in r.warnings:
            print(f"  warn : {msg}")

    print(
        f"\nSummary: checked={len(results)} ok={ok_count} failed={fail_count}"
    )
    if missing_files:
        print(f"Missing expected files: {len(missing_files)}")
    if fail_count == 0:
        print(
            "Structural and alignment checks passed. This is high confidence, not a 100% guarantee of scientific correctness."
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report to: {out_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
