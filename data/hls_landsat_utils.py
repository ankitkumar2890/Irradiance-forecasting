"""
Utilities for serving Landsat-8/HLS feature cubes to the FNO dataset builder.

Expected source layout:
  hls_landsat_tile000_year2013.nc
  hls_landsat_tile000_year2014.nc
  ...

Each NetCDF file must contain one data variable with dims:
  time x channel x latitude x longitude

The provider applies a real-world availability lag: for a MODIS/FNO sample at time T,
the selected HLS observation must be at or before T - lag_days.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import re

import numpy as np
import xarray as xr


MIN_HLS_YEAR = 2013
TILE_FILE_RE = re.compile(
    r"hls_landsat_tile(?P<tile>\d+)_year(?P<year>\d{4})\.nc$"
)


def _pick_name(candidates, available, label: str) -> str:
    for name in candidates:
        if name in available:
            return name
    raise ValueError(f"Could not infer {label}. Available names: {list(available)}")


def _standardize_dataarray(
    da: xr.DataArray,
    *,
    time_dim: Optional[str] = None,
    channel_dim: Optional[str] = None,
    lat_dim: Optional[str] = None,
    lon_dim: Optional[str] = None,
) -> xr.DataArray:
    dims = da.dims
    t_dim = time_dim or _pick_name(
        ["time", "valid_time", "timestamp", "datetime"], dims, "time dimension"
    )
    c_dim = channel_dim or _pick_name(
        ["channel", "feature", "band", "dim"], dims, "channel dimension"
    )
    la_dim = lat_dim or _pick_name(
        ["latitude", "lat", "y"], dims, "latitude dimension"
    )
    lo_dim = lon_dim or _pick_name(
        ["longitude", "lon", "x"], dims, "longitude dimension"
    )
    da = da.transpose(t_dim, c_dim, la_dim, lo_dim)
    da = da.rename(
        {
            t_dim: "time",
            c_dim: "channel",
            la_dim: "latitude",
            lo_dim: "longitude",
        }
    )
    if not np.all(np.diff(da["latitude"].values) < 0):
        da = da.isel(latitude=np.argsort(da["latitude"].values)[::-1])
    if not np.all(np.diff(da["longitude"].values) > 0):
        da = da.isel(longitude=np.argsort(da["longitude"].values))
    return da.sortby("time")


def _latest_past_time_index(time_values: np.ndarray, cutoff_time: datetime) -> tuple[int, int]:
    cutoff64 = np.datetime64(cutoff_time)
    valid_idx = np.where(time_values <= cutoff64)[0]
    if valid_idx.size == 0:
        return -1, 0
    idx = int(valid_idx[-1])
    age_minutes = int((cutoff64 - time_values[idx]) / np.timedelta64(1, "m"))
    return idx, age_minutes


def _coords_match(expected: np.ndarray, actual: np.ndarray, atol: float = 1e-6) -> bool:
    if expected.shape != actual.shape:
        return False
    return bool(np.allclose(expected, actual, atol=atol, rtol=0.0, equal_nan=False))


class LandsatHLSDynamicProvider:
    """Loads per-tile HLS feature stacks and serves lagged observations."""

    def __init__(
        self,
        source_path: Path,
        *,
        variable_name: Optional[str] = None,
        lag_days: int = 8,
        max_staleness_days: int = 45,
        time_dim: Optional[str] = None,
        channel_dim: Optional[str] = None,
        lat_dim: Optional[str] = None,
        lon_dim: Optional[str] = None,
        tile_cache_size: int = 32,
        coord_tolerance_deg: float = 1e-6,
    ):
        self.source_path = Path(source_path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Landsat/HLS source not found: {self.source_path}")

        self.variable_name = variable_name
        self.lag_days = int(lag_days)
        self.max_staleness_minutes = int(max_staleness_days) * 24 * 60
        self.time_dim = time_dim
        self.channel_dim = channel_dim
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim
        self.tile_cache_size = max(int(tile_cache_size), 0)
        self.coord_tolerance_deg = float(coord_tolerance_deg)
        self._tile_cache: dict[int, xr.DataArray] = {}
        self._tile_sources = self._discover_tile_sources()
        if not self._tile_sources:
            raise FileNotFoundError(
                f"No HLS tile NetCDF files found under: {self.source_path}"
            )

        first_tile_id = sorted(self._tile_sources)[0]
        first_da = self._load_tile_da(first_tile_id)
        self.n_channels = int(first_da.sizes["channel"])

    def _discover_tile_sources(self) -> dict[int, list[Path]]:
        out: dict[int, list[Path]] = {}
        for path in sorted(self.source_path.glob("*.nc")):
            match = TILE_FILE_RE.match(path.name)
            if not match:
                continue
            tile_id = int(match.group("tile"))
            out.setdefault(tile_id, []).append(path)
        return out

    def _load_tile_da(self, tile_id: int) -> xr.DataArray:
        if tile_id in self._tile_cache:
            return self._tile_cache[tile_id]

        paths = self._tile_sources.get(int(tile_id))
        if not paths:
            raise KeyError(f"No HLS tile file found for tile_id={tile_id}")

        ds = xr.open_mfdataset(paths, combine="by_coords")
        if self.variable_name is None:
            if len(ds.data_vars) != 1:
                raise ValueError(
                    f"HLS tile sources for tile_id={tile_id} expose multiple variables. "
                    "Set variable_name explicitly."
                )
            variable_name = list(ds.data_vars)[0]
        else:
            variable_name = self.variable_name

        da = _standardize_dataarray(
            ds[variable_name].load(),
            time_dim=self.time_dim,
            channel_dim=self.channel_dim,
            lat_dim=self.lat_dim,
            lon_dim=self.lon_dim,
        )
        if self.tile_cache_size > 0:
            if len(self._tile_cache) >= self.tile_cache_size:
                self._tile_cache.clear()
            self._tile_cache[int(tile_id)] = da
        return da

    def sample_tile_for_time(
        self,
        *,
        target_time: datetime,
        tile_id: int,
        lats: Optional[np.ndarray] = None,
        lons: Optional[np.ndarray] = None,
    ) -> tuple[Optional[np.ndarray], int]:
        if target_time.year < MIN_HLS_YEAR:
            return None, 0

        if int(tile_id) not in self._tile_sources:
            return None, 0

        da = self._load_tile_da(int(tile_id))
        cutoff_time = target_time - timedelta(days=self.lag_days)
        time_idx, age_minutes = _latest_past_time_index(da["time"].values, cutoff_time)
        if time_idx < 0 or age_minutes > self.max_staleness_minutes:
            return None, age_minutes

        sample = da.isel(time=int(time_idx)).values.astype(np.float32)
        if sample.ndim != 3:
            raise ValueError(f"Expected sample ndim=3, got shape={sample.shape}")

        if lats is not None:
            stored_lats = np.asarray(da["latitude"].values, dtype=np.float64)
            requested_lats = np.asarray(lats, dtype=np.float64)
            if sample.shape[1] != len(requested_lats):
                raise ValueError(
                    f"HLS latitude size mismatch for tile_id={tile_id}: "
                    f"sample={sample.shape[1]}, requested={len(requested_lats)}"
                )
            if not _coords_match(requested_lats, stored_lats, atol=self.coord_tolerance_deg):
                raise ValueError(
                    f"HLS latitude coordinates do not align with requested tile grid for tile_id={tile_id}."
                )
        if lons is not None:
            stored_lons = np.asarray(da["longitude"].values, dtype=np.float64)
            requested_lons = np.asarray(lons, dtype=np.float64)
            if sample.shape[2] != len(requested_lons):
                raise ValueError(
                    f"HLS longitude size mismatch for tile_id={tile_id}: "
                    f"sample={sample.shape[2]}, requested={len(requested_lons)}"
                )
            if not _coords_match(requested_lons, stored_lons, atol=self.coord_tolerance_deg):
                raise ValueError(
                    f"HLS longitude coordinates do not align with requested tile grid for tile_id={tile_id}."
                )

        return sample, age_minutes
