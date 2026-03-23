"""
Utilities for integrating dynamic Prithvi embeddings into the FNO pipeline.

Supported source layouts:
  1. A single NetCDF/Zarr with dims time x channel x latitude x longitude
  2. A directory of per-ERA5-tile NetCDF files named:
       prithvi_tile_lat{lat_min:.2f}_lon{lon_min:.2f}.nc
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
import re

import numpy as np
import xarray as xr


MIN_PRITHVI_YEAR = 2013
TILE_FILE_RE = re.compile(
    r"prithvi_tile_lat(?P<lat>-?\d+(?:\.\d+)?)_lon(?P<lon>-?\d+(?:\.\d+)?)\.nc$"
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
        ["channel", "embedding", "feature", "band", "dim"], dims, "channel dimension"
    )
    la_dim = lat_dim or _pick_name(
        ["latitude", "lat", "y"], dims, "latitude dimension"
    )
    lo_dim = lon_dim or _pick_name(
        ["longitude", "lon", "x"], dims, "longitude dimension"
    )

    if len({t_dim, c_dim, la_dim, lo_dim}) != 4:
        raise ValueError(
            "Time/channel/lat/lon dimensions must be distinct. "
            f"Got: time={t_dim}, channel={c_dim}, lat={la_dim}, lon={lo_dim}"
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

    if "latitude" not in da.coords or "longitude" not in da.coords:
        raise ValueError(
            "Prithvi variable must expose 1D latitude/longitude coordinates for interpolation."
        )

    lat = da["latitude"].values
    lon = da["longitude"].values
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Only 1D lat/lon coordinates are supported in this pipeline.")

    if not np.all(np.diff(lat) < 0):
        lat_idx = np.argsort(lat)[::-1]
        da = da.isel(latitude=lat_idx)
    if not np.all(np.diff(lon) > 0):
        lon_idx = np.argsort(lon)
        da = da.isel(longitude=lon_idx)

    return da


def load_prithvi_dataarray(
    source_path: Path,
    variable_name: Optional[str] = None,
    time_dim: Optional[str] = None,
    channel_dim: Optional[str] = None,
    lat_dim: Optional[str] = None,
    lon_dim: Optional[str] = None,
) -> xr.DataArray:
    """Open a single-file Prithvi source and return dims [time, channel, latitude, longitude]."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Prithvi source not found: {source_path}")
    if source_path.is_dir():
        raise ValueError(
            "load_prithvi_dataarray() expects a NetCDF/Zarr file. "
            "For tiled sources, use PrithviDynamicProvider directly."
        )

    if source_path.suffix.lower() == ".zarr":
        ds = xr.open_zarr(source_path)
    else:
        ds = xr.open_dataset(source_path)

    if variable_name is None:
        if len(ds.data_vars) != 1:
            raise ValueError(
                "Prithvi source has multiple variables. Please set variable_name explicitly."
            )
        variable_name = list(ds.data_vars)[0]

    if variable_name not in ds.data_vars:
        raise KeyError(
            f"Variable '{variable_name}' not found in {source_path}. "
            f"Available: {list(ds.data_vars)}"
        )

    return _standardize_dataarray(
        ds[variable_name],
        time_dim=time_dim,
        channel_dim=channel_dim,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
    )


def nearest_time_index(time_values: np.ndarray, target_time: datetime) -> tuple[int, int]:
    diffs = np.abs(time_values - np.datetime64(target_time))
    idx = int(np.argmin(diffs))
    diff_minutes = int(diffs[idx] / np.timedelta64(1, "m"))
    return idx, diff_minutes


class PrithviDynamicProvider:
    """Dynamic Prithvi feature provider with nearest-time matching + PCA projection."""

    def __init__(
        self,
        source_path: Path,
        pca_model_path: Path,
        *,
        variable_name: Optional[str] = None,
        max_time_diff_minutes: int = 24 * 60,
        time_dim: Optional[str] = None,
        channel_dim: Optional[str] = None,
        lat_dim: Optional[str] = None,
        lon_dim: Optional[str] = None,
        tile_cache_size: int = 128,
    ):
        self.source_path = Path(source_path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Prithvi source not found: {self.source_path}")

        self.variable_name = variable_name
        self.time_dim = time_dim
        self.channel_dim = channel_dim
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim
        self.max_time_diff_minutes = int(max_time_diff_minutes)
        self.tile_cache_size = max(int(tile_cache_size), 0)
        self._tile_cache: dict[tuple[str, int, int], np.ndarray] = {}
        self._source_mode = "directory" if self.source_path.is_dir() else "file"
        self._tile_sources = self._discover_tile_sources() if self._source_mode == "directory" else []
        self._loaded_tiles: dict[str, xr.DataArray] = {}

        if self._source_mode == "file":
            self.da = load_prithvi_dataarray(
                source_path=self.source_path,
                variable_name=variable_name,
                time_dim=time_dim,
                channel_dim=channel_dim,
                lat_dim=lat_dim,
                lon_dim=lon_dim,
            )
            self.time_values = self.da["time"].values
            raw_channels = int(self.da.sizes["channel"])
        else:
            if not self._tile_sources:
                raise FileNotFoundError(
                    f"No tile NetCDF files found under Prithvi source directory: {self.source_path}"
                )
            first_tile = self._load_tile_da(self._tile_sources[0]["path"])
            raw_channels = int(first_tile.sizes["channel"])
            self.time_values = np.array([], dtype="datetime64[ns]")

        pca_model_path = Path(pca_model_path)
        if not pca_model_path.exists():
            raise FileNotFoundError(f"Prithvi PCA model not found: {pca_model_path}")
        model = np.load(pca_model_path)

        self.raw_mean = np.asarray(model["raw_mean"], dtype=np.float32)
        self.raw_std = np.asarray(model["raw_std"], dtype=np.float32)
        self.components = np.asarray(model["components"], dtype=np.float32)
        self.score_std = np.asarray(model["score_std"], dtype=np.float32)

        if self.components.shape[1] != raw_channels:
            raise ValueError(
                "PCA model channel mismatch: "
                f"components expect raw_channels={self.components.shape[1]}, "
                f"but Prithvi source has channel={raw_channels}."
            )
        if self.raw_mean.shape[0] != raw_channels or self.raw_std.shape[0] != raw_channels:
            raise ValueError("PCA raw_mean/raw_std shape mismatch with raw channel count.")
        if self.score_std.shape[0] != self.components.shape[0]:
            raise ValueError("PCA score_std length must equal number of PCA components.")

        self.n_components = int(self.components.shape[0])

    def _discover_tile_sources(self) -> list[dict[str, object]]:
        sources: list[dict[str, object]] = []
        for path in sorted(self.source_path.glob("*.nc")):
            match = TILE_FILE_RE.match(path.name)
            if not match:
                continue
            lat_min = float(match.group("lat"))
            lon_min = float(match.group("lon"))
            sources.append(
                {
                    "path": path,
                    "lat_min": lat_min,
                    "lon_min": lon_min,
                }
            )
        return sources

    def _load_tile_da(self, path: Path) -> xr.DataArray:
        cache_key = str(path)
        cached = self._loaded_tiles.get(cache_key)
        if cached is not None:
            return cached

        ds = xr.open_dataset(path)
        if self.variable_name is None:
            if len(ds.data_vars) != 1:
                raise ValueError(
                    f"Tile source {path} has multiple variables. Set variable_name explicitly."
                )
            variable_name = list(ds.data_vars)[0]
        else:
            variable_name = self.variable_name

        if variable_name not in ds.data_vars:
            raise KeyError(
                f"Variable '{variable_name}' not found in {path}. "
                f"Available: {list(ds.data_vars)}"
            )

        da = _standardize_dataarray(
            ds[variable_name].load(),
            time_dim=self.time_dim,
            channel_dim=self.channel_dim,
            lat_dim=self.lat_dim,
            lon_dim=self.lon_dim,
        )
        self._loaded_tiles[cache_key] = da
        return da

    def _project_raw_to_pca(self, raw_stack: np.ndarray) -> np.ndarray:
        c_raw, h, w = raw_stack.shape
        flat = raw_stack.reshape(c_raw, -1).T

        valid = np.isfinite(flat).all(axis=1)
        projected = np.full((flat.shape[0], self.n_components), np.nan, dtype=np.float32)

        if np.any(valid):
            z = (flat[valid] - self.raw_mean[None, :]) / self.raw_std[None, :]
            scores = z @ self.components.T
            scores = scores / self.score_std[None, :]
            projected[valid] = scores.astype(np.float32)

        return projected.T.reshape(self.n_components, h, w)

    def _sample_from_da(
        self,
        *,
        da: xr.DataArray,
        time_idx: int,
        lats: np.ndarray,
        lons: np.ndarray,
        cache_name: str,
        tile_id: Optional[int] = None,
    ) -> np.ndarray:
        cache_key = None
        if tile_id is not None and self.tile_cache_size > 0:
            cache_key = (cache_name, int(time_idx), int(tile_id))
            hit = self._tile_cache.get(cache_key)
            if hit is not None:
                return hit.copy()

        raw = da.isel(time=int(time_idx))
        linear = raw.interp(
            latitude=xr.DataArray(lats, dims="latitude"),
            longitude=xr.DataArray(lons, dims="longitude"),
            method="linear",
        )
        nearest = raw.interp(
            latitude=xr.DataArray(lats, dims="latitude"),
            longitude=xr.DataArray(lons, dims="longitude"),
            method="nearest",
        )
        filled = linear.where(np.isfinite(linear), nearest)
        raw_stack = filled.values.astype(np.float32)

        if raw_stack.ndim != 3:
            raise ValueError(f"Expected raw_stack ndim=3, got shape={raw_stack.shape}")

        out = self._project_raw_to_pca(raw_stack)

        if cache_key is not None:
            if len(self._tile_cache) >= self.tile_cache_size:
                self._tile_cache.clear()
            self._tile_cache[cache_key] = out.copy()

        return out

    def _find_tile_source(self, lats: np.ndarray, lons: np.ndarray) -> Optional[Path]:
        lat_max = float(np.max(lats))
        lat_min = float(np.min(lats))
        lon_min = float(np.min(lons))
        lon_max = float(np.max(lons))
        tol = 1e-6

        for source in self._tile_sources:
            da = self._load_tile_da(source["path"])
            src_lat = da["latitude"].values
            src_lon = da["longitude"].values
            src_lat_max = float(np.max(src_lat))
            src_lat_min = float(np.min(src_lat))
            src_lon_min = float(np.min(src_lon))
            src_lon_max = float(np.max(src_lon))
            if (
                lat_max <= src_lat_max + tol
                and lat_min >= src_lat_min - tol
                and lon_min >= src_lon_min - tol
                and lon_max <= src_lon_max + tol
            ):
                return source["path"]
        return None

    def nearest_match(self, target_time: datetime) -> tuple[int, int]:
        if target_time.year < MIN_PRITHVI_YEAR or self._source_mode != "file":
            return -1, 0
        idx, diff_minutes = nearest_time_index(self.time_values, target_time)
        if diff_minutes > self.max_time_diff_minutes:
            return -1, diff_minutes
        return idx, diff_minutes

    def sample_tile(
        self,
        *,
        time_idx: int,
        lats: np.ndarray,
        lons: np.ndarray,
        tile_id: Optional[int] = None,
    ) -> np.ndarray:
        if self._source_mode != "file":
            raise ValueError("sample_tile() only supports single-file Prithvi sources.")
        return self._sample_from_da(
            da=self.da,
            time_idx=int(time_idx),
            lats=lats,
            lons=lons,
            cache_name=str(self.source_path),
            tile_id=tile_id,
        )

    def sample_tile_for_time(
        self,
        *,
        target_time: datetime,
        lats: np.ndarray,
        lons: np.ndarray,
        tile_id: Optional[int] = None,
    ) -> tuple[Optional[np.ndarray], int]:
        if target_time.year < MIN_PRITHVI_YEAR:
            return None, 0

        if self._source_mode == "file":
            time_idx, diff_minutes = self.nearest_match(target_time)
            if time_idx < 0:
                return None, diff_minutes
            sample = self.sample_tile(
                time_idx=int(time_idx),
                lats=lats,
                lons=lons,
                tile_id=tile_id,
            )
            return sample, diff_minutes

        tile_path = self._find_tile_source(lats=lats, lons=lons)
        if tile_path is None:
            return None, 0

        da = self._load_tile_da(tile_path)
        time_values = da["time"].values
        if time_values.size == 0:
            return None, 0

        time_idx, diff_minutes = nearest_time_index(time_values, target_time)
        if diff_minutes > self.max_time_diff_minutes:
            return None, diff_minutes

        sample = self._sample_from_da(
            da=da,
            time_idx=int(time_idx),
            lats=lats,
            lons=lons,
            cache_name=str(tile_path),
            tile_id=tile_id,
        )
        return sample, diff_minutes
