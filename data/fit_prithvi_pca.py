"""
Fit PCA for dynamic Prithvi embeddings (train years only) and save projection model.

The saved model is consumed by data/fno_dataset.py through PrithviDynamicProvider.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from data.prithvi_utils import TILE_FILE_RE, load_prithvi_dataarray
except ImportError:
    from prithvi_utils import TILE_FILE_RE, load_prithvi_dataarray


def parse_year_spec(spec: str) -> list[int]:
    years = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            lo_i, hi_i = int(lo), int(hi)
            if hi_i < lo_i:
                raise ValueError(f"Invalid year range: {token}")
            years.update(range(lo_i, hi_i + 1))
        else:
            years.add(int(token))
    if not years:
        raise ValueError("No years parsed from --train-years")
    return sorted(years)


def years_from_time_array(time_values: np.ndarray) -> np.ndarray:
    # datetime64[Y] is years since 1970
    return time_values.astype("datetime64[Y]").astype(np.int32) + 1970


def fit_pca(z: np.ndarray, n_components: int):
    n_samples, n_features = z.shape
    if n_components > n_features:
        raise ValueError(
            f"n_components={n_components} exceeds raw feature count={n_features}"
        )
    if n_samples < n_components:
        raise ValueError(
            f"Need at least n_components samples. Got samples={n_samples}, "
            f"n_components={n_components}."
        )

    # Z is already standardized.
    _, svals, vt = np.linalg.svd(z, full_matrices=False)
    components = vt[:n_components].astype(np.float32)

    eigvals = (svals ** 2) / max(n_samples - 1, 1)
    total_var = float(np.sum(eigvals) + 1e-12)
    explained = (eigvals / total_var).astype(np.float32)
    explained_k = explained[:n_components]

    scores = z @ components.T
    score_std = np.std(scores, axis=0).astype(np.float32) + 1e-6
    return components, explained_k, explained, score_std


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit PCA model for Prithvi embeddings.")
    p.add_argument("--source", type=str, required=True, help="Path to source NetCDF/Zarr.")
    p.add_argument(
        "--output-model",
        type=str,
        required=True,
        help="Output .npz path for PCA parameters.",
    )
    p.add_argument(
        "--variable",
        type=str,
        default=None,
        help="Data variable name inside source (required if multiple vars).",
    )
    p.add_argument("--time-dim", type=str, default=None)
    p.add_argument("--channel-dim", type=str, default=None)
    p.add_argument("--lat-dim", type=str, default=None)
    p.add_argument("--lon-dim", type=str, default=None)
    p.add_argument("--train-years", type=str, default="2013-2017")
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--max-total-samples", type=int, default=300000)
    p.add_argument("--max-samples-per-time", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p


def iter_tiled_sources(source_dir: Path) -> list[Path]:
    files = []
    for path in sorted(source_dir.glob("*.nc")):
        if TILE_FILE_RE.match(path.name):
            files.append(path)
    if not files:
        raise RuntimeError(f"No Prithvi tile NetCDF files found in: {source_dir}")
    return files


def main():
    args = build_parser().parse_args()

    source = Path(args.source)
    out_model = Path(args.output_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    train_years = parse_year_spec(args.train_years)

    rng = np.random.default_rng(args.seed)
    max_total = max(int(args.max_total_samples), 1)
    max_per_time = max(int(args.max_samples_per_time), 1)

    rows = []
    gathered = 0
    raw_channels = None

    if source.is_dir():
        print(f"Loaded tiled source directory: {source}")
        tile_files = iter_tiled_sources(source)
        rng.shuffle(tile_files)
        available_years = set()

        for tile_path in tile_files:
            da = load_prithvi_dataarray(
                source_path=tile_path,
                variable_name=args.variable,
                time_dim=args.time_dim,
                channel_dim=args.channel_dim,
                lat_dim=args.lat_dim,
                lon_dim=args.lon_dim,
            )
            if raw_channels is None:
                raw_channels = int(da.sizes["channel"])

            years = years_from_time_array(da["time"].values)
            available_years.update(years.tolist())
            train_idx = np.where(np.isin(years, train_years))[0]
            if train_idx.size == 0:
                continue

            shuffled = np.array(train_idx, copy=True)
            rng.shuffle(shuffled)
            for t_idx in shuffled:
                arr = da.isel(time=int(t_idx)).values.astype(np.float32)
                flat = arr.reshape(raw_channels, -1).T
                valid = np.isfinite(flat).all(axis=1)
                valid_idx = np.where(valid)[0]
                if valid_idx.size == 0:
                    continue

                n_take = min(max_per_time, valid_idx.size, max_total - gathered)
                chosen = rng.choice(valid_idx, size=n_take, replace=False)
                rows.append(flat[chosen])
                gathered += n_take
                if gathered >= max_total:
                    break
            if gathered >= max_total:
                break

        if not rows:
            if available_years:
                raise RuntimeError(
                    f"No Prithvi timesteps found for train years {train_years}. "
                    f"Available year range: {min(available_years)}..{max(available_years)}"
                )
            raise RuntimeError("No valid tile files were available for PCA fitting.")
    else:
        da = load_prithvi_dataarray(
            source_path=source,
            variable_name=args.variable,
            time_dim=args.time_dim,
            channel_dim=args.channel_dim,
            lat_dim=args.lat_dim,
            lon_dim=args.lon_dim,
        )
        print(f"Loaded source: {source}")
        print(f"Data shape [time,channel,lat,lon]: {tuple(da.shape)}")

        years = years_from_time_array(da["time"].values)
        train_idx = np.where(np.isin(years, train_years))[0]
        if train_idx.size == 0:
            raise RuntimeError(
                f"No Prithvi timesteps found for train years {train_years}. "
                f"Available year range: {years.min()}..{years.max()}"
            )

        shuffled = np.array(train_idx, copy=True)
        rng.shuffle(shuffled)
        raw_channels = int(da.sizes["channel"])
        for t_idx in shuffled:
            arr = da.isel(time=int(t_idx)).values.astype(np.float32)
            flat = arr.reshape(raw_channels, -1).T
            valid = np.isfinite(flat).all(axis=1)
            valid_idx = np.where(valid)[0]
            if valid_idx.size == 0:
                continue

            n_take = min(max_per_time, valid_idx.size, max_total - gathered)
            chosen = rng.choice(valid_idx, size=n_take, replace=False)
            rows.append(flat[chosen])
            gathered += n_take
            if gathered >= max_total:
                break

    if not rows:
        raise RuntimeError("No valid pixels available to fit PCA.")

    x = np.concatenate(rows, axis=0).astype(np.float32)
    print(f"PCA fit samples: {x.shape[0]} pixels, raw channels: {x.shape[1]}")

    raw_mean = np.mean(x, axis=0).astype(np.float32)
    raw_std = np.std(x, axis=0).astype(np.float32) + 1e-6
    z = (x - raw_mean[None, :]) / raw_std[None, :]

    components, explained_k, explained_all, score_std = fit_pca(
        z, n_components=int(args.n_components)
    )
    print(
        f"Explained variance (first {args.n_components}): "
        f"{float(np.sum(explained_k)):.4f}"
    )

    np.savez_compressed(
        out_model,
        raw_mean=raw_mean,
        raw_std=raw_std,
        components=components,
        score_std=score_std,
        explained_variance_ratio=explained_k,
        explained_variance_ratio_all=explained_all,
        train_years=np.asarray(train_years, dtype=np.int16),
    )
    print(f"Saved PCA model: {out_model}")


if __name__ == "__main__":
    main()
