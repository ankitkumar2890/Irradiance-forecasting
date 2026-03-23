"""
Check year coverage (>=2010) for OCF ICON archives.

Default behavior:
    - ICON Global: openclimatefix/dwd-icon-global (Hugging Face dataset repo)
    - ICON EU    : openclimatefix/dwd-icon-eu (Hugging Face dataset repo)

You can also pass direct Zarr paths (e.g. gs://..., s3://..., local .zarr).
"""

import argparse
from datetime import datetime, timezone
import re

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

TIME_CANDIDATES = [
    "init_time",
    "time",
    "forecast_reference_time",
    "valid_time",
]

HF_DEFAULT_GLOBAL = "openclimatefix/dwd-icon-global"
HF_DEFAULT_EU = "openclimatefix/dwd-icon-eu"
YEAR_RE = re.compile(r"(20\d{2})")
VAR_CANDIDATES = {
    "qc": ["qc", "clwmr", "qcl"],
    "qi": ["qi", "ciwmr", "qci"],
    "cloud_cover": ["clct", "clc", "tcdc"],
}


def open_zarr(url: str) -> xr.Dataset:
    mapper = fsspec.get_mapper(url)
    return xr.open_zarr(mapper, consolidated=False)


def pick_time_coord(ds: xr.Dataset):
    for c in TIME_CANDIDATES:
        if c in ds.coords:
            return c
    for c in ds.coords:
        if "time" in c.lower():
            return c
    raise ValueError(f"No time-like coordinate found. Coords: {list(ds.coords)}")


def to_datetime_index(values) -> pd.DatetimeIndex:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        dt = pd.to_datetime(arr)
    else:
        dt = pd.to_datetime(arr.astype(str), errors="coerce")
    dt = dt[~pd.isna(dt)]
    if len(dt) == 0:
        raise ValueError("Could not parse any datetime values from time coordinate.")
    return pd.DatetimeIndex(dt)


def summarize_years(name: str, years: np.ndarray, source_desc: str):
    print(f"\n=== {name} ===")
    print(f"Source: {source_desc}")

    if years.size == 0:
        print("No years found.")
        return

    years_2010 = years[years >= 2010]
    print(f"All years found: {years.tolist()}")
    print(f"Years >= 2010  : {years_2010.tolist()}")
    print(f"Count (>=2010) : {len(years_2010)}")

    current_year = datetime.now(timezone.utc).year
    expected = set(range(2010, current_year + 1))
    missing = sorted(expected - set(years_2010))
    print(f"Missing years from 2010..{current_year}: {missing}")


def print_var_report(var_hits: dict[str, list[str]], mode_label: str):
    print(f"Variable check ({mode_label}):")
    for target, matches in var_hits.items():
        if matches:
            print(f"  {target}: FOUND -> {matches}")
        else:
            print(f"  {target}: NOT FOUND")


def summarize_zarr(name: str, url: str):
    ds = open_zarr(url)
    tname = pick_time_coord(ds)
    dt = to_datetime_index(ds[tname].values)
    years = np.array(sorted(set(dt.year)))
    data_vars = {v.lower(): v for v in ds.data_vars}

    print(f"\n=== {name} ===")
    print(f"Source: {url}")
    print(f"Time coord used: {tname}")
    print(f"First timestamp: {dt.min()}")
    print(f"Last timestamp : {dt.max()}")

    years_2010 = years[years >= 2010]
    print(f"All years found: {years.tolist()}")
    print(f"Years >= 2010  : {years_2010.tolist()}")
    print(f"Count (>=2010) : {len(years_2010)}")

    current_year = datetime.now(timezone.utc).year
    expected = set(range(2010, current_year + 1))
    missing = sorted(expected - set(years_2010))
    print(f"Missing years from 2010..{current_year}: {missing}")

    var_hits = {}
    for target, cands in VAR_CANDIDATES.items():
        var_hits[target] = [data_vars[c] for c in cands if c in data_vars]
    print_var_report(var_hits, "exact from dataset variables")


def list_hf_dataset_files(repo_id: str) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for dataset-repo mode. "
            "Install with: pip install huggingface_hub"
        ) from e

    api = HfApi()
    return api.list_repo_files(repo_id=repo_id, repo_type="dataset")


def extract_years_from_paths(paths: list[str]) -> np.ndarray:
    years = set()
    for p in paths:
        for match in YEAR_RE.findall(p):
            y = int(match)
            if 2000 <= y <= 2100:
                years.add(y)
    return np.array(sorted(years), dtype=int)


def summarize_hf_repo(name: str, repo_id: str):
    files = list_hf_dataset_files(repo_id)
    years = extract_years_from_paths(files)
    summarize_years(name=name, years=years, source_desc=f"Hugging Face dataset repo: {repo_id}")
    # Repo listing mode: variable check is heuristic based on file/path names.
    files_l = [p.lower() for p in files]
    var_hits = {}
    for target, cands in VAR_CANDIDATES.items():
        found = sorted({cand for cand in cands if any(cand in p for p in files_l)})
        var_hits[target] = found
    print_var_report(var_hits, "heuristic from repo file names")


def route_source(name: str, src: str):
    # Heuristic: treat likely object-store/local paths as Zarr sources, otherwise HF dataset repo.
    if src.startswith(("gs://", "s3://", "http://", "https://", "/", "./", "../")) or src.endswith(".zarr"):
        summarize_zarr(name, src)
    else:
        summarize_hf_repo(name, src)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--icon-global",
        default=HF_DEFAULT_GLOBAL,
        help=(
            "ICON Global source. Either Hugging Face dataset repo id "
            f"(default: {HF_DEFAULT_GLOBAL}) or direct Zarr path."
        ),
    )
    parser.add_argument(
        "--icon-eu",
        default=HF_DEFAULT_EU,
        help=(
            "ICON EU source. Either Hugging Face dataset repo id "
            f"(default: {HF_DEFAULT_EU}) or direct Zarr path."
        ),
    )
    args = parser.parse_args()

    route_source("OCF ICON-Global", args.icon_global)
    route_source("OCF ICON-EU", args.icon_eu)


if __name__ == "__main__":
    main()
