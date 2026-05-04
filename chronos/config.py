"""Standalone configuration for the local Chronos direct-GHI workflow."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DOWNLOADS_DIR = BASE_DIR / "downloads"
MULTI_STATION_DOWNLOADS_DIR = DOWNLOADS_DIR / "multi_station"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

for directory in [DATASET_DIR, DOWNLOADS_DIR, MULTI_STATION_DOWNLOADS_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

PREDICTION_LENGTH = 24
GHI_SCALE_FACTOR = 1000.0
CLUSTER_NAME = "chronos_direct_ghi"


def infer_station_ids() -> list[str]:
    if MULTI_STATION_DOWNLOADS_DIR.exists():
        station_ids = sorted(
            path.name
            for path in MULTI_STATION_DOWNLOADS_DIR.iterdir()
            if path.is_dir()
        )
        if station_ids:
            return station_ids

    station_path = DATASET_DIR / "station_ids_val.npy"
    if station_path.exists():
        import numpy as np

        station_ids = np.load(station_path, allow_pickle=True)
        return sorted({str(station_id) for station_id in station_ids})

    return []


def infer_years() -> list[int]:
    years: set[int] = set()
    for station_id in infer_station_ids():
        station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
        if not station_dir.exists():
            continue
        for path in station_dir.glob("ghi_*.csv"):
            stem = path.stem
            try:
                years.add(int(stem.split("_")[-1]))
            except ValueError:
                continue
    return sorted(years) if years else [2017, 2018, 2019]


STATIONS = [{"id": station_id} for station_id in infer_station_ids()]
YEARS = infer_years()
