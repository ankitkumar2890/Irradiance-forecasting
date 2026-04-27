import json
import os
from pathlib import Path
from math import cos, radians

from dotenv import load_dotenv

# ---- Directories ----
BASE_DIR       = Path(__file__).resolve().parent
PROJECT_ROOT   = BASE_DIR.parent

# Auto-load .env from project root so API keys are always available
load_dotenv(PROJECT_ROOT / ".env")
DOWNLOADS_DIR  = BASE_DIR / "downloads"
MULTI_STATION_DOWNLOADS_DIR = DOWNLOADS_DIR / "multi_station"
PROCESSED_DOWNLOADS_DIR = DOWNLOADS_DIR / "processed"
DATASET_DIR    = BASE_DIR / "dataset"
ARROW_DIR      = DATASET_DIR / "arrow"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR    = BASE_DIR / "results"

# Per-station SRTM elevation cache (populated by 01_fetch_data.py). Applied as
# an override on top of the station list so each site gets its real ground
# elevation instead of the grid-center default.
STATION_ELEVATIONS_CACHE = DOWNLOADS_DIR / "station_elevations.json"

# Phase 1 synthetic ICON lives here:
PHASE1_DIR = PROJECT_ROOT / "phase1_cloud_mapper"
PHASE1_DOWNLOADS_DIR = PHASE1_DIR / "downloads"

for d in [
    DOWNLOADS_DIR,
    MULTI_STATION_DOWNLOADS_DIR,
    DATASET_DIR,
    ARROW_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ---- API Keys ----
NREL_API_KEY = os.environ.get("NREL_API_KEY", "")
NREL_EMAIL   = os.environ.get("NREL_EMAIL", "")
CDSAPI_KEY   = os.environ.get("CDSAPI_KEY", "")

# Reuse the Phase 1 ERA5 export when available.
ERA5_SOURCE_FILE = Path(
    os.environ.get(
        "FINETUNE_ERA5_SOURCE_FILE",
        str(PHASE1_DOWNLOADS_DIR / "era5_2017_2019.csv"),
    )
)

# ---- Station layout ----
# Default: a 3x3 local grid around Tirunelveli using adjacent 4 km cells.
# Set CLUSTER_LAYOUT="explicit" to fall back to a hand-picked station list.
CLUSTER_LAYOUT = "grid"
CLUSTER_NAME = "tirunelveli_3x3_grid_4km_spacing"

GRID_CENTER_LAT = 9.14
GRID_CENTER_LON = 77.92
GRID_CENTER_ALT_M = 45.0
BOX_SIZE_KM = 4.0
GRID_SPACING_KM = 4.0

EXPLICIT_STATIONS = [
    {"id": "tirunelveli", "lat": 9.14, "lon": 77.92, "alt_m": 45.0},
    {"id": "madurai",     "lat": 9.93, "lon": 78.12, "alt_m": 101.0},
    {"id": "coimbatore",  "lat": 11.02, "lon": 76.96, "alt_m": 411.0},
    {"id": "trichy",      "lat": 10.79, "lon": 78.70, "alt_m": 88.0},
    {"id": "chennai",     "lat": 13.08, "lon": 80.27, "alt_m": 7.0},
]


def _km_to_lat_deg(km: float) -> float:
    return km / 110.574


def _km_to_lon_deg(km: float, lat_deg: float) -> float:
    return km / (111.320 * cos(radians(lat_deg)))


def _build_grid_stations():
    stations = []
    spacing_km = GRID_SPACING_KM
    spacing_label = f"{int(spacing_km)}km" if float(spacing_km).is_integer() else f"{spacing_km:g}km"

    for row_offset in [1, 0, -1]:
        lat = GRID_CENTER_LAT + _km_to_lat_deg(row_offset * spacing_km)
        for col_offset in [-1, 0, 1]:
            lon = GRID_CENTER_LON + _km_to_lon_deg(col_offset * spacing_km, GRID_CENTER_LAT)
            station_id = (
                f"box{spacing_label}_r{row_offset:+d}_c{col_offset:+d}"
                .replace("+", "p")
                .replace("-", "m")
            )
            stations.append(
                {
                    "id": station_id,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "alt_m": GRID_CENTER_ALT_M,
                }
            )
    return stations


def _build_explicit_stations():
    return [
        {
            "id": str(station["id"]),
            "lat": round(float(station["lat"]), 6),
            "lon": round(float(station["lon"]), 6),
            "alt_m": float(station.get("alt_m", GRID_CENTER_ALT_M)),
        }
        for station in EXPLICIT_STATIONS
    ]


def _build_stations():
    if CLUSTER_LAYOUT == "grid":
        return _build_grid_stations()
    if CLUSTER_LAYOUT == "explicit":
        stations = _build_explicit_stations()
        if not stations:
            raise ValueError("CLUSTER_LAYOUT='explicit' requires at least one EXPLICIT_STATIONS entry.")
        return stations
    raise ValueError(f"Unsupported CLUSTER_LAYOUT={CLUSTER_LAYOUT!r}")


def _load_station_elevation_overrides() -> dict[str, float]:
    """Read cached per-station SRTM elevations produced by 01_fetch_data.py."""
    if not STATION_ELEVATIONS_CACHE.exists():
        return {}
    try:
        with open(STATION_ELEVATIONS_CACHE, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    overrides: dict[str, float] = {}
    for key, value in raw.items():
        try:
            overrides[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return overrides


def _apply_elevation_overrides(stations: list[dict]) -> list[dict]:
    overrides = _load_station_elevation_overrides()
    if not overrides:
        return stations
    return [
        {**station, "alt_m": overrides.get(station["id"], station["alt_m"])}
        for station in stations
    ]


STATIONS = _apply_elevation_overrides(_build_stations())
STATION_IDS = [station["id"] for station in STATIONS]
ELEVATION_MAP = {station["id"]: station["alt_m"] for station in STATIONS}

# For Moirai fine-tuning, use the center grid box by default.
FINETUNE_STATION = "box4km_rp0_cp0" if CLUSTER_LAYOUT == "grid" else STATIONS[0]["id"]
SELECTED_STATION = next(s for s in STATIONS if s["id"] == FINETUNE_STATION)

# ---- Data years ----
YEARS = [2017, 2018, 2019]

# Temporal split across 3 years:
#   Train: Jan 2017 – Dec 2018
#   Val:   Jan 2019 – Dec 2019
#   Test:  reserved / empty unless later data is added
TRAIN_END   = "2018-12-31 23:00"
VAL_START   = "2019-01-01 00:00"
VAL_END     = "2019-12-31 23:00"
TEST_START  = "2020-01-01 00:00"

# ---- Sliding window params ----
PAST_HOURS   = 72
FUTURE_HOURS = 24

# ---- ERA5 feature mapping ----
ERA5_CANONICAL_COLUMNS = {
    "tcc": ("tcc", "total_cloud_cover"),
    "lcc": ("lcc", "low_cloud_cover"),
    "mcc": ("mcc", "medium_cloud_cover"),
    "hcc": ("hcc", "high_cloud_cover"),
    "u10": ("u10", "10m_u_component_of_wind", "u_component_of_wind_10m"),
    "v10": ("v10", "10m_v_component_of_wind", "v_component_of_wind_10m"),
}
ERA5_FEATURE_COLUMNS = list(ERA5_CANONICAL_COLUMNS.keys())

# Feature columns
PAST_FEATURES = [
    "CAF",
    "clear_sky_ghi",
    "tcc",
    "lcc",
    "mcc",
    "hcc",
    "wind_speed",
    "wind_direction",
    "zenith_angle",
    "azimuth_angle",
    "elevation_m",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]
FUTURE_FEATURES = [
    "clear_sky_ghi",
    "tcc",
    "lcc",
    "mcc",
    "hcc",
    "wind_speed",
    "wind_direction",
    "zenith_angle",
    "azimuth_angle",
    "elevation_m",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

# ---- Model ----
MODEL_ID           = "Salesforce/moirai-1.1-R-large"
CONTEXT_LENGTH     = PAST_HOURS
PREDICTION_LENGTH  = FUTURE_HOURS
TARGET_DIM         = 1
FEAT_DIM           = len(FUTURE_FEATURES)

# ---- LoRA Fine-Tuning ----
LORA_RANK          = 16
LORA_ALPHA         = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT       = 0.05

FT_LR              = 1e-4
FT_WEIGHT_DECAY    = 0.01
FT_MAX_EPOCHS      = 50
FT_PATIENCE        = 5
FT_BATCH_SIZE      = 32
FT_GRADIENT_CLIP   = 1.0
