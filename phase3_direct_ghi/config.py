"""Phase 3: Direct GHI Forecasting — configuration.

Key difference from Phase 2: the target variable is GHI (W/m²) directly,
not the Cloud Attenuation Factor (CAF). Features include solar geometry,
clear-sky irradiance, cloud-transition signals, richer near-surface weather,
and simple neighbor-context summaries from the 3x3 grid.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
PROJECT_ROOT   = BASE_DIR.parent

# Auto-load .env from project root so API keys are always available
load_dotenv(PROJECT_ROOT / ".env")

DOWNLOADS_DIR              = BASE_DIR / "downloads"
MULTI_STATION_DOWNLOADS_DIR = DOWNLOADS_DIR / "multi_station"
PROCESSED_DOWNLOADS_DIR    = DOWNLOADS_DIR / "processed"
DATASET_DIR                = BASE_DIR / "dataset"
ARROW_DIR                  = DATASET_DIR / "arrow"
CHECKPOINT_DIR             = BASE_DIR / "checkpoints"
RESULTS_DIR                = BASE_DIR / "results"

# Phase 1 raw ERA5 source (fallback)
PHASE1_DIR = PROJECT_ROOT / "phase1_cloud_mapper"
PHASE1_DOWNLOADS_DIR = PHASE1_DIR / "downloads"

# Phase 2 ERA5 source (richer fallback — already has cloud + wind fields)
PHASE2_DIR = PROJECT_ROOT / "phase2_finetuning"
PHASE2_DOWNLOADS_DIR = PHASE2_DIR / "downloads" / "multi_station"

for d in [
    DOWNLOADS_DIR,
    MULTI_STATION_DOWNLOADS_DIR,
    PROCESSED_DOWNLOADS_DIR,
    DATASET_DIR,
    ARROW_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ─────────────────────────────────────────────────────────────────
NREL_API_KEY = os.environ.get("NREL_API_KEY", "")
NREL_EMAIL   = os.environ.get("NREL_EMAIL", "")
CDSAPI_KEY   = os.environ.get("CDSAPI_KEY", "")

# ── ERA5 source resolution ───────────────────────────────────────────────────
ERA5_SOURCE_FILE = Path(
    os.environ.get(
        "DIRECT_GHI_ERA5_SOURCE_FILE",
        str(PHASE1_DOWNLOADS_DIR / "era5_2017_2019.csv"),
    )
)

# ── Cluster Definition ────────────────────────────────────────────────────────
# Two supported layouts:
# - "grid": build a 3x3 grid around the center with wider center-to-center
#   spacing to reduce near-duplicate neighboring series.
# - "explicit": use a hand-picked list of stations that can intentionally span
#   more diverse terrain or cloud regimes.
#
# Default back to the original 5-station South Tamil Nadu cluster.
CLUSTER_LAYOUT = "explicit"
CLUSTER_NAME = "south_tamilnadu"

GRID_CENTER_LAT = 9.14
GRID_CENTER_LON = 77.92
GRID_CENTER_ALT_M = 45.0
BOX_SIZE_KM = 4.0
GRID_SPACING_KM = 20.0
GRID_NEIGHBOR_COUNT = 8

# Original 5-station cluster used by the phase 3 direct GHI pipeline. Keep IDs
# stable once data has been downloaded so cached files continue to match the
# station metadata.
EXPLICIT_STATIONS: list[dict] = [
    {
        "id": "tirunelveli",
        "lat": 9.14,
        "lon": 77.92,
        "alt_m": 45.0,
    },
    {
        "id": "madurai",
        "lat": 9.93,
        "lon": 78.12,
        "alt_m": 101.0,
    },
    {
        "id": "coimbatore",
        "lat": 11.02,
        "lon": 76.96,
        "alt_m": 411.0,
    },
    {
        "id": "trichy",
        "lat": 10.79,
        "lon": 78.70,
        "alt_m": 88.0,
    },
    {
        "id": "chennai",
        "lat": 13.08,
        "lon": 80.27,
        "alt_m": 7.0,
    },
]


def _km_to_lat_deg(km: float) -> float:
    return km / 110.574


def _km_to_lon_deg(km: float, lat_deg: float) -> float:
    from math import cos, radians

    return km / (111.320 * cos(radians(lat_deg)))


def _build_grid_stations() -> list[dict]:
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
                    # Until per-box elevation is fetched, reuse the known center elevation.
                    "alt_m": GRID_CENTER_ALT_M,
                }
            )
    return stations


def _build_explicit_stations() -> list[dict]:
    stations = []
    for station in EXPLICIT_STATIONS:
        stations.append(
            {
                "id": str(station["id"]),
                "lat": round(float(station["lat"]), 6),
                "lon": round(float(station["lon"]), 6),
                "alt_m": float(station.get("alt_m", GRID_CENTER_ALT_M)),
            }
        )
    return stations


def _build_stations() -> list[dict]:
    if CLUSTER_LAYOUT == "grid":
        return _build_grid_stations()
    if CLUSTER_LAYOUT == "explicit":
        stations = _build_explicit_stations()
        if not stations:
            raise ValueError("CLUSTER_LAYOUT='explicit' requires at least one entry in EXPLICIT_STATIONS.")
        return stations
    raise ValueError(f"Unsupported CLUSTER_LAYOUT={CLUSTER_LAYOUT!r}")


STATIONS = _build_stations()

# Elevation lookup for the static spatial feature
ELEVATION_MAP = {s["id"]: s["alt_m"] for s in STATIONS}

# ── Data years ───────────────────────────────────────────────────────────────
YEARS = [2017, 2018, 2019]

# Temporal split across 3 years:
#   Train: Jan 2017 – Dec 2018
#   Val:   Jan 2019 – Dec 2019
#   Test:  reserved / empty unless later data is added
TRAIN_END   = "2018-12-31 23:00"
VAL_START   = "2019-01-01 00:00"
VAL_END     = "2019-12-31 23:00"
TEST_START  = "2020-01-01 00:00"

# ── Sliding window params ────────────────────────────────────────────────────
PAST_HOURS   = 72
FUTURE_HOURS = 24

# ── ERA5 feature mapping ────────────────────────────────────────────────────
ERA5_CANONICAL_COLUMNS = {
    "tcc": ("tcc", "total_cloud_cover"),
    "lcc": ("lcc", "low_cloud_cover"),
    "mcc": ("mcc", "medium_cloud_cover"),
    "hcc": ("hcc", "high_cloud_cover"),
    "u10": ("u10", "10m_u_component_of_wind", "u_component_of_wind_10m"),
    "v10": ("v10", "10m_v_component_of_wind", "v_component_of_wind_10m"),
}
ERA5_FEATURE_COLUMNS = list(ERA5_CANONICAL_COLUMNS.keys())

# ── Feature Columns for Direct GHI ───────────────────────────────────────────
# Past context (encoder): includes historical measured GHI plus transition-aware
# weather features. clear_sky_index is past-only to avoid target leakage.
PAST_FEATURES = [
    "w_ghr",            # Autoregressive: historical measured GHI
    "zenith_angle",
    "azimuth_angle",    # Solar azimuth
    "clearsky_ghi",
    "clear_sky_index",
    "tcc", "lcc", "mcc", "hcc",
    "tcc_delta_1h", "lcc_delta_1h", "mcc_delta_1h", "hcc_delta_1h",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "temperature_2m_delta_1h",
    "relative_humidity_2m_delta_1h",
    "surface_pressure_delta_1h",
    "wind_speed",       # Derived from u10/v10
    "wind_direction",   # Derived from u10/v10
    "wind_gusts_10m",
    "wind_speed_delta_1h",
    "neighbor_tcc_mean",
    "neighbor_temperature_2m_mean",
    "neighbor_relative_humidity_2m_mean",
    "neighbor_wind_speed_mean",
    "elevation_m",      # Static spatial feature
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
]

# Future context (decoder known covariates): same minus measured GHI and any
# target-derived quantities.
FUTURE_FEATURES = [
    "zenith_angle",
    "azimuth_angle",
    "clearsky_ghi",
    "tcc", "lcc", "mcc", "hcc",
    "tcc_delta_1h", "lcc_delta_1h", "mcc_delta_1h", "hcc_delta_1h",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "temperature_2m_delta_1h",
    "relative_humidity_2m_delta_1h",
    "surface_pressure_delta_1h",
    "wind_speed",
    "wind_direction",
    "wind_gusts_10m",
    "wind_speed_delta_1h",
    "neighbor_tcc_mean",
    "neighbor_temperature_2m_mean",
    "neighbor_relative_humidity_2m_mean",
    "neighbor_wind_speed_mean",
    "elevation_m",
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
]

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID           = "Salesforce/moirai-2.0-R-small"
# Per-variant overrides (consumed by moirai/moirai.py::resolve_model_id).
MODEL_ID_MOIRAI1   = "Salesforce/moirai-1.1-R-base"
MODEL_ID_MOIRAI2   = MODEL_ID
CONTEXT_LENGTH     = PAST_HOURS
PREDICTION_LENGTH  = FUTURE_HOURS
TARGET_DIM         = 1          # Direct GHI (single univariate target)
FEAT_DIM           = len(FUTURE_FEATURES)

# ── Window selection ────────────────────────────────────────────────────────
ANCHOR_HOURS       = [6]
MIN_PAST_DATES     = 1

# ── Eval column hints (consumed by moirai/functions/results.py) ──────────────
TARGET_COL         = "w_ghr"
CLEARSKY_GHI_COL   = "clearsky_ghi"
MEASURED_GHI_COL   = "w_ghr"

# ── LoRA Fine-Tuning ─────────────────────────────────────────────────────────
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

# ── GHI normalisation (for training stability) ──────────────────────────────
# Divide GHI values by this constant before feeding to the model, and multiply
# predictions back.  This keeps the target in a similar range to CAF ∈ [0,1].
GHI_SCALE_FACTOR = 1000.0

# ── Daylight-aware output masking ────────────────────────────────────────────
# Full output is allowed below the daylight zenith threshold. A linear taper
# suppresses predictions as the sun approaches/descends below the horizon.
DAYLIGHT_ZENITH_DEG = 85.0
NIGHT_ZENITH_DEG = 90.0
