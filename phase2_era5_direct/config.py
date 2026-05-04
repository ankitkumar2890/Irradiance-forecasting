import os
from pathlib import Path

# ---- Directories ----
BASE_DIR       = Path(__file__).resolve().parent
PROJECT_ROOT   = BASE_DIR.parent
DOWNLOADS_DIR  = BASE_DIR / "downloads"
DATASET_DIR    = BASE_DIR / "dataset"
ARROW_DIR      = DATASET_DIR / "arrow"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR    = BASE_DIR / "results"

# Phase 1 raw ERA5 source lives here:
PHASE1_DIR = PROJECT_ROOT / "phase1_cloud_mapper"
PHASE1_DOWNLOADS_DIR = PHASE1_DIR / "downloads"
ERA5_SOURCE_FILE = PHASE1_DOWNLOADS_DIR / "era5_2017_2019.csv"

for d in [DOWNLOADS_DIR, DATASET_DIR, ARROW_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- API Keys ----
NREL_API_KEY = os.environ.get("NREL_API_KEY", "")
NREL_EMAIL   = os.environ.get("NREL_EMAIL", "")

# ---- Sites (same 5 stations as Phase 1 CloudMapper) ----
STATIONS = [
    {"id": "tirunelveli", "lat": 9.14, "lon": 77.92, "alt_m": 45.0},
    {"id": "madurai",     "lat": 9.93, "lon": 78.12, "alt_m": 101.0},
    {"id": "coimbatore",  "lat": 11.02, "lon": 76.96, "alt_m": 411.0},
    {"id": "trichy",      "lat": 10.79, "lon": 78.70, "alt_m": 88.0},
    {"id": "chennai",     "lat": 13.08, "lon": 80.27, "alt_m": 7.0},
]
# For Moirai fine-tuning, we use a single station's time series
FINETUNE_STATION = "chennai"
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

# Feature columns for the direct-ERA5 experiment
PAST_FEATURES = [
    "CAF",
    "clear_sky_ghi",
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "cloud_liquid_water",
    "cloud_ice_water",
    "water_vapour",
    "zenith_angle",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]
FUTURE_FEATURES = [
    "clear_sky_ghi",
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "cloud_liquid_water",
    "cloud_ice_water",
    "water_vapour",
    "zenith_angle",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

# ---- Model ----
MODEL_ID           = "Salesforce/moirai-1.1-R-base"
# Per-variant overrides (consumed by moirai/moirai.py::resolve_model_id).
# Set MODEL_ID_MOIRAI2 if you intend to ever pass --variant moirai2 for this method.
MODEL_ID_MOIRAI1   = MODEL_ID
MODEL_ID_MOIRAI2   = "Salesforce/moirai-2.0-R-small"
CONTEXT_LENGTH     = PAST_HOURS
PREDICTION_LENGTH  = FUTURE_HOURS
TARGET_DIM         = 1
FEAT_DIM           = len(FUTURE_FEATURES)

# ---- Window selection ----
# ANCHOR_HOURS is the list of allowed first-forecast-hours (local clock).
# None  -> keep every hour (one window per timestamp).
# [6]   -> legacy "one forecast per day at 06:00" behaviour.
# [0,6,12,18] -> four forecasts per day.
ANCHOR_HOURS       = [6]
MIN_PAST_DATES     = 1

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
