"""Method 1 config: CAF target, PVLib clear-sky, single station, ERA5 covariates.

Originally lived in ``phase2_era5_direct/config.py``. Only the constants
consumed by the ``moirai/`` pipeline are kept here - everything related
to API keys, downloads, and CSV preparation belongs in the phase folder
that produces ``moirai/final_csv/method1/processed_data_2017_2019.csv``.
"""
from __future__ import annotations

# ---- Item id (single-station) ------------------------------------------
# Matches the station the prepared CSV in moirai/final_csv/method1/ was
# produced for. Used as ``item_id`` in the GluonTS dataset.
FINETUNE_STATION = "chennai"

# ---- Temporal split ----------------------------------------------------
TRAIN_END  = "2018-12-31 23:00"
VAL_START  = "2019-01-01 00:00"
VAL_END    = "2019-12-31 23:00"
TEST_START = "2020-01-01 00:00"

# ---- Sliding window ----------------------------------------------------
PAST_HOURS   = 72
FUTURE_HOURS = 24

# ---- Features ----------------------------------------------------------
# Past context (encoder): includes the autoregressive target (CAF).
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
# Future covariates (decoder): same minus the target.
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

# ---- Window selection --------------------------------------------------
# ANCHOR_HOURS is the list of allowed first-forecast-hours (local clock).
# None  -> keep every hour (one window per timestamp).
# [6]   -> "one forecast per day at 06:00".
# [0,6,12,18] -> four forecasts per day.
ANCHOR_HOURS   = None
MIN_PAST_DATES = 1

# ---- Model -------------------------------------------------------------
# Default backbone for Method 1.
MODEL_ID         = "Salesforce/moirai-1.1-R-base"
# Per-variant overrides used by ``moirai/moirai.py::resolve_model_id``.
# When you pass ``--variant moirai2`` the CLI uses ``MODEL_ID_MOIRAI2``
# instead of ``MODEL_ID`` (and vice versa) so we never load the wrong
# checkpoint into the wrong loader.
MODEL_ID_MOIRAI1 = MODEL_ID
MODEL_ID_MOIRAI2 = "Salesforce/moirai-2.0-R-small"

CONTEXT_LENGTH    = PAST_HOURS
PREDICTION_LENGTH = FUTURE_HOURS
TARGET_DIM        = 1
FEAT_DIM          = len(FUTURE_FEATURES)

# ---- LoRA fine-tuning --------------------------------------------------
LORA_RANK           = 32
LORA_ALPHA          = 64
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT        = 0.02

FT_LR             = 5e-5
FT_WEIGHT_DECAY   = 0.01
FT_MAX_EPOCHS     = 80
FT_PATIENCE       = 8
FT_BATCH_SIZE     = 32
FT_GRADIENT_CLIP  = 1.0
