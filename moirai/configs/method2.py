"""Method 2 config: CAF target, NSRDB clear-sky, multi-station 3x3 grid.

Originally lived in ``phase2_finetuning/config.py``. Only the constants
consumed by the ``moirai/`` pipeline are kept here - everything related
to API keys, station-grid construction, and CSV preparation belongs in
the phase folder that produces
``moirai/final_csv/method2/processed_data_2017_2019.csv``.
"""
from __future__ import annotations

# ---- Item id (multi-station; default to centre cell) -------------------
# Method 2 trains on all 9 grid stations (the dataset builder iterates
# over them), but ``cmd_dataset`` still passes a single ``item_id``.
# The centre of the 3x3 4 km grid is ``box4km_rp0_cp0``.
FINETUNE_STATION = "box4km_rp0_cp0"

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
# Future covariates (decoder): same minus the target.
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

# ---- Window selection --------------------------------------------------
ANCHOR_HOURS   = [6]
MIN_PAST_DATES = 1

# ---- Eval column hints (consumed by moirai/functions/results.py) -------
# Method 2's prepared CSV uses the PVLib naming convention
# ``clear_sky_ghi`` (not the NSRDB ``clearsky_ghi``).  These names are
# also passed through to ``build_dataset_method2`` so the schema check
# matches the actual CSV.
TARGET_COL       = "CAF"
CLEARSKY_GHI_COL = "clear_sky_ghi"
MEASURED_GHI_COL = "w_ghr"

# ---- Model -------------------------------------------------------------
# Per-variant overrides used by ``moirai/moirai.py::resolve_model_id``.
# When you pass ``--variant moirai1`` the CLI uses ``MODEL_ID_MOIRAI1``
# instead of ``MODEL_ID`` so we never load a Moirai-2 checkpoint into the
# Moirai-1 loader.
MODEL_ID_MOIRAI1 = "Salesforce/moirai-1.1-R-small"
MODEL_ID_MOIRAI2 = "Salesforce/moirai-2.0-R-small"
# Default backbone for Method 2: small Moirai 2.0 - much faster on CPU
# than the 1.1-R-large alternative. Switch to the large model with
# ``--variant moirai1``.
MODEL_ID = MODEL_ID_MOIRAI2

CONTEXT_LENGTH    = PAST_HOURS
PREDICTION_LENGTH = FUTURE_HOURS
TARGET_DIM        = 1
FEAT_DIM          = len(FUTURE_FEATURES)

# ---- LoRA fine-tuning --------------------------------------------------
LORA_RANK           = 16
LORA_ALPHA          = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_DROPOUT        = 0.05

FT_LR             = 1e-4
FT_WEIGHT_DECAY   = 0.01
FT_MAX_EPOCHS     = 50
FT_PATIENCE       = 5
FT_BATCH_SIZE     = 32
FT_GRADIENT_CLIP  = 1.0
