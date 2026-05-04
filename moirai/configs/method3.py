"""Method 3 config: direct GHI target, multi-station, neighbour context.

Originally lived in ``phase3_direct_ghi/config.py``. Only the constants
consumed by the ``moirai/`` pipeline are kept here - everything related
to API keys, station-cluster construction, and CSV preparation belongs
in the phase folder that produces
``moirai/final_csv/method3/processed_data_2017_2019.csv``.
"""
from __future__ import annotations

# ---- Temporal split ----------------------------------------------------
TRAIN_END  = "2018-12-31 23:00"
VAL_START  = "2019-01-01 00:00"
VAL_END    = "2019-12-31 23:00"
TEST_START = "2020-01-01 00:00"

# ---- Sliding window ----------------------------------------------------
PAST_HOURS   = 72
FUTURE_HOURS = 24

# ---- Features ----------------------------------------------------------
# Past context (encoder): includes historical measured GHI plus
# transition-aware weather features. The measured GHI target track is
# the only past-only signal.
PAST_FEATURES = [
    "w_ghr",                         # Autoregressive: historical measured GHI.
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
    "wind_speed",                    # Derived from u10/v10.
    "wind_direction",                # Derived from u10/v10.
    "wind_gusts_10m",
    "wind_speed_delta_1h",
    "neighbor_tcc_mean",
    "neighbor_temperature_2m_mean",
    "neighbor_relative_humidity_2m_mean",
    "neighbor_wind_speed_mean",
    "elevation_m",                   # Static spatial feature.
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
]
# Future covariates (decoder): same minus measured GHI and any
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

# ---- Window selection --------------------------------------------------
ANCHOR_HOURS   = [6]
MIN_PAST_DATES = 1

# ---- Eval column hints (consumed by moirai/functions/results.py) -------
TARGET_COL       = "w_ghr"
CLEARSKY_GHI_COL = "clearsky_ghi"
MEASURED_GHI_COL = "w_ghr"

# ---- Model -------------------------------------------------------------
# Default backbone for Method 3: Moirai 2.0 small (matches the published
# Moirai-2 quantile head used by the inference path).
MODEL_ID         = "Salesforce/moirai-2.0-R-small"
MODEL_ID_MOIRAI1 = "Salesforce/moirai-1.1-R-base"
MODEL_ID_MOIRAI2 = MODEL_ID

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

# ---- GHI normalisation (training stability) ----------------------------
# Divide GHI values by this constant before feeding to the model and
# multiply predictions back. Keeps the target in a similar range to
# CAF in [0, 1].
GHI_SCALE_FACTOR = 1000.0
