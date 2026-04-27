"""
TFT configuration for the ERA5-based CAF forecasting pipeline.

This pipeline is self-contained inside ``tft_model/`` and does not reuse the
``phase2_finetuning`` synthetic-cloud dataset.
"""
import os
from pathlib import Path

# ---- Directories ----
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
MULTI_STATION_DOWNLOADS_DIR = DOWNLOADS_DIR / "multi_station_era5"
DATASET_DIR = BASE_DIR / "dataset"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

PHASE1_DIR = PROJECT_ROOT / "phase1_cloud_mapper"
PHASE1_DOWNLOADS_DIR = PHASE1_DIR / "downloads"

for d in [DOWNLOADS_DIR, MULTI_STATION_DOWNLOADS_DIR, DATASET_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Optional override for a raw ERA5 CSV that contains the requested columns.
ERA5_SOURCE_FILE = Path(
    os.environ.get(
        "TFT_ERA5_SOURCE_FILE",
        str(PHASE1_DOWNLOADS_DIR / "era5_2017_2019.csv"),
    )
)

# ---- API keys ----
NREL_API_KEY = os.environ.get("NREL_API_KEY", "")
NREL_EMAIL = os.environ.get("NREL_EMAIL", "")
CDSAPI_KEY = os.environ.get("CDSAPI_KEY", "")

# ---- Stations ----
STATIONS = [
    {"id": "tirunelveli", "lat": 9.14, "lon": 77.92, "alt_m": 45.0},
    {"id": "madurai", "lat": 9.93, "lon": 78.12, "alt_m": 101.0},
    {"id": "coimbatore", "lat": 11.02, "lon": 76.96, "alt_m": 411.0},
    {"id": "trichy", "lat": 10.79, "lon": 78.70, "alt_m": 88.0},
    {"id": "chennai", "lat": 13.08, "lon": 80.27, "alt_m": 7.0},
]
STATION_IDS = [station["id"] for station in STATIONS]

# ---- Data years and split ----
YEARS = [2017, 2018, 2019]
TRAIN_END = "2018-12-31 23:00"
VAL_START = "2019-01-01 00:00"
VAL_END = "2019-12-31 23:00"
TEST_START = "2020-01-01 00:00"

# ---- Window geometry ----
PAST_STEPS = 72
FUTURE_STEPS = 24

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

PAST_FEATURES = [
    "CAF",
    "clear_sky_ghi",
    *ERA5_FEATURE_COLUMNS,
    "zenith_angle",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]
FUTURE_FEATURES = [
    "clear_sky_ghi",
    *ERA5_FEATURE_COLUMNS,
    "zenith_angle",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

ENCODER_INPUT_DIM = len(PAST_FEATURES)
DECODER_INPUT_DIM = len(FUTURE_FEATURES)
TARGET_DIM = 1

# ---- TFT architecture ----
HIDDEN_SIZE = 64
NUM_ATTENTION_HEADS = 4
LSTM_LAYERS = 1
DROPOUT = 0.1

# ---- Training ----
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
PATIENCE = 10
GRADIENT_CLIP = 1.0
LOSS_FN = "mse"

# ---- Misc ----
SEED = 42
NUM_WORKERS = 0
