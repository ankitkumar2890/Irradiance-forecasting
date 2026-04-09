# ==============================================================================
# config.py — Phase 1: CloudMapper v6
# Single source of truth for all paths, secrets, and constants.
# ==============================================================================
import os
from pathlib import Path

# ---- Directories ----
BASE_DIR       = Path(__file__).resolve().parent
DOWNLOADS_DIR  = BASE_DIR / "downloads"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
ERA5_DIR       = BASE_DIR / "era5"
RESULTS_DIR    = BASE_DIR / "results"

for d in [DOWNLOADS_DIR, CHECKPOINT_DIR, ERA5_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- API Keys ----
CDSAPI_KEY   = os.environ.get("CDSAPI_KEY",   "")
NREL_API_KEY = os.environ.get("NREL_API_KEY", "")
NREL_EMAIL   = os.environ.get("NREL_EMAIL",   "")

# ---- Sites ----
STATIONS = [
    {"id": "tirunelveli", "lat": 9.14, "lon": 77.92, "alt_m": 45.0},
    {"id": "madurai", "lat": 9.93, "lon": 78.12, "alt_m": 101.0},
    {"id": "coimbatore", "lat": 11.02, "lon": 76.96, "alt_m": 411.0},
    {"id": "trichy", "lat": 10.79, "lon": 78.70, "alt_m": 88.0},
    {"id": "chennai", "lat": 13.08, "lon": 80.27, "alt_m": 7.0},
]
PRIMARY_STATION = STATIONS[0]

# ---- Column names ----
ERA5_FRACTION_COLS = [
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
]
ICON_COLS = ["cloud_cover"]  # Single target: 4 ERA5 fractions → 1 ICON cloud cover

# ---- Model hyperparameters (v6 Residual MLP) ----
HIDDEN_DIM     = 64        # Smaller: 15 features + low SNR → 64 is sufficient
NUM_RES_BLOCKS = 2         # Two residual blocks for non-linear interactions
DROPOUT        = 0.25      # Higher dropout for noisy targets (0.5-0.6 corr ceiling)
BATCH_SIZE     = 256
EPOCHS         = 200
LR             = 1e-3      # MLP can handle higher LR than GRU
WEIGHT_DECAY   = 1e-3      # Aggressive: needed in low-SNR to prevent noise memorization
PATIENCE       = 30        # Early stopping patience

# ---- Loss ----
HUBER_DELTA = 0.1          # SmoothL1 transition: MSE-like below, MAE-like above

# ---- Station filter ----
TRAIN_STATION_ID = None    # None = all stations
SAMPLE_ONE_STATION_PER_TIMESTEP = False  # Used by fetch_data.py to sample stations

# ---- Version tag ----
VERSION = "v6"
