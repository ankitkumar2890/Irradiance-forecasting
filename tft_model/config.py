"""
TFT Configuration — hyperparameters and paths.

Reuses the same data directories and temporal split from the
existing phase2_finetuning pipeline.
"""
import os
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent                      # tft_model/
PROJECT_ROOT = BASE_DIR.parent                                  # moirai_finetuning/
PHASE2_DIR = PROJECT_ROOT / "phase2_finetuning"

# Reuse the existing dataset produced by 03_build_dataset.py
DATASET_DIR = PHASE2_DIR / "dataset"

# TFT-specific output dirs
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

for d in [CHECKPOINT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Sequence geometry (must match 03_build_dataset.py) ───────────────────────
PAST_STEPS = 72          # encoder length  (same as PAST_HOURS)
FUTURE_STEPS = 24        # decoder / prediction length (same as FUTURE_HOURS)

# Feature dimensions — derived from the .npy arrays
#   X_past  shape: (N, 72, 8)  → 8 features (CAF + 7 covariates)
#   X_future shape: (N, 24, 7) → 7 known-future covariates
ENCODER_INPUT_DIM = 8     # CAF, clear_sky_ghi, cloud_cover, zenith, h_sin, h_cos, d_sin, d_cos
DECODER_INPUT_DIM = 7     # clear_sky_ghi, cloud_cover, zenith, h_sin, h_cos, d_sin, d_cos
TARGET_DIM = 1            # CAF (scalar)

# Past-only features: CAF (index 0) — not available in the future
NUM_PAST_ONLY_FEATURES = 1
# Known-future features: indices 1..7 of X_past match X_future columns
NUM_KNOWN_FEATURES = 7

# ── TFT Architecture ────────────────────────────────────────────────────────
HIDDEN_SIZE = 64          # hidden dimension across all sub-networks
NUM_ATTENTION_HEADS = 4   # interpretable multi-head attention heads
LSTM_LAYERS = 1           # LSTM encoder/decoder depth
DROPOUT = 0.1             # dropout rate for GRN, attention, LSTM

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
PATIENCE = 10             # early-stopping patience (epochs without val improvement)
GRADIENT_CLIP = 1.0       # max gradient norm

# ── Loss ─────────────────────────────────────────────────────────────────────
LOSS_FN = "mse"           # "mse" or "huber"

# ── Misc ─────────────────────────────────────────────────────────────────────
SEED = 42
NUM_WORKERS = 0           # DataLoader workers (0 = main process)
