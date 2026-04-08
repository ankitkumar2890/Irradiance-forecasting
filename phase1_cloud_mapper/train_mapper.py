# ==============================================================================
# train_mapper.py — Phase 1: CloudMapper v6 (Residual MLP)
#
# v6 approach:
#   - Input: ERA5 cloud fractions at t and t-1 + time + station = 15 features
#   - Model: 2-block Residual MLP (64-dim, ~21K params)
#   - Loss: SmoothL1 only (no correlation penalty)
#   - Regularization: dropout=0.25, weight_decay=1e-3
#   - Post-training: isotonic calibration
#
# Why this works:
#   ERA5→ICON is fundamentally a cross-sectional mapping with ~0.5-0.6
#   correlation ceiling. One lag (t-1) gives trend info. Everything else
#   is regularization to prevent memorizing noise.
#
# Run:    python train_mapper.py
# ==============================================================================
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DOWNLOADS_DIR, CHECKPOINT_DIR,
    ERA5_FRACTION_COLS, ICON_COLS,
    HIDDEN_DIM, NUM_RES_BLOCKS, DROPOUT,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    HUBER_DELTA, TRAIN_STATION_ID, VERSION,
)
from features import build_all_features
from model_architecture import TAFResNet

RESULTS_DIR = Path(__file__).parent.parent / "results" / f"cloud_mapper_{VERSION}_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset — tabular, no sequences
# ---------------------------------------------------------------------------
class TabularDataset(Dataset):
    """
    Simple tabular dataset returning (features, era5_raw, target) per sample.

    features:  StandardScaler'd input features (15 dims)
    era5_raw:  unscaled ERA5 cloud fractions for α-blend residual (4 dims)
    target:    ICON cloud fractions to predict (4 dims)
    """

    def __init__(
        self,
        features: np.ndarray,
        era5_raw: np.ndarray,
        targets: np.ndarray,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.era5_raw = torch.tensor(era5_raw, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.era5_raw[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calc_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute RMSE, MAE, bias, and Pearson correlation."""
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    mae = float(np.mean(np.abs(predicted - actual)))
    bias = float(np.mean(predicted - actual))
    if len(actual) > 1 and np.std(actual) > 1e-8 and np.std(predicted) > 1e-8:
        corr = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        corr = float("nan")
    return {"RMSE": rmse, "MAE": mae, "bias": bias, "corr": corr}


# ---------------------------------------------------------------------------
# Isotonic calibration — post-training correction
# ---------------------------------------------------------------------------
def fit_isotonic(preds, actuals, names):
    """
    Fit one isotonic regression per variable.

    Isotonic regression learns a monotone function mapping raw predictions
    to calibrated predictions. It corrects systematic biases, e.g. if the
    model consistently under-predicts when cloud cover > 0.8.
    """
    calibrators = []
    print("\n  === ISOTONIC CALIBRATION ===")
    for i, name in enumerate(names):
        p, a = preds[:, i], actuals[:, i]
        rmse_before = np.sqrt(np.mean((p - a) ** 2))
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, a)
        cal = iso.predict(p)
        rmse_after = np.sqrt(np.mean((cal - a) ** 2))
        pct = (1 - rmse_after / rmse_before) * 100
        print(f"    {name:<12}  RMSE: {rmse_before:.4f} → {rmse_after:.4f} ({pct:+.1f}%)")
        calibrators.append(iso)
    return calibrators


def apply_isotonic(preds, calibrators):
    out = preds.copy()
    for i, iso in enumerate(calibrators):
        out[:, i] = iso.predict(preds[:, i])
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n  ── Step 1: Loading data ──")
    era5 = pd.read_csv(DOWNLOADS_DIR / "era5_2022_2023.csv")
    era5["datetime"] = pd.to_datetime(era5["datetime"], utc=True)
    if TRAIN_STATION_ID and "station_id" in era5.columns:
        era5 = era5[era5["station_id"] == TRAIN_STATION_ID].copy()
    print(f"  ERA5: {len(era5)} rows, stations: "
          f"{sorted(era5['station_id'].unique()) if 'station_id' in era5.columns else 'single'}")

    icon_parts = []
    for f in ["icon_2022.csv", "icon_2023.csv"]:
        p = DOWNLOADS_DIR / f
        if p.exists():
            df = pd.read_csv(p)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            if TRAIN_STATION_ID and "station_id" in df.columns:
                df = df[df["station_id"] == TRAIN_STATION_ID].copy()
            icon_parts.append(df)
            print(f"  Loaded {f}: {len(df)} rows")
    icon = pd.concat(icon_parts, ignore_index=True)

    # Merge on (datetime, station_id)
    merge_keys = ["datetime"]
    if "station_id" in era5.columns and "station_id" in icon.columns:
        merge_keys.append("station_id")

    era5 = era5.drop_duplicates(subset=merge_keys)
    icon = icon.drop_duplicates(subset=merge_keys)
    merged = era5.merge(icon, on=merge_keys, how="inner").sort_values(merge_keys).reset_index(drop=True)
    print(f"  Merged: {len(merged)} rows")

    # ── 2. Feature engineering ───────────────────────────────────────────
    print("\n  ── Step 2: Feature engineering ──")

    # Clip ICON targets to [0, 1]
    for col in ICON_COLS:
        if col in merged.columns:
            merged[col] = merged[col].clip(0.0, 1.0)

    # Build all features: clip ERA5 + time + station + t-1 lag
    merged, input_cols = build_all_features(merged)
    merged = merged.dropna(subset=input_cols + ICON_COLS).reset_index(drop=True)

    print(f"  Features ({len(input_cols)}): {input_cols}")
    print(f"  Final rows: {len(merged)}")
    print(f"  ICON means: {merged[ICON_COLS].mean().round(3).to_dict()}")

    # ── 3. Train/val split ───────────────────────────────────────────────
    print("\n  ── Step 3: Train/val split (80/20 time-based) ──")
    unique_dt = pd.Series(merged["datetime"].sort_values().unique())
    split_time = unique_dt.iloc[int(len(unique_dt) * 0.8)]
    train_mask = merged["datetime"] < split_time
    val_mask = ~train_mask
    print(f"  Split time: {split_time}")
    print(f"  Train: {train_mask.sum()} rows, Val: {val_mask.sum()} rows")

    # Save raw ERA5 total_cloud_cover for α-blend residual (BEFORE scaling)
    # This is the closest ERA5 analog to ICON cloud_cover
    train_era5_raw = merged.loc[train_mask, ["total_cloud_cover"]].values.astype(np.float32)
    val_era5_raw = merged.loc[val_mask, ["total_cloud_cover"]].values.astype(np.float32)

    # ── 4. Scale features ────────────────────────────────────────────────
    print("\n  ── Step 4: Scaling features ──")
    scaler = StandardScaler()
    train_features = scaler.fit_transform(
        merged.loc[train_mask, input_cols].values.astype(np.float32)
    )
    val_features = scaler.transform(
        merged.loc[val_mask, input_cols].values.astype(np.float32)
    )
    joblib.dump(scaler, CHECKPOINT_DIR / f"era5_scaler_{VERSION}.pkl")
    print(f"  Scaler fitted on train, saved → checkpoints/era5_scaler_{VERSION}.pkl")

    # Targets
    train_targets = merged.loc[train_mask, ICON_COLS].values.astype(np.float32)
    val_targets = merged.loc[val_mask, ICON_COLS].values.astype(np.float32)

    # Validation metadata
    val_datetimes = merged.loc[val_mask, "datetime"].tolist()
    val_station_ids = (
        merged.loc[val_mask, "station_id"].tolist()
        if "station_id" in merged.columns else []
    )

    # DataLoaders
    train_ds = TabularDataset(train_features, train_era5_raw, train_targets)
    val_ds = TabularDataset(val_features, val_era5_raw, val_targets)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train samples: {len(train_ds)}   Val samples: {len(val_ds)}")

    # ── 5. Model setup ──────────────────────────────────────────────────
    print("\n  ── Step 5: Model setup ──")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = TAFResNet(
        input_size=len(input_cols),
        hidden_dim=HIDDEN_DIM,
        num_res_blocks=NUM_RES_BLOCKS,
        output_size=len(ICON_COLS),  # 1
        dropout=DROPOUT,
    ).to(device)

    print(f"  Device: {device}")
    print(f"  Parameters: {model.param_count():,}")
    print(f"  Architecture: ResidualMLP(in={len(input_cols)}, hidden={HIDDEN_DIM}, "
          f"blocks={NUM_RES_BLOCKS}, drop={DROPOUT})")

    # ── 6. Loss + optimizer ──────────────────────────────────────────────
    print("\n  ── Step 6: Loss + optimizer ──")

    # SmoothL1 only — no correlation penalty
    criterion = nn.SmoothL1Loss(beta=HUBER_DELTA)
    mse_fn = nn.MSELoss()  # For validation metric only

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 100.0
    )

    print(f"  Loss: SmoothL1(beta={HUBER_DELTA})")
    print(f"  Optimizer: AdamW(lr={LR}, weight_decay={WEIGHT_DECAY})")
    print(f"  Scheduler: CosineAnnealing(T_max={EPOCHS}, eta_min={LR/100:.1e})")

    # ── 7. Training loop ─────────────────────────────────────────────────
    print(f"\n  ── Step 7: Training (max {EPOCHS} epochs, patience {PATIENCE}) ──")

    best_mse = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    print(f"\n  {'Epoch':>5}  {'Train':>10}  {'Val MSE':>10}  {'Val RMSE':>10}  {'LR':>10}  {'Status'}")
    print("  " + "-" * 65)

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        tloss = 0.0
        for bx, b_era5, by in train_dl:
            bx, b_era5, by = bx.to(device), b_era5.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx, era5_raw=b_era5)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tloss += loss.item()

        scheduler.step()

        # --- Validate ---
        model.eval()
        vmse = 0.0
        with torch.no_grad():
            for bx, b_era5, by in val_dl:
                pred = model(bx.to(device), era5_raw=b_era5.to(device))
                vmse += mse_fn(pred, by.to(device)).item()

        tloss /= max(len(train_dl), 1)
        vmse /= max(len(val_dl), 1)
        vrmse = vmse ** 0.5

        status = ""
        if vmse < best_mse:
            best_mse = vmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
            status = " ★"
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch <= 5 or status:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  {epoch:>5}  {tloss:>10.5f}  {vmse:>10.5f}  {vrmse:>10.4f}  {lr:>10.2e}{status}")

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    # ── 8. Load best weights ─────────────────────────────────────────────
    model.load_state_dict(best_state)
    best_rmse = best_mse ** 0.5
    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Best val RMSE: {best_rmse:.4f}")

    # ── 9. Alpha value ───────────────────────────────────────────────────
    alphas = model.get_alpha_values()
    print(f"\n  Learned α (1.0 = trust network, 0.0 = trust ERA5):")
    print(f"    cloud_cover    α = {alphas['cloud_cover']:.3f}")

    # ── 10. Predictions ──────────────────────────────────────────────────
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for bx, b_era5, by in val_dl:
            all_p.append(model(bx.to(device), era5_raw=b_era5.to(device)).cpu().numpy())
            all_t.append(by.numpy())
    preds = np.concatenate(all_p).flatten()
    targets = np.concatenate(all_t).flatten()
    val_era5_flat = val_era5_raw.flatten()

    # ── 11. Results before calibration ───────────────────────────────────
    m = calc_metrics(targets, preds)
    e = calc_metrics(targets, val_era5_flat)
    imp = (1 - m["RMSE"] / e["RMSE"]) * 100
    c = f"{m['corr']:.3f}" if not np.isnan(m["corr"]) else "N/A"
    print(f"\n  === RESULTS (before calibration) ===")
    print(f"    cloud_cover    RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} corr={c}  "
          f"ERA5={e['RMSE']:.4f}  Δ={imp:+.1f}%")

    rmse_before = m["RMSE"]

    # ── 12. Isotonic calibration ─────────────────────────────────────────
    # Reshape for isotonic (expects 2D)
    preds_2d = preds.reshape(-1, 1)
    targets_2d = targets.reshape(-1, 1)
    calibrators = fit_isotonic(preds_2d, targets_2d, ICON_COLS)
    joblib.dump(calibrators, CHECKPOINT_DIR / f"isotonic_calibrators_{VERSION}.pkl")
    preds_cal = apply_isotonic(preds_2d, calibrators).flatten()

    # ── 13. Results after calibration ────────────────────────────────────
    m_cal = calc_metrics(targets, preds_cal)
    imp_cal = (1 - m_cal["RMSE"] / e["RMSE"]) * 100
    c_cal = f"{m_cal['corr']:.3f}" if not np.isnan(m_cal["corr"]) else "N/A"
    print(f"\n  === RESULTS (after calibration) ===")
    print(f"    cloud_cover    RMSE={m_cal['RMSE']:.4f} MAE={m_cal['MAE']:.4f} corr={c_cal}  "
          f"ERA5={e['RMSE']:.4f}  Δ={imp_cal:+.1f}%")

    rmse_after = m_cal["RMSE"]
    print(f"\n  RMSE: {rmse_before:.4f} → {rmse_after:.4f} (calibrated)")

    # ── 14. ERA5 baseline ────────────────────────────────────────────────
    print(f"\n  --- RAW ERA5 BASELINE (total_cloud_cover → cloud_cover) ---")
    print(f"    RMSE={e['RMSE']:.4f}  MAE={e['MAE']:.4f}  "
          f"bias={e['bias']:+.4f}  corr={e['corr']:.3f}")

    era5_baseline_rmse = e["RMSE"]

    # ── 15. Save ─────────────────────────────────────────────────────────
    model.freeze()
    torch.save(model.state_dict(), CHECKPOINT_DIR / f"cloud_mapper_{VERSION}_frozen.pt")
    print(f"\n  Model saved → checkpoints/cloud_mapper_{VERSION}_frozen.pt")

    config = {
        "version": VERSION,
        "architecture": "TAFResNet",
        "input_cols": input_cols,
        "input_size": len(input_cols),
        "hidden_dim": HIDDEN_DIM,
        "num_res_blocks": NUM_RES_BLOCKS,
        "dropout": DROPOUT,
        "residual_skip": "total_cloud_cover",
        "best_epoch": best_epoch,
        "best_val_rmse": float(best_rmse),
        "overall_rmse_calibrated": float(rmse_after),
        "era5_fraction_cols": ERA5_FRACTION_COLS,
        "icon_cols": ICON_COLS,
        "alphas": alphas,
        "params": model.param_count(),
    }
    with open(CHECKPOINT_DIR / f"cloud_mapper_{VERSION}_config.json", "w") as fj:
        json.dump(config, fj, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Residual MLP CloudMapper {VERSION}")
    print(f"  Mapping: 4 ERA5 cloud fractions → 1 ICON cloud_cover")
    print(f"  Input: {len(input_cols)} features (4 ERA5@t + 4 ERA5@t-1 + 4 time + 3 station)")
    print(f"  Parameters: {model.param_count():,}")
    print(f"  Val RMSE (raw):        {best_rmse:.4f}")
    print(f"  Val RMSE (calibrated): {rmse_after:.4f}")
    print(f"  ERA5 baseline RMSE:    {era5_baseline_rmse:.4f}")
    print(f"  Improvement over ERA5: {(1 - rmse_after/era5_baseline_rmse)*100:+.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Phase 1: Residual MLP CloudMapper {VERSION}")
    print("  4 ERA5 fractions → 1 ICON cloud_cover")
    print("  t/t-1 lag + SmoothL1 + high regularization")
    print("=" * 60)
    train()
    print("\nDone.")