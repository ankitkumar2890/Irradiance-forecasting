"""Evaluate fine-tuned predictions: GHI recovery, metrics, stratified analysis, plots."""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASET_DIR, RESULTS_DIR, PREDICTION_LENGTH


def compute_metrics(y_true, y_pred, label=""):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {}
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mean_obs = np.mean(y_true)
    nrmse = (rmse / mean_obs * 100) if mean_obs > 0 else float("nan")
    day = y_true > 0.01
    mape = np.mean(np.abs((y_pred[day] - y_true[day]) / (y_true[day] + 1e-8))) * 100 if day.any() else float("nan")
    return {"label": label, "MAE": mae, "RMSE": rmse, "nRMSE_pct": nrmse, "MAPE_pct": mape, "N": int(len(y_true))}


def main():
    # ---- Load predictions ----
    pred_file = RESULTS_DIR / "finetuned_predictions.csv"
    if not pred_file.exists():
        print(f"Error: {pred_file} not found. Run 05_finetuned_inference.py first.")
        sys.exit(1)

    df = pd.read_csv(pred_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # ---- Load clear-sky for GHI recovery ----
    proc = pd.read_csv(DATASET_DIR / "processed_data_2017_2019.csv")
    proc["datetime"] = pd.to_datetime(proc["datetime"])
    cs_map = proc.set_index("datetime")["clear_sky_ghi"].to_dict()
    zen_map = proc.set_index("datetime")["zenith_angle"].to_dict()

    df["clear_sky_ghi"] = df["datetime"].map(cs_map)
    df["zenith_angle"] = df["datetime"].map(zen_map)
    df = df.dropna(subset=["clear_sky_ghi", "zenith_angle"])

    # ---- GHI recovery ----
    df["GHI_true"] = df["CAF_true"] * df["clear_sky_ghi"]
    df["GHI_pred"] = df["CAF_pred"] * df["clear_sky_ghi"]

    # ---- Daytime filter ----
    daytime = df[df["zenith_angle"] < 85].copy()
    print(f"Total rows: {len(df)}  Daytime rows: {len(daytime)}")

    # ---- Persistence baseline ----
    # Persistence assumes the CAF right now will be the CAF for all future steps.
    daytime_sorted = daytime.sort_values("datetime")
    persist_caf = daytime_sorted["CAF_true"].shift(PREDICTION_LENGTH).dropna().values
    actual_caf = daytime_sorted["CAF_true"].iloc[PREDICTION_LENGTH:].values
    n = min(len(persist_caf), len(actual_caf))
    persist_rmse = np.sqrt(np.mean((persist_caf[:n] - actual_caf[:n]) ** 2))

    # ---- CAF Metrics (daytime) ----
    caf_m = compute_metrics(daytime["CAF_true"].values, daytime["CAF_pred"].values, "CAF_finetuned")
    caf_m["Skill_vs_persist"] = 1.0 - caf_m["RMSE"] / persist_rmse if persist_rmse > 0 else float("nan")

    # ---- GHI Metrics (daytime) ----
    ghi_m = compute_metrics(daytime["GHI_true"].values, daytime["GHI_pred"].values, "GHI_finetuned")

    print("\n" + "=" * 55)
    print("  FINE-TUNED MODEL — DAYTIME METRICS (zenith < 85)")
    print("=" * 55)
    for label, m in [("CAF", caf_m), ("GHI (W/m2)", ghi_m)]:
        print(f"\n  {label}:")
        for k, v in m.items():
            if k == "label":
                continue
            print(f"    {k:15s}: {v:.4f}" if isinstance(v, float) else f"    {k:15s}: {v}")
    print(f"\n  Persistence RMSE: {persist_rmse:.4f}")
    print(f"  Forecast Skill:   {caf_m.get('Skill_vs_persist', 'N/A'):.4f}")

    # ---- Stratified ----
    print("\n  Stratified CAF RMSE:")
    for name, lo, hi in [("Clear (>0.7)", 0.7, 1.01), ("Partly (0.3-0.7)", 0.3, 0.7), ("Overcast (<0.3)", -0.01, 0.3)]:
        mask = (daytime["CAF_true"] >= lo) & (daytime["CAF_true"] < hi)
        sub = daytime[mask]
        if len(sub) > 0:
            r = np.sqrt(np.mean((sub["CAF_pred"] - sub["CAF_true"]) ** 2))
            print(f"    {name:20s}: RMSE={r:.4f}  N={len(sub)}")

    # ---- Multi-horizon ----
    print("\n  RMSE by forecast hour:")
    preds_by_hour = {h: [] for h in range(PREDICTION_LENGTH)}
    truth_by_hour = {h: [] for h in range(PREDICTION_LENGTH)}
    for _, row in df.iterrows():
        h = int(row["hour"]) - 6  # offset from 06:00
        if 0 <= h < PREDICTION_LENGTH:
            preds_by_hour[h].append(row["CAF_pred"])
            truth_by_hour[h].append(row["CAF_true"])

    horizon_rmse = {}
    for h in range(PREDICTION_LENGTH):
        if truth_by_hour[h]:
            t, p = np.array(truth_by_hour[h]), np.array(preds_by_hour[h])
            horizon_rmse[h] = np.sqrt(np.mean((p - t) ** 2))

    for h in [0, 5, 11, 23]:
        if h in horizon_rmse:
            print(f"    +{h+1:2d}h: RMSE={horizon_rmse[h]:.4f}")

    # ---- Save all metrics ----
    all_metrics = {"caf": caf_m, "ghi": ghi_m, "persistence_rmse": float(persist_rmse)}
    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # ---- Plot 1: Week time-series (Using a week from Nov 2019 test set) ----
    week = daytime[(daytime["datetime"] >= "2019-11-01") & (daytime["datetime"] < "2019-11-08")]
    if len(week) < 10:
        week = daytime.iloc[:168]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Fine-Tuned Moirai — CAF & GHI (Nov 2019 Test Set)", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(week["datetime"], week["GHI_true"], label="GHI True", color="#1f77b4", lw=1.5)
    ax.plot(week["datetime"], week["GHI_pred"], label="GHI Pred (fine-tuned)", color="#ff7f0e", lw=1.3, ls="--")
    ax.set_ylabel("GHI (W/m²)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(week["datetime"], week["CAF_true"], label="CAF True", color="#1f77b4", lw=1.5)
    ax2.plot(week["datetime"], week["CAF_pred"], label="CAF Pred", color="#ff7f0e", lw=1.3, ls="--")
    ax2.set_ylabel("CAF [0,1]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "finetuned_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 2: Scatter ----
    fig2, ax3 = plt.subplots(figsize=(7, 6))
    ax3.scatter(daytime["CAF_true"], daytime["CAF_pred"], alpha=0.3, s=3, c=daytime["hour"], cmap="hsv")
    ax3.plot([0, 1], [0, 1], "k--", lw=1)
    ax3.set_xlabel("CAF True")
    ax3.set_ylabel("CAF Predicted")
    ax3.set_title(f"Fine-Tuned Scatter | RMSE={caf_m['RMSE']:.4f}")
    fig2.savefig(RESULTS_DIR / "finetuned_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 3: Horizon degradation ----
    if horizon_rmse:
        fig3, ax4 = plt.subplots(figsize=(10, 5))
        hours = sorted(horizon_rmse.keys())
        rmses = [horizon_rmse[h] for h in hours]
        ax4.plot([h+1 for h in hours], rmses, "o-", color="#1f77b4")
        ax4.set_xlabel("Forecast Hour")
        ax4.set_ylabel("CAF RMSE")
        ax4.set_title("Forecast Degradation Curve")
        ax4.grid(True, alpha=0.3)
        fig3.savefig(RESULTS_DIR / "finetuned_horizon_rmse.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n  Plots saved → {RESULTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
