"""Evaluate ERA5-direct fine-tuned validation predictions against measured GHI."""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_DIR, DOWNLOADS_DIR, RESULTS_DIR, PREDICTION_LENGTH, YEARS, FINETUNE_STATION
from master_metrics import calculate_metrics, print_metrics
from master_plots import plot_4panel_evaluation, plot_time_series
from export_pdf import build_finetuned_pdf


VALIDATION_GHI_FILTER_WM2 = 20.0


def normalize_ghi_to_hour_grid(df):
    df = df.copy()
    minutes = sorted(df["datetime"].dt.minute.dropna().unique().tolist())
    if minutes == [30]:
        df["datetime"] = df["datetime"] - pd.Timedelta(minutes=30)
    return df


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


def load_measured_ghi():
    frames = []
    for year in YEARS:
        ghi_file = DOWNLOADS_DIR / f"ghi_{year}.csv"
        df = pd.read_csv(ghi_file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = normalize_ghi_to_hour_grid(df)
        frames.append(df[["datetime", "w_ghr"]])
    ghi = pd.concat(frames, ignore_index=True)
    return ghi.drop_duplicates(subset=["datetime"]).sort_values("datetime")


def main():
    # ---- Load predictions ----
    pred_file = RESULTS_DIR / "finetuned_predictions.csv"
    if not pred_file.exists():
        print(f"Error: {pred_file} not found. Run 05_finetuned_inference.py first.")
        sys.exit(1)

    df = pd.read_csv(pred_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # ---- Load processed features + measured GHI ----
    proc = pd.read_csv(DATASET_DIR / "processed_data_2017_2019.csv")
    proc["datetime"] = pd.to_datetime(proc["datetime"])
    ghi = load_measured_ghi()
    proc = proc[["datetime", "clear_sky_ghi", "zenith_angle"]].drop_duplicates(subset=["datetime"])
    df = (
        df.merge(proc, on="datetime", how="left")
        .merge(ghi, on="datetime", how="left")
        .sort_values(["datetime", "lead_time_h"] if "lead_time_h" in df.columns else ["datetime"])
        .reset_index(drop=True)
    )
    df = df.dropna(subset=["clear_sky_ghi", "zenith_angle", "w_ghr"])
    if df.empty:
        print("No aligned evaluation rows were found after merging predictions, processed features, and measured GHI.")
        print("This usually means the timestamp grids still do not match between predictions and ghi_*.csv.")
        return

    # ---- GHI recovery and measured-truth evaluation target ----
    df["GHI_true"] = df["w_ghr"]
    df["GHI_pred"] = df["CAF_pred"] * df["clear_sky_ghi"]

    # ---- Post-reconstruction filter ----
    validation_df = df[df["GHI_true"] > VALIDATION_GHI_FILTER_WM2].copy()
    print(f"Total rows: {len(df)}  Filtered rows (GHI_true > {VALIDATION_GHI_FILTER_WM2:.0f}): {len(validation_df)}")
    if validation_df.empty:
        print("No validation rows passed the post-reconstruction GHI filter.")
        return

    # ---- Persistence baseline on hourly timeline ----
    proc_full = (
        proc.merge(ghi, on="datetime", how="inner")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    proc_full["CAF_measured"] = np.where(
        proc_full["clear_sky_ghi"] > 1.0,
        (proc_full["w_ghr"] / proc_full["clear_sky_ghi"]).clip(0.0, 1.0),
        0.0,
    )
    proc_full["CAF_persist_24h"] = proc_full["CAF_measured"].shift(PREDICTION_LENGTH)
    proc_full["GHI_persist_24h"] = proc_full["CAF_persist_24h"] * proc_full["clear_sky_ghi"]
    persist_eval = proc_full[
        (proc_full["w_ghr"] > VALIDATION_GHI_FILTER_WM2) &
        proc_full["CAF_persist_24h"].notna()
    ].copy()
    persist_rmse = float("nan")
    if not persist_eval.empty:
        persist_rmse = np.sqrt(
            np.mean((persist_eval["CAF_persist_24h"] - persist_eval["CAF_measured"]) ** 2)
        )

    # ---- CAF Metrics (daytime) ----
    caf_m = compute_metrics(validation_df["CAF_true"].values, validation_df["CAF_pred"].values, "CAF_finetuned")
    caf_m["Skill_vs_persist"] = 1.0 - caf_m["RMSE"] / persist_rmse if persist_rmse > 0 else float("nan")

    # ---- GHI Metrics (daytime) ----
    ghi_m = compute_metrics(validation_df["GHI_true"].values, validation_df["GHI_pred"].values, "GHI_finetuned")
    master_ghi_m = calculate_metrics(
        validation_df["GHI_true"].to_numpy(),
        validation_df["GHI_pred"].to_numpy(),
        mape_threshold=VALIDATION_GHI_FILTER_WM2,
    )

    print("\n" + "=" * 55)
    print("  ERA5-DIRECT MODEL — DAYTIME METRICS")
    print("=" * 55)
    for label, m in [("CAF", caf_m), ("GHI (W/m2)", ghi_m)]:
        print(f"\n  {label}:")
        for k, v in m.items():
            if k == "label":
                continue
            print(f"    {k:15s}: {v:.4f}" if isinstance(v, float) else f"    {k:15s}: {v}")
    print(f"\n  Persistence RMSE: {persist_rmse:.4f}")
    print(f"  Forecast Skill:   {caf_m.get('Skill_vs_persist', 'N/A'):.4f}")
    print_metrics(
        master_ghi_m,
        title="MASTER GHI METRICS vs w_ghr (FILTER: measured GHI_true > 20)",
    )

    # ---- Stratified ----
    print("\n  Stratified CAF RMSE:")
    for name, lo, hi in [("Clear (>0.7)", 0.7, 1.01), ("Partly (0.3-0.7)", 0.3, 0.7), ("Overcast (<0.3)", -0.01, 0.3)]:
        mask = (validation_df["CAF_true"] >= lo) & (validation_df["CAF_true"] < hi)
        sub = validation_df[mask]
        if len(sub) > 0:
            r = np.sqrt(np.mean((sub["CAF_pred"] - sub["CAF_true"]) ** 2))
            print(f"    {name:20s}: RMSE={r:.4f}  N={len(sub)}")

    # ---- Multi-horizon ----
    print("\n  RMSE by forecast hour:")
    horizon_rmse = {}
    if "lead_time_h" in validation_df.columns:
        for lead_time, group in validation_df.groupby("lead_time_h"):
            t = group["CAF_true"].to_numpy()
            p = group["CAF_pred"].to_numpy()
            horizon_rmse[int(lead_time)] = float(np.sqrt(np.mean((p - t) ** 2)))

    for h in [1, 6, 12, 24]:
        if h in horizon_rmse:
            print(f"    +{h:2d}h: RMSE={horizon_rmse[h]:.4f}")

    # ---- Save all metrics ----
    all_metrics = {
        "filters": {"ghi_true_gt_wm2": VALIDATION_GHI_FILTER_WM2},
        "caf": caf_m,
        "ghi": ghi_m,
        "ghi_master_metrics": {k: float(v) for k, v in master_ghi_m.items()},
        "persistence_rmse": float(persist_rmse),
        "horizon_rmse": horizon_rmse,
    }
    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    report_lines = [
        "ERA5-DIRECT MOIRAI EVALUATION REPORT",
        f"Filter: measured GHI_true > {VALIDATION_GHI_FILTER_WM2:.0f} W/m^2",
        f"Rows evaluated: {len(validation_df)}",
        "",
        "CAF Metrics:",
        f"  RMSE: {caf_m['RMSE']:.4f}",
        f"  MAE: {caf_m['MAE']:.4f}",
        f"  nRMSE_pct: {caf_m['nRMSE_pct']:.4f}",
        f"  MAPE_pct: {caf_m['MAPE_pct']:.4f}",
        f"  Skill_vs_persist: {caf_m['Skill_vs_persist']:.4f}",
        "",
        "GHI Master Metrics vs measured w_ghr:",
        f"  RMSE: {master_ghi_m['RMSE']:.4f}",
        f"  nRMSE: {master_ghi_m['nRMSE']:.4f}",
        f"  MAE: {master_ghi_m['MAE']:.4f}",
        f"  MAPE: {master_ghi_m['MAPE']:.4f}",
    ]
    if horizon_rmse:
        report_lines.extend([
            "",
            "Horizon RMSE (CAF):",
        ])
        report_lines.extend(
            [f"  +{lead:02d}h: {rmse:.4f}" for lead, rmse in sorted(horizon_rmse.items())]
        )
    with open(RESULTS_DIR / "evaluation_report.txt", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    # ---- Plot 1: Week time-series from the 2019 validation period ----
    week = validation_df[(validation_df["datetime"] >= "2019-11-01") & (validation_df["datetime"] < "2019-11-08")]
    if len(week) < 10:
        week = validation_df.iloc[:168]
    plot_time_series(
        week["datetime"].to_numpy(),
        week["GHI_true"].to_numpy(),
        week["GHI_pred"].to_numpy(),
        title="ERA5-Direct Moirai — Measured vs Predicted GHI (Filtered Validation Window)",
        save_path=str(RESULTS_DIR / "finetuned_timeseries.png"),
    )
    plot_4panel_evaluation(
        validation_df["GHI_true"].to_numpy(),
        validation_df["GHI_pred"].to_numpy(),
        hour_array=validation_df["hour"].to_numpy(),
        title="ERA5-Direct Moirai — Validation GHI Evaluation",
        save_path=str(RESULTS_DIR / "finetuned_4panel.png"),
    )

    validation_report = validation_df[[
        "datetime", "hour", "lead_time_h", "forecast_start",
        "CAF_true", "CAF_pred", "clear_sky_ghi", "zenith_angle",
        "GHI_true", "GHI_pred",
    ]].copy()
    validation_report["station_id"] = FINETUNE_STATION
    validation_report.to_csv(RESULTS_DIR / "validation_report_data.csv", index=False)

    # ---- Plot 3: Horizon degradation ----
    if horizon_rmse:
        fig3, ax4 = plt.subplots(figsize=(10, 5))
        hours = sorted(horizon_rmse.keys())
        rmses = [horizon_rmse[h] for h in hours]
        ax4.plot(hours, rmses, "o-", color="#1f77b4")
        ax4.set_xlabel("Forecast Lead Time (h)")
        ax4.set_ylabel("CAF RMSE")
        ax4.set_title("Forecast Degradation Curve")
        ax4.grid(True, alpha=0.3)
        fig3.savefig(RESULTS_DIR / "finetuned_horizon_rmse.png", dpi=150, bbox_inches="tight")
        plt.close()

    build_finetuned_pdf(str(RESULTS_DIR))

    print(f"\n  Plots saved → {RESULTS_DIR}/")
    print(f"  Validation CSV saved → {RESULTS_DIR / 'validation_report_data.csv'}")
    print(f"  Report saved → {RESULTS_DIR / 'evaluation_report.txt'}")
    print("Done.")


if __name__ == "__main__":
    main()
