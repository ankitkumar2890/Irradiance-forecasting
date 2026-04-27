"""Evaluate fine-tuned validation predictions against measured GHI with shared metrics/plots utilities."""
import sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_DIR,
    MULTI_STATION_DOWNLOADS_DIR,
    RESULTS_DIR,
    PREDICTION_LENGTH,
    STATIONS,
    YEARS,
)
from master_metrics import calculate_metrics, print_metrics
from master_plots import (
    plot_4panel_evaluation,
    plot_prediction_interval_comparison,
    plot_two_week_comparison,
)
from export_pdf import build_finetuned_pdf


VALIDATION_GHI_FILTER_WM2 = 20.0
ONE_WEEK_DAYS = 7
RANDOM_WEEK_SEED = 42


def normalize_ghi_to_hour_grid(df):
    df = df.copy()
    minutes = sorted(df["datetime"].dt.minute.dropna().unique().tolist())
    if minutes == [30]:
        prev_hour = df.copy()
        prev_hour["datetime"] = prev_hour["datetime"] - pd.Timedelta(minutes=30)
        next_hour = df.copy()
        next_hour["datetime"] = next_hour["datetime"] + pd.Timedelta(minutes=30)
        df = (
            pd.concat([prev_hour, next_hour], ignore_index=True)
            .groupby("datetime", as_index=False)["w_ghr"]
            .mean()
            .sort_values("datetime")
        )
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


def slice_window(df, start, days=ONE_WEEK_DAYS):
    end = pd.to_datetime(start) + pd.Timedelta(days=days)
    return df[(df["datetime"] >= start) & (df["datetime"] < end)].copy()


def choose_random_later_week_start(df, first_start, days=ONE_WEEK_DAYS, seed=RANDOM_WEEK_SEED):
    latest_start = df["datetime"].max() - pd.Timedelta(days=days)
    earliest_start = pd.to_datetime(first_start) + pd.Timedelta(days=days * 2)
    candidate_days = (
        df.loc[
            (df["datetime"] >= earliest_start) &
            (df["datetime"] <= latest_start),
            "datetime",
        ]
        .dt.floor("D")
        .drop_duplicates()
        .sort_values()
    )
    if candidate_days.empty:
        candidate_days = (
            df.loc[df["datetime"] <= latest_start, "datetime"]
            .dt.floor("D")
            .drop_duplicates()
            .sort_values()
        )
    if candidate_days.empty:
        return pd.to_datetime(first_start) + pd.Timedelta(days=days)
    return candidate_days.sample(1, random_state=seed).iloc[0]


def load_measured_ghi():
    frames = []
    for station in STATIONS:
        station_id = station["id"]
        for year in YEARS:
            ghi_file = MULTI_STATION_DOWNLOADS_DIR / station_id / f"ghi_{year}.csv"
            df = pd.read_csv(ghi_file)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = normalize_ghi_to_hour_grid(df)
            df["station_id"] = station_id
            frames.append(df[["station_id", "datetime", "w_ghr"]])
    ghi = pd.concat(frames, ignore_index=True)
    return ghi.drop_duplicates(subset=["station_id", "datetime"]).sort_values(["station_id", "datetime"])


def build_station_metrics(df, value_true, value_pred, label_prefix):
    metrics = {}
    for station_id, station_df in df.groupby("station_id", sort=True):
        metrics[station_id] = compute_metrics(
            station_df[value_true].to_numpy(),
            station_df[value_pred].to_numpy(),
            f"{label_prefix}_{station_id}",
        )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ghi-filter-wm2",
        type=float,
        default=VALIDATION_GHI_FILTER_WM2,
        help="Only evaluate rows with measured GHI above this threshold.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ghi_filter_wm2 = float(args.ghi_filter_wm2)

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
    proc = proc[["station_id", "datetime", "clear_sky_ghi", "zenith_angle"]].drop_duplicates(
        subset=["station_id", "datetime"]
    )
    df = (
        df.merge(proc, on=["station_id", "datetime"], how="left")
        .merge(ghi, on=["station_id", "datetime"], how="left")
        .sort_values(
            ["station_id", "datetime", "lead_time_h"]
            if "lead_time_h" in df.columns
            else ["station_id", "datetime"]
        )
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
    if "CAF_p10" in df.columns:
        df["GHI_pred_p10"] = df["CAF_p10"] * df["clear_sky_ghi"]
    if "CAF_p90" in df.columns:
        df["GHI_pred_p90"] = df["CAF_p90"] * df["clear_sky_ghi"]

    # ---- Post-reconstruction filter ----
    validation_df = df[df["GHI_true"] > ghi_filter_wm2].copy()
    print(f"Total rows: {len(df)}  Filtered rows (GHI_true > {ghi_filter_wm2:.0f}): {len(validation_df)}")
    if validation_df.empty:
        print("No validation rows passed the post-reconstruction GHI filter.")
        return
    plot_station_id = (
        validation_df["station_id"].value_counts().sort_values(ascending=False).index[0]
    )
    plot_df = df[df["station_id"] == plot_station_id].copy()
    plot_validation_df = validation_df[validation_df["station_id"] == plot_station_id].copy()
    first_week_start = plot_df["datetime"].min()
    first_week_plot_df = slice_window(plot_df, first_week_start)
    first_week_metrics_df = slice_window(plot_validation_df, first_week_start)
    random_week_start = choose_random_later_week_start(plot_validation_df, first_week_start)
    random_week_plot_df = slice_window(plot_df, random_week_start)
    random_week_metrics_df = slice_window(plot_validation_df, random_week_start)

    # ---- Persistence baseline on hourly timeline ----
    proc_full = (
        proc.merge(ghi, on=["station_id", "datetime"], how="inner")
        .sort_values(["station_id", "datetime"])
        .reset_index(drop=True)
    )
    proc_full["CAF_measured"] = np.where(
        proc_full["clear_sky_ghi"] > 1.0,
        (proc_full["w_ghr"] / proc_full["clear_sky_ghi"]).clip(0.0, 1.0),
        0.0,
    )
    proc_full["CAF_persist_24h"] = proc_full.groupby("station_id")["CAF_measured"].shift(PREDICTION_LENGTH)
    proc_full["GHI_persist_24h"] = proc_full["CAF_persist_24h"] * proc_full["clear_sky_ghi"]
    persist_eval = proc_full[
        (proc_full["w_ghr"] > ghi_filter_wm2) &
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
    first_week_caf_m = compute_metrics(first_week_metrics_df["CAF_true"].values, first_week_metrics_df["CAF_pred"].values, "CAF_week_1")
    random_week_caf_m = compute_metrics(random_week_metrics_df["CAF_true"].values, random_week_metrics_df["CAF_pred"].values, "CAF_random_week")

    # ---- GHI Metrics (daytime) ----
    ghi_m = compute_metrics(validation_df["GHI_true"].values, validation_df["GHI_pred"].values, "GHI_finetuned")
    first_week_ghi_m = compute_metrics(first_week_metrics_df["GHI_true"].values, first_week_metrics_df["GHI_pred"].values, "GHI_week_1")
    random_week_ghi_m = compute_metrics(random_week_metrics_df["GHI_true"].values, random_week_metrics_df["GHI_pred"].values, "GHI_random_week")
    master_ghi_m = calculate_metrics(
        validation_df["GHI_true"].to_numpy(),
        validation_df["GHI_pred"].to_numpy(),
        mape_threshold=ghi_filter_wm2,
    )
    station_caf_metrics = build_station_metrics(validation_df, "CAF_true", "CAF_pred", "CAF")
    station_ghi_metrics = build_station_metrics(validation_df, "GHI_true", "GHI_pred", "GHI")

    print("\n" + "=" * 55)
    print("  FINE-TUNED MODEL — DAYTIME METRICS")
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
        title=f"MASTER GHI METRICS vs w_ghr (FILTER: measured GHI_true > {ghi_filter_wm2:.0f})",
    )
    print(f"\n  Plot station for weekly figures: {plot_station_id}")
    print("\n  One-Week Window Metrics:")
    print_metrics(first_week_ghi_m, title="ONE-WEEK GHI METRICS", unit="W/m²")
    print("\n  Random Later One-Week Window Metrics:")
    print_metrics(random_week_ghi_m, title="RANDOM ONE-WEEK GHI METRICS", unit="W/m²")
    print("\n  Per-station GHI RMSE:")
    for station_id, metrics in station_ghi_metrics.items():
        print(f"    {station_id:12s}: RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  N={metrics['N']}")

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
        "filters": {"ghi_true_gt_wm2": ghi_filter_wm2},
        "caf": caf_m,
        "caf_one_week": first_week_caf_m,
        "caf_random_week": random_week_caf_m,
        "ghi": ghi_m,
        "ghi_one_week": first_week_ghi_m,
        "ghi_random_week": random_week_ghi_m,
        "ghi_master_metrics": {k: float(v) for k, v in master_ghi_m.items()},
        "plot_station_id": plot_station_id,
        "station_caf_metrics": station_caf_metrics,
        "station_ghi_metrics": station_ghi_metrics,
        "persistence_rmse": float(persist_rmse),
        "horizon_rmse": horizon_rmse,
    }
    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    report_lines = [
        "FINE-TUNED MOIRAI EVALUATION REPORT",
        f"Filter: measured GHI_true > {ghi_filter_wm2:.0f} W/m^2",
        f"Rows evaluated: {len(validation_df)}",
        f"Stations evaluated: {', '.join(sorted(validation_df['station_id'].dropna().unique().tolist()))}",
        f"Weekly plot station: {plot_station_id}",
        "",
        "CAF Metrics:",
        f"  RMSE: {caf_m['RMSE']:.4f}",
        f"  MAE: {caf_m['MAE']:.4f}",
        f"  nRMSE_pct: {caf_m['nRMSE_pct']:.4f}",
        f"  MAPE_pct: {caf_m['MAPE_pct']:.4f}",
        f"  Skill_vs_persist: {caf_m['Skill_vs_persist']:.4f}",
        "",
        "One-Week CAF Metrics:",
        f"  N: {first_week_caf_m['N']}",
        f"  RMSE: {first_week_caf_m['RMSE']:.4f}",
        f"  MAE: {first_week_caf_m['MAE']:.4f}",
        f"  nRMSE_pct: {first_week_caf_m['nRMSE_pct']:.4f}",
        f"  MAPE_pct: {first_week_caf_m['MAPE_pct']:.4f}",
        "",
        "Random Later One-Week CAF Metrics:",
        f"  N: {random_week_caf_m['N']}",
        f"  RMSE: {random_week_caf_m['RMSE']:.4f}",
        f"  MAE: {random_week_caf_m['MAE']:.4f}",
        f"  nRMSE_pct: {random_week_caf_m['nRMSE_pct']:.4f}",
        f"  MAPE_pct: {random_week_caf_m['MAPE_pct']:.4f}",
        "",
        "GHI Master Metrics vs measured w_ghr:",
        f"  RMSE: {master_ghi_m['RMSE']:.4f}",
        f"  nRMSE: {master_ghi_m['nRMSE']:.4f}",
        f"  MAE: {master_ghi_m['MAE']:.4f}",
        f"  MAPE: {master_ghi_m['MAPE']:.4f}",
        "",
        "One-Week GHI Metrics:",
        f"  N: {first_week_ghi_m['N']}",
        f"  RMSE: {first_week_ghi_m['RMSE']:.4f}",
        f"  nRMSE_pct: {first_week_ghi_m['nRMSE_pct']:.4f}",
        f"  MAE: {first_week_ghi_m['MAE']:.4f}",
        f"  MAPE_pct: {first_week_ghi_m['MAPE_pct']:.4f}",
        "",
        "Random Later One-Week GHI Metrics:",
        f"  N: {random_week_ghi_m['N']}",
        f"  RMSE: {random_week_ghi_m['RMSE']:.4f}",
        f"  nRMSE_pct: {random_week_ghi_m['nRMSE_pct']:.4f}",
        f"  MAE: {random_week_ghi_m['MAE']:.4f}",
        f"  MAPE_pct: {random_week_ghi_m['MAPE_pct']:.4f}",
    ]
    report_lines.extend([
        "",
        "Per-Station GHI Metrics:",
    ])
    report_lines.extend(
        [
            f"  {station_id}: RMSE={metrics['RMSE']:.4f}, nRMSE_pct={metrics['nRMSE_pct']:.4f}, MAE={metrics['MAE']:.4f}, MAPE_pct={metrics['MAPE_pct']:.4f}, N={metrics['N']}"
            for station_id, metrics in station_ghi_metrics.items()
        ]
    )
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

    plot_two_week_comparison(
        first_week_plot_df["datetime"].to_numpy(),
        first_week_plot_df["GHI_true"].to_numpy(),
        first_week_plot_df["GHI_pred"].to_numpy(),
        start=first_week_start,
        window_days=ONE_WEEK_DAYS,
        title=f"Fine-Tuned Moirai — One-Week Measured vs Predicted GHI ({plot_station_id})",
        save_path=str(RESULTS_DIR / "finetuned_one_week_comparison.png"),
        n_points=len(first_week_plot_df),
        metrics=first_week_ghi_m,
        ghi_threshold=ghi_filter_wm2,
    )
    plot_two_week_comparison(
        random_week_plot_df["datetime"].to_numpy(),
        random_week_plot_df["GHI_true"].to_numpy(),
        random_week_plot_df["GHI_pred"].to_numpy(),
        start=random_week_start,
        window_days=ONE_WEEK_DAYS,
        title=f"Fine-Tuned Moirai — Random Later One-Week Measured vs Predicted GHI ({plot_station_id})",
        save_path=str(RESULTS_DIR / "finetuned_random_week_comparison.png"),
        n_points=len(random_week_plot_df),
        metrics=random_week_ghi_m,
        ghi_threshold=ghi_filter_wm2,
    )
    if {"GHI_pred_p10", "GHI_pred_p90"}.issubset(first_week_plot_df.columns):
        plot_prediction_interval_comparison(
            first_week_plot_df["datetime"].to_numpy(),
            first_week_plot_df["GHI_true"].to_numpy(),
            first_week_plot_df["GHI_pred"].to_numpy(),
            first_week_plot_df["GHI_pred_p10"].to_numpy(),
            first_week_plot_df["GHI_pred_p90"].to_numpy(),
            start=first_week_start,
            window_days=ONE_WEEK_DAYS,
            title=f"Fine-Tuned Moirai — One-Week GHI Prediction Interval ({plot_station_id})",
            save_path=str(RESULTS_DIR / "finetuned_two_week_prediction_interval.png"),
            n_points=len(first_week_plot_df),
            ghi_threshold=ghi_filter_wm2,
        )
    if {"GHI_pred_p10", "GHI_pred_p90"}.issubset(random_week_plot_df.columns):
        plot_prediction_interval_comparison(
            random_week_plot_df["datetime"].to_numpy(),
            random_week_plot_df["GHI_true"].to_numpy(),
            random_week_plot_df["GHI_pred"].to_numpy(),
            random_week_plot_df["GHI_pred_p10"].to_numpy(),
            random_week_plot_df["GHI_pred_p90"].to_numpy(),
            start=random_week_start,
            window_days=ONE_WEEK_DAYS,
            title=f"Fine-Tuned Moirai — Random Later One-Week GHI Prediction Interval ({plot_station_id})",
            save_path=str(RESULTS_DIR / "finetuned_random_two_week_prediction_interval.png"),
            n_points=len(random_week_plot_df),
            ghi_threshold=ghi_filter_wm2,
        )
    plot_4panel_evaluation(
        df["GHI_true"].to_numpy(),
        df["GHI_pred"].to_numpy(),
        hour_array=df["hour"].to_numpy(),
        title="Fine-Tuned Moirai — Validation GHI Evaluation",
        save_path=str(RESULTS_DIR / "finetuned_4panel.png"),
        n_points=len(df),
    )

    validation_report = df[[
        "station_id", "datetime", "hour", "lead_time_h", "forecast_start",
        "CAF_true",
        *([col for col in ["CAF_p10"] if col in df.columns]),
        "CAF_pred",
        *([col for col in ["CAF_p90"] if col in df.columns]),
        "clear_sky_ghi", "zenith_angle",
        "GHI_true",
        *([col for col in ["GHI_pred_p10"] if col in df.columns]),
        "GHI_pred",
        *([col for col in ["GHI_pred_p90"] if col in df.columns]),
    ]].copy()
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

    build_finetuned_pdf(str(RESULTS_DIR), ghi_filter_wm2=ghi_filter_wm2)

    print(f"\n  Plots saved → {RESULTS_DIR}/")
    print(f"  Validation CSV saved → {RESULTS_DIR / 'validation_report_data.csv'}")
    print(f"  Report saved → {RESULTS_DIR / 'evaluation_report.txt'}")
    print("Done.")


if __name__ == "__main__":
    main()
