"""Evaluate Direct GHI predictions — per-station and cluster-level metrics.

Phase 3: Unlike Phase 2, predictions are already in W/m² — no CAF recovery needed.
Computes metrics against measured GHI (w_ghr), persistence baseline, and generates
comprehensive plots.
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_DIR, MULTI_STATION_DOWNLOADS_DIR, RESULTS_DIR,
    PREDICTION_LENGTH, STATIONS, YEARS, CLUSTER_NAME,
)
from master_metrics import calculate_metrics, print_metrics
from master_plots import plot_4panel_evaluation, plot_two_week_comparison


DEFAULT_GHI_FILTER_WM2 = 20.0
ONE_WEEK_DAYS = 7
RANDOM_WEEK_SEED = 42


def normalize_ghi_to_hour_grid(df):
    """Average :30 timestamps to hour grid if needed."""
    df = df.copy()
    minutes = sorted(df["datetime"].dt.minute.dropna().unique().tolist())
    if set(minutes).issubset({0, 15, 30, 45}):
        df = (
            df.set_index("datetime")[["w_ghr"]]
            .apply(pd.to_numeric, errors="coerce")
            .resample("1h", label="right", closed="right")
            .mean()
            .reset_index()
            .sort_values("datetime")
        )
    elif minutes == [30]:
        shifted = df.copy()
        shifted["datetime"] = shifted["datetime"] - pd.Timedelta(minutes=30)
        df = shifted.sort_values("datetime")[["datetime", "w_ghr"]]
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


def summarize_subset(df, label, mape_threshold=DEFAULT_GHI_FILTER_WM2):
    """Return consistent metrics for one evaluation subset."""
    if df.empty:
        return None, None
    metrics = compute_metrics(df["GHI_true"].values, df["GHI_pred"].values, label)
    master = calculate_metrics(
        df["GHI_true"].to_numpy(),
        df["GHI_pred"].to_numpy(),
        mape_threshold=mape_threshold,
    )
    return metrics, master


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
    """Load measured GHI for all stations across all years."""
    frames = []
    for station in STATIONS:
        station_id = station["id"]
        for year in YEARS:
            ghi_file = MULTI_STATION_DOWNLOADS_DIR / station_id / f"ghi_{year}.csv"
            if not ghi_file.exists():
                print(f"  WARNING: Missing {ghi_file}")
                continue
            df = pd.read_csv(ghi_file)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = normalize_ghi_to_hour_grid(df)
            df["station_id"] = station_id
            frames.append(df[["station_id", "datetime", "w_ghr"]])
    if not frames:
        raise FileNotFoundError("No measured GHI files found.")
    ghi = pd.concat(frames, ignore_index=True)
    return ghi.drop_duplicates(subset=["station_id", "datetime"]).sort_values(["station_id", "datetime"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ghi-filter-wm2",
        type=float,
        default=DEFAULT_GHI_FILTER_WM2,
        help="Only evaluate rows with measured GHI above this threshold.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ghi_filter_wm2 = float(args.ghi_filter_wm2)
    pred_file = RESULTS_DIR / "direct_ghi_predictions.csv"
    if not pred_file.exists():
        print(f"Error: {pred_file} not found. Run 05_finetuned_inference.py first.")
        sys.exit(1)

    df = pd.read_csv(pred_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Load measured GHI for reference (the prediction file already has GHI_true)
    ghi = load_measured_ghi()

    # Cross-reference with measured GHI to ensure alignment
    # (pred file has GHI_true from scaled y_future, but let's also merge w_ghr for validation)
    df = df.merge(
        ghi.rename(columns={"w_ghr": "w_ghr_measured"}),
        on=["station_id", "datetime"],
        how="left",
    )

    # Use predicted GHI_true from the pipeline (derived from y_future)
    # Also load processed data for zenith_angle for reporting
    proc_path = DATASET_DIR / "processed_data_2017_2019.csv"
    if proc_path.exists():
        proc = pd.read_csv(proc_path, parse_dates=["datetime"])
        proc_cols = ["station_id", "datetime", "zenith_angle"]
        if "azimuth_angle" in proc.columns:
            proc_cols.append("azimuth_angle")
        proc = proc[proc_cols].drop_duplicates(subset=["station_id", "datetime"])
        df = df.merge(proc, on=["station_id", "datetime"], how="left")

    df = df.dropna(subset=["GHI_true", "GHI_pred"])
    if df.empty:
        print("No valid evaluation rows found after merging.")
        return

    # Ensure non-negative predictions
    df["GHI_pred"] = df["GHI_pred"].clip(lower=0)
    if "GHI_p10" in df.columns:
        df["GHI_p10"] = df["GHI_p10"].clip(lower=0)
    if "GHI_p90" in df.columns:
        df["GHI_p90"] = df["GHI_p90"].clip(lower=0)

    # ── Evaluation subsets ──
    all_hours_df = df.copy()
    solar_day_df = df[df["zenith_angle"] < 90].copy() if "zenith_angle" in df.columns else pd.DataFrame()
    validation_df = df[df["GHI_true"] > ghi_filter_wm2].copy()
    print(
        f"Total rows: {len(df)}  Solar-day rows: {len(solar_day_df)}  "
        f"Filtered rows (GHI_true > {ghi_filter_wm2:.0f}): {len(validation_df)}"
    )
    if validation_df.empty:
        print("No validation rows passed the GHI filter.")
        return

    # ── Choose a station for detailed plots ──
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

    # ── Persistence baseline (24h lag) ──
    persist_eval = ghi.copy()
    persist_eval["GHI_persist_24h"] = persist_eval.groupby("station_id")["w_ghr"].shift(PREDICTION_LENGTH)
    persist_eval = persist_eval[
        (persist_eval["w_ghr"] > ghi_filter_wm2) &
        persist_eval["GHI_persist_24h"].notna()
    ].copy()
    persist_rmse = float("nan")
    if not persist_eval.empty:
        persist_rmse = np.sqrt(
            np.mean((persist_eval["GHI_persist_24h"] - persist_eval["w_ghr"]) ** 2)
        )

    # ── Compute metrics ──
    all_hours_m, all_hours_master = summarize_subset(all_hours_df, "GHI_all_hours", ghi_filter_wm2)
    solar_day_m, solar_day_master = summarize_subset(solar_day_df, "GHI_solar_day", ghi_filter_wm2)
    ghi_m = compute_metrics(validation_df["GHI_true"].values, validation_df["GHI_pred"].values, "GHI_direct")
    ghi_m["Skill_vs_persist"] = 1.0 - ghi_m["RMSE"] / persist_rmse if persist_rmse > 0 else float("nan")
    first_week_ghi_m = compute_metrics(first_week_metrics_df["GHI_true"].values, first_week_metrics_df["GHI_pred"].values, "GHI_week_1")
    random_week_ghi_m = compute_metrics(random_week_metrics_df["GHI_true"].values, random_week_metrics_df["GHI_pred"].values, "GHI_random_week")
    master_ghi_m = calculate_metrics(
        validation_df["GHI_true"].to_numpy(),
        validation_df["GHI_pred"].to_numpy(),
        mape_threshold=ghi_filter_wm2,
    )

    # ── Print results ──
    print("\n" + "=" * 60)
    print(f"  DIRECT GHI FORECASTING — CLUSTER: {CLUSTER_NAME}")
    print("  METRIC VIEWS")
    print("=" * 60)
    print("\n  All-hours metrics (W/m²):")
    if all_hours_m:
        for k, v in all_hours_m.items():
            if k == "label":
                continue
            print(f"    {k:15s}: {v:.4f}" if isinstance(v, float) else f"    {k:15s}: {v}")
    if all_hours_master is not None:
        print_metrics(all_hours_master, title="MASTER GHI METRICS (ALL HOURS)")

    if solar_day_m:
        print("\n  Solar-day metrics (zenith_angle < 90°):")
        for k, v in solar_day_m.items():
            if k == "label":
                continue
            print(f"    {k:15s}: {v:.4f}" if isinstance(v, float) else f"    {k:15s}: {v}")
        print_metrics(solar_day_master, title="MASTER GHI METRICS (SOLAR DAY)")

    print(f"\n  Filtered validation metrics (GHI_true > {ghi_filter_wm2:.0f} W/m²):")
    for k, v in ghi_m.items():
        if k == "label":
            continue
        print(f"    {k:15s}: {v:.4f}" if isinstance(v, float) else f"    {k:15s}: {v}")
    print(f"\n  Persistence RMSE: {persist_rmse:.2f} W/m²")
    print(f"  Forecast Skill:   {ghi_m.get('Skill_vs_persist', 'N/A'):.4f}")
    print_metrics(
        master_ghi_m,
        title=f"MASTER GHI METRICS (FILTER: GHI_true > {ghi_filter_wm2:.0f})",
    )

    # ── Per-station metrics ──
    print("\n  Per-Station GHI RMSE:")
    station_metrics = {}
    for station_id, group in validation_df.groupby("station_id"):
        sm = compute_metrics(group["GHI_true"].values, group["GHI_pred"].values, station_id)
        station_metrics[station_id] = sm
        print(f"    {station_id:15s}: RMSE={sm['RMSE']:.2f}  MAE={sm['MAE']:.2f}  N={sm['N']}")

    print(f"\n  Plot station for weekly figures: {plot_station_id}")
    print("\n  One-Week Window Metrics:")
    print_metrics(first_week_ghi_m, title="ONE-WEEK GHI METRICS", unit="W/m²")
    print("\n  Random Later One-Week Window Metrics:")
    print_metrics(random_week_ghi_m, title="RANDOM ONE-WEEK GHI METRICS", unit="W/m²")

    # ── Horizon RMSE ──
    print("\n  RMSE by forecast hour:")
    horizon_rmse = {}
    if "lead_time_h" in validation_df.columns:
        for lead_time, group in validation_df.groupby("lead_time_h"):
            t = group["GHI_true"].to_numpy()
            p = group["GHI_pred"].to_numpy()
            horizon_rmse[int(lead_time)] = float(np.sqrt(np.mean((p - t) ** 2)))

    for h in [1, 6, 12, 18, 24]:
        if h in horizon_rmse:
            print(f"    +{h:2d}h: RMSE={horizon_rmse[h]:.2f} W/m²")

    # ── Solar intensity stratification ──
    print("\n  Stratified GHI RMSE:")
    for name, lo, hi in [
        ("High (>600)", 600, 1500),
        ("Medium (300-600)", 300, 600),
        ("Low (20-300)", ghi_filter_wm2, 300),
    ]:
        mask = (validation_df["GHI_true"] >= lo) & (validation_df["GHI_true"] < hi)
        sub = validation_df[mask]
        if len(sub) > 0:
            r = np.sqrt(np.mean((sub["GHI_pred"] - sub["GHI_true"]) ** 2))
            print(f"    {name:20s}: RMSE={r:.2f}  N={len(sub)}")

    # ── Save metrics ──
    all_metrics = {
        "pipeline": "phase3_direct_ghi",
        "cluster": CLUSTER_NAME,
        "filters": {"ghi_true_gt_wm2": ghi_filter_wm2},
        "ghi_all_hours": all_hours_m,
        "ghi_all_hours_master_metrics": {k: float(v) for k, v in all_hours_master.items()} if all_hours_master is not None else None,
        "ghi_solar_day": solar_day_m,
        "ghi_solar_day_master_metrics": {k: float(v) for k, v in solar_day_master.items()} if solar_day_master is not None else None,
        "ghi": ghi_m,
        "ghi_one_week": first_week_ghi_m,
        "ghi_random_week": random_week_ghi_m,
        "ghi_master_metrics": {k: float(v) for k, v in master_ghi_m.items()},
        "persistence_rmse": float(persist_rmse),
        "per_station": {k: v for k, v in station_metrics.items()},
        "horizon_rmse": horizon_rmse,
    }
    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # ── Text report ──
    report_lines = [
        f"DIRECT GHI FORECASTING EVALUATION — Cluster: {CLUSTER_NAME}",
        f"Filter: GHI_true > {ghi_filter_wm2:.0f} W/m²",
        f"Rows total: {len(all_hours_df)}",
        f"Rows solar-day: {len(solar_day_df)}",
        f"Rows filtered validation: {len(validation_df)}",
        "",
        "All-Hours Metrics (W/m²):",
        f"  RMSE: {all_hours_m['RMSE']:.2f}",
        f"  MAE: {all_hours_m['MAE']:.2f}",
        f"  nRMSE_pct: {all_hours_m['nRMSE_pct']:.2f}",
        f"  MAPE_pct: {all_hours_m['MAPE_pct']:.2f}",
        "",
    ]
    if all_hours_master is not None:
        report_lines.extend([
            "All-Hours Master Metrics:",
            f"  RMSE: {all_hours_master['RMSE']:.2f}",
            f"  nRMSE: {all_hours_master['nRMSE']:.2f}",
            f"  MAE: {all_hours_master['MAE']:.2f}",
            f"  MAPE: {all_hours_master['MAPE']:.2f}",
            "",
        ])
    if solar_day_m:
        report_lines.extend([
            "Solar-Day Metrics (zenith_angle < 90°):",
            f"  RMSE: {solar_day_m['RMSE']:.2f}",
            f"  MAE: {solar_day_m['MAE']:.2f}",
            f"  nRMSE_pct: {solar_day_m['nRMSE_pct']:.2f}",
            f"  MAPE_pct: {solar_day_m['MAPE_pct']:.2f}",
            "",
        ])
    if solar_day_master is not None:
        report_lines.extend([
            "Solar-Day Master Metrics:",
            f"  RMSE: {solar_day_master['RMSE']:.2f}",
            f"  nRMSE: {solar_day_master['nRMSE']:.2f}",
            f"  MAE: {solar_day_master['MAE']:.2f}",
            f"  MAPE: {solar_day_master['MAPE']:.2f}",
            "",
        ])
    report_lines.extend([
        f"Filtered Validation Metrics (GHI_true > {ghi_filter_wm2:.0f} W/m²):",
        "",
        f"  RMSE: {ghi_m['RMSE']:.2f}",
        f"  MAE: {ghi_m['MAE']:.2f}",
        f"  nRMSE_pct: {ghi_m['nRMSE_pct']:.2f}",
        f"  MAPE_pct: {ghi_m['MAPE_pct']:.2f}",
        f"  Skill_vs_persist: {ghi_m.get('Skill_vs_persist', float('nan')):.4f}",
        "",
        "Master Metrics:",
        f"  RMSE: {master_ghi_m['RMSE']:.2f}",
        f"  nRMSE: {master_ghi_m['nRMSE']:.2f}",
        f"  MAE: {master_ghi_m['MAE']:.2f}",
        f"  MAPE: {master_ghi_m['MAPE']:.2f}",
        "",
        "Per-Station RMSE:",
    ])
    for sid, sm in station_metrics.items():
        report_lines.append(f"  {sid}: RMSE={sm['RMSE']:.2f}  MAE={sm['MAE']:.2f}  N={sm['N']}")
    if horizon_rmse:
        report_lines.extend(["", "Horizon RMSE (W/m²):"])
        report_lines.extend(
            [f"  +{lead:02d}h: {rmse:.2f}" for lead, rmse in sorted(horizon_rmse.items())]
        )
    with open(RESULTS_DIR / "evaluation_report.txt", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    # ── Plots ──
    try:
        plot_two_week_comparison(
            first_week_plot_df["datetime"].to_numpy(),
            first_week_plot_df["GHI_true"].to_numpy(),
            first_week_plot_df["GHI_pred"].to_numpy(),
            start=first_week_start,
            window_days=ONE_WEEK_DAYS,
            title=f"Direct GHI — One-Week Measured vs Predicted ({plot_station_id})",
            save_path=str(RESULTS_DIR / "direct_ghi_one_week.png"),
            n_points=len(first_week_plot_df),
            metrics=first_week_ghi_m,
            ghi_threshold=ghi_filter_wm2,
        )
    except Exception as e:
        print(f"  Warning: Could not generate one-week plot: {e}")

    try:
        plot_two_week_comparison(
            random_week_plot_df["datetime"].to_numpy(),
            random_week_plot_df["GHI_true"].to_numpy(),
            random_week_plot_df["GHI_pred"].to_numpy(),
            start=random_week_start,
            window_days=ONE_WEEK_DAYS,
            title=f"Direct GHI — Random Later One-Week ({plot_station_id})",
            save_path=str(RESULTS_DIR / "direct_ghi_random_week.png"),
            n_points=len(random_week_plot_df),
            metrics=random_week_ghi_m,
            ghi_threshold=ghi_filter_wm2,
        )
    except Exception as e:
        print(f"  Warning: Could not generate random-week plot: {e}")

    try:
        plot_4panel_evaluation(
            validation_df["GHI_true"].to_numpy(),
            validation_df["GHI_pred"].to_numpy(),
            hour_array=validation_df["hour"].to_numpy(),
            title=f"Direct GHI — Cluster Evaluation ({CLUSTER_NAME})",
            save_path=str(RESULTS_DIR / "direct_ghi_4panel.png"),
            n_points=len(validation_df),
        )
    except Exception as e:
        print(f"  Warning: Could not generate 4-panel plot: {e}")

    # ── Per-station 4-panel ──
    for sid in station_metrics:
        try:
            sdf = validation_df[validation_df["station_id"] == sid]
            if len(sdf) < 10:
                continue
            plot_4panel_evaluation(
                sdf["GHI_true"].to_numpy(),
                sdf["GHI_pred"].to_numpy(),
                hour_array=sdf["hour"].to_numpy(),
                title=f"Direct GHI — {sid}",
                save_path=str(RESULTS_DIR / f"direct_ghi_4panel_{sid}.png"),
                n_points=len(sdf),
            )
        except Exception as e:
            print(f"  Warning: Could not generate 4-panel for {sid}: {e}")

    # ── Save validation report CSV ──
    report_cols = [
        "station_id", "datetime", "hour", "lead_time_h", "forecast_start",
        "GHI_true",
    ]
    for col in ["GHI_p10", "GHI_pred", "GHI_p90"]:
        if col in df.columns:
            report_cols.append(col)
    if "zenith_angle" in df.columns:
        report_cols.append("zenith_angle")

    validation_report = df[[c for c in report_cols if c in df.columns]].copy()
    validation_report.to_csv(RESULTS_DIR / "validation_report_data.csv", index=False)

    # ── PDF report (reuse project-level export_pdf) ──
    try:
        from export_pdf import build_finetuned_pdf
        build_finetuned_pdf(str(RESULTS_DIR), ghi_filter_wm2=ghi_filter_wm2)
    except Exception as e:
        print(f"  Warning: Could not generate PDF report: {e}")

    print(f"\n  Plots saved → {RESULTS_DIR}/")
    print(f"  Validation CSV → {RESULTS_DIR / 'validation_report_data.csv'}")
    print(f"  Report → {RESULTS_DIR / 'evaluation_report.txt'}")
    print(f"  Metrics → {RESULTS_DIR / 'evaluation_metrics.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
