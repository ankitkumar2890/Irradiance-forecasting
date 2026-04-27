"""Evaluate fine-tuned test predictions against measured GHI for the custom test pipeline."""
import sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_DIR, RESULTS_DIR, PREDICTION_LENGTH
from master_metrics import calculate_metrics, print_metrics
from master_plots import plot_4panel_evaluation, plot_two_week_comparison
from export_pdf import build_finetuned_pdf


VALIDATION_GHI_FILTER_WM2 = 20.0
ONE_WEEK_DAYS = 7
RANDOM_WEEK_SEED = 42


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
        df.loc[(df["datetime"] >= earliest_start) & (df["datetime"] <= latest_start), "datetime"]
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
    parser.add_argument("--ghi-filter-wm2", type=float, default=VALIDATION_GHI_FILTER_WM2)
    parser.add_argument(
        "--processed-file",
        type=Path,
        default=DATASET_DIR / "processed_test_from_persistence.csv",
        help="Processed test CSV used to build the test windows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ghi_filter_wm2 = float(args.ghi_filter_wm2)

    pred_file = RESULTS_DIR / "finetuned_predictions_test.csv"
    if not pred_file.exists():
        print(f"Error: {pred_file} not found. Run 05_finetuned_inference_test.py first.")
        sys.exit(1)

    df = pd.read_csv(pred_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    proc = pd.read_csv(args.processed_file)
    proc["datetime"] = pd.to_datetime(proc["datetime"])
    proc = proc[["station_id", "datetime", "clear_sky_ghi", "zenith_angle", "GHI"]].drop_duplicates(
        subset=["station_id", "datetime"]
    )
    df = (
        df.merge(proc, on=["station_id", "datetime"], how="left")
        .sort_values(["station_id", "datetime", "lead_time_h"] if "lead_time_h" in df.columns else ["station_id", "datetime"])
        .reset_index(drop=True)
    )
    df = df.dropna(subset=["clear_sky_ghi", "zenith_angle", "GHI"])
    if df.empty:
        print("No aligned evaluation rows were found after merging predictions and processed test features.")
        return

    df["GHI_true"] = df["GHI"]
    df["GHI_pred"] = df["CAF_pred"] * df["clear_sky_ghi"]

    validation_df = df[df["GHI_true"] > ghi_filter_wm2].copy()
    print(f"Total rows: {len(df)}  Filtered rows (GHI_true > {ghi_filter_wm2:.0f}): {len(validation_df)}")
    if validation_df.empty:
        print("No test rows passed the post-reconstruction GHI filter.")
        return

    plot_station_id = validation_df["station_id"].value_counts().sort_values(ascending=False).index[0]
    plot_df = df[df["station_id"] == plot_station_id].copy()
    plot_validation_df = validation_df[validation_df["station_id"] == plot_station_id].copy()
    first_week_start = plot_df["datetime"].min()
    first_week_plot_df = slice_window(plot_df, first_week_start)
    first_week_metrics_df = slice_window(plot_validation_df, first_week_start)
    random_week_start = choose_random_later_week_start(plot_validation_df, first_week_start)
    random_week_plot_df = slice_window(plot_df, random_week_start)
    random_week_metrics_df = slice_window(plot_validation_df, random_week_start)

    proc_full = proc.sort_values(["station_id", "datetime"]).reset_index(drop=True)
    proc_full["CAF_measured"] = np.where(
        proc_full["clear_sky_ghi"] > 1.0,
        (proc_full["GHI"] / proc_full["clear_sky_ghi"]).clip(0.0, 1.0),
        0.0,
    )
    proc_full["CAF_persist_24h"] = proc_full.groupby("station_id")["CAF_measured"].shift(PREDICTION_LENGTH)
    persist_eval = proc_full[(proc_full["GHI"] > ghi_filter_wm2) & proc_full["CAF_persist_24h"].notna()].copy()
    persist_rmse = float("nan")
    if not persist_eval.empty:
        persist_rmse = np.sqrt(np.mean((persist_eval["CAF_persist_24h"] - persist_eval["CAF_measured"]) ** 2))

    caf_m = compute_metrics(validation_df["CAF_true"].values, validation_df["CAF_pred"].values, "CAF_test")
    caf_m["Skill_vs_persist"] = 1.0 - caf_m["RMSE"] / persist_rmse if persist_rmse > 0 else float("nan")
    first_week_caf_m = compute_metrics(first_week_metrics_df["CAF_true"].values, first_week_metrics_df["CAF_pred"].values, "CAF_week_1")
    random_week_caf_m = compute_metrics(random_week_metrics_df["CAF_true"].values, random_week_metrics_df["CAF_pred"].values, "CAF_random_week")

    ghi_m = compute_metrics(validation_df["GHI_true"].values, validation_df["GHI_pred"].values, "GHI_test")
    first_week_ghi_m = compute_metrics(first_week_metrics_df["GHI_true"].values, first_week_metrics_df["GHI_pred"].values, "GHI_week_1")
    random_week_ghi_m = compute_metrics(random_week_metrics_df["GHI_true"].values, random_week_metrics_df["GHI_pred"].values, "GHI_random_week")
    master_ghi_m = calculate_metrics(validation_df["GHI_true"].to_numpy(), validation_df["GHI_pred"].to_numpy(), mape_threshold=ghi_filter_wm2)
    station_caf_metrics = build_station_metrics(validation_df, "CAF_true", "CAF_pred", "CAF")
    station_ghi_metrics = build_station_metrics(validation_df, "GHI_true", "GHI_pred", "GHI")

    print("\n" + "=" * 55)
    print("  FINE-TUNED MODEL — TEST DAYTIME METRICS")
    print("=" * 55)
    for label, m in [("CAF", caf_m), ("GHI (W/m2)", ghi_m)]:
        print(f"\n  {label}:")
        for k, v in m.items():
            if k == "label":
                continue
            print(f"    {k:15s}: {v:.4f}" if isinstance(v, float) else f"    {k:15s}: {v}")
    print(f"\n  Persistence RMSE: {persist_rmse:.4f}")
    print(f"  Forecast Skill:   {caf_m.get('Skill_vs_persist', 'N/A'):.4f}")
    print_metrics(master_ghi_m, title=f"MASTER GHI METRICS vs GHI (FILTER: measured GHI_true > {ghi_filter_wm2:.0f})")

    horizon_rmse = {}
    if "lead_time_h" in validation_df.columns:
        for lead_time, group in validation_df.groupby("lead_time_h"):
            t = group["CAF_true"].to_numpy()
            p = group["CAF_pred"].to_numpy()
            horizon_rmse[int(lead_time)] = float(np.sqrt(np.mean((p - t) ** 2)))

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
    with open(RESULTS_DIR / "evaluation_metrics_test.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    report_lines = [
        "FINE-TUNED MOIRAI TEST EVALUATION REPORT",
        f"Filter: measured GHI_true > {ghi_filter_wm2:.0f} W/m^2",
        f"Rows evaluated: {len(validation_df)}",
        f"Stations evaluated: {', '.join(sorted(validation_df['station_id'].dropna().unique().tolist()))}",
        "",
        f"CAF RMSE: {caf_m['RMSE']:.4f}",
        f"CAF MAE: {caf_m['MAE']:.4f}",
        f"GHI RMSE: {master_ghi_m['RMSE']:.4f}",
        f"GHI nRMSE: {master_ghi_m['nRMSE']:.4f}",
        f"GHI MAE: {master_ghi_m['MAE']:.4f}",
        f"GHI MAPE: {master_ghi_m['MAPE']:.4f}",
    ]
    with open(RESULTS_DIR / "evaluation_report_test.txt", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    plot_two_week_comparison(
        first_week_plot_df["datetime"].to_numpy(),
        first_week_plot_df["GHI_true"].to_numpy(),
        first_week_plot_df["GHI_pred"].to_numpy(),
        start=first_week_start,
        window_days=ONE_WEEK_DAYS,
        title=f"Fine-Tuned Moirai — One-Week Measured vs Predicted GHI ({plot_station_id})",
        save_path=str(RESULTS_DIR / "finetuned_one_week_comparison_test.png"),
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
        save_path=str(RESULTS_DIR / "finetuned_random_week_comparison_test.png"),
        n_points=len(random_week_plot_df),
        metrics=random_week_ghi_m,
        ghi_threshold=ghi_filter_wm2,
    )
    plot_4panel_evaluation(
        df["GHI_true"].to_numpy(),
        df["GHI_pred"].to_numpy(),
        hour_array=df["hour"].to_numpy(),
        title="Fine-Tuned Moirai — Test GHI Evaluation",
        save_path=str(RESULTS_DIR / "finetuned_4panel_test.png"),
        n_points=len(df),
    )

    test_report = df[[
        "station_id", "datetime", "hour", "lead_time_h", "forecast_start",
        "CAF_true", "CAF_pred", "clear_sky_ghi", "zenith_angle",
        "GHI_true", "GHI_pred",
    ]].copy()
    test_report.to_csv(RESULTS_DIR / "validation_report_data_test.csv", index=False)

    if horizon_rmse:
        fig3, ax4 = plt.subplots(figsize=(10, 5))
        hours = sorted(horizon_rmse.keys())
        rmses = [horizon_rmse[h] for h in hours]
        ax4.plot(hours, rmses, "o-", color="#1f77b4")
        ax4.set_xlabel("Forecast Lead Time (h)")
        ax4.set_ylabel("CAF RMSE")
        ax4.set_title("Forecast Degradation Curve")
        ax4.grid(True, alpha=0.3)
        fig3.savefig(RESULTS_DIR / "finetuned_horizon_rmse_test.png", dpi=150, bbox_inches="tight")
        plt.close()

    build_finetuned_pdf(
        str(RESULTS_DIR),
        ghi_filter_wm2=ghi_filter_wm2,
        csv_filename="validation_report_data_test.csv",
        out_pdf_filename="finetuned_test_report.pdf",
        metrics_filename="evaluation_metrics_test.json",
    )

    print(f"\n  Plots saved → {RESULTS_DIR}/")
    print(f"  Test CSV saved → {RESULTS_DIR / 'validation_report_data_test.csv'}")
    print(f"  Report saved → {RESULTS_DIR / 'evaluation_report_test.txt'}")
    print("Done.")


if __name__ == "__main__":
    main()
