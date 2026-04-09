import os
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

from master_metrics import calculate_metrics

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
CSV     = os.path.join(BASE, "results", "persistence_ghi_predictions.csv")
OUT_PDF = os.path.join(BASE, "results", "persistence_results_report.pdf")

# ── Style ──────────────────────────────────────────────────────────────────────
STYLE = "seaborn-v0_8-darkgrid"


# ── Page helpers ───────────────────────────────────────────────────────────────

def _cover_page(pdf):
    plt.style.use(STYLE)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#f7f9fc")

    fig.text(0.5, 0.76, "Persistence Model — GHI Evaluation",
             ha="center", fontsize=30, fontweight="bold", color="#1a2e4a")
    fig.text(0.5, 0.69, "24-Hour Lag · Two Methods Compared",
             ha="center", fontsize=18, color="#3a5f8a")
    fig.text(0.5, 0.62, f"Generated: {datetime.now().strftime('%d %b %Y  %H:%M')}",
             ha="center", fontsize=12, color="#666666")

    ax_line = fig.add_axes([0.1, 0.59, 0.8, 0.004])
    ax_line.set_facecolor("#3a5f8a")
    ax_line.axis("off")

    # Side-by-side metrics table
    rows = [
        ("RMSE",  f"{metrics_caf['RMSE']:.2f} W/m²",  f"{metrics_direct['RMSE']:.2f} W/m²"),
        ("nRMSE", f"{metrics_caf['nRMSE']:.2f} %",    f"{metrics_direct['nRMSE']:.2f} %"),
        ("MAE",   f"{metrics_caf['MAE']:.2f} W/m²",   f"{metrics_direct['MAE']:.2f} W/m²"),
        ("MAPE",  f"{metrics_caf['MAPE']:.2f} %",     f"{metrics_direct['MAPE']:.2f} %"),
    ]
    col_labels = ["Metric", "CAF-based", "Direct GHI"]
    ax_t = fig.add_axes([0.2, 0.22, 0.6, 0.32])
    ax_t.axis("off")
    tbl = ax_t.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a2e4a")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#dde8f5" if r % 2 == 0 else "#ffffff")

    fig.text(0.5, 0.12,
             f"Filter: GHI_true > 20 W/m²  |  "
             f"CAF rows: {len(df_day):,}  |  Direct rows: {len(df_direct):,}",
             ha="center", fontsize=11, color="#555555")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _cloudmapper_cover_page(pdf, metrics_model, metrics_baseline, station_id, row_count, generated_at, output_label):
    plt.style.use(STYLE)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#f7f9fc")

    fig.text(0.5, 0.76, "CloudMapper Validation Report",
             ha="center", fontsize=28, fontweight="bold", color="#1a2e4a")
    fig.text(0.5, 0.69, f"Train: 2022  |  Validation: 2023  |  Station: {station_id}",
             ha="center", fontsize=16, color="#3a5f8a")
    fig.text(0.5, 0.62, f"Generated: {generated_at}",
             ha="center", fontsize=12, color="#666666")

    ax_line = fig.add_axes([0.1, 0.59, 0.8, 0.004])
    ax_line.set_facecolor("#3a5f8a")
    ax_line.axis("off")

    rows = [
        ("RMSE",  f"{metrics_model['RMSE']:.4f}", f"{metrics_baseline['RMSE']:.4f}"),
        ("nRMSE", f"{metrics_model['nRMSE']:.2f} %", f"{metrics_baseline['nRMSE']:.2f} %"),
        ("MAE",   f"{metrics_model['MAE']:.4f}", f"{metrics_baseline['MAE']:.4f}"),
        ("MAPE",  f"{metrics_model['MAPE']:.2f} %", f"{metrics_baseline['MAPE']:.2f} %"),
    ]
    col_labels = ["Metric", output_label, "ERA5 baseline"]
    ax_t = fig.add_axes([0.2, 0.22, 0.6, 0.32])
    ax_t.axis("off")
    tbl = ax_t.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a2e4a")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#dde8f5" if r % 2 == 0 else "#ffffff")

    fig.text(0.5, 0.12, f"Validation rows: {row_count:,}", ha="center", fontsize=11, color="#555555")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _finetuned_cover_page(pdf, metrics_model, station_id, row_count, generated_at):
    plt.style.use(STYLE)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#f7f9fc")

    fig.text(0.5, 0.76, "Fine-tuned Moirai Validation Report",
             ha="center", fontsize=28, fontweight="bold", color="#1a2e4a")
    fig.text(0.5, 0.69, f"Train: 2017-2018  |  Validation: 2019  |  Station: {station_id}",
             ha="center", fontsize=16, color="#3a5f8a")
    fig.text(0.5, 0.62, f"Generated: {generated_at}",
             ha="center", fontsize=12, color="#666666")

    ax_line = fig.add_axes([0.1, 0.59, 0.8, 0.004])
    ax_line.set_facecolor("#3a5f8a")
    ax_line.axis("off")

    rows = [
        ("RMSE", f"{metrics_model['RMSE']:.2f} W/m²"),
        ("nRMSE", f"{metrics_model['nRMSE']:.2f} %"),
        ("MAE", f"{metrics_model['MAE']:.2f} W/m²"),
        ("MAPE", f"{metrics_model['MAPE']:.2f} %"),
    ]
    col_labels = ["Metric", "Recovered GHI"]
    ax_t = fig.add_axes([0.25, 0.24, 0.5, 0.28])
    ax_t.axis("off")
    tbl = ax_t.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a2e4a")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#dde8f5" if r % 2 == 0 else "#ffffff")

    fig.text(0.5, 0.12, f"Filter: GHI_true > 20 W/m²  |  Validation rows: {row_count:,}",
             ha="center", fontsize=11, color="#555555")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _time_series_page(
    pdf,
    times_,
    actual_,
    predicted_,
    title,
    actual_label="Measured (Ground Truth)",
    predicted_label="Predicted GHI",
    y_label="Irradiance (W/m²)",
):
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times_, actual_,    label=actual_label,
            color="dodgerblue", linewidth=1.5, alpha=0.85)
    ax.plot(times_, predicted_, label=predicted_label,
            color="coral",      linewidth=1.5, linestyle="dashed")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(fontsize=11)
    fig.autofmt_xdate()
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _four_panel_page(
    pdf,
    actual_,
    predicted_,
    errors_,
    hours_,
    title,
    measured_label="Measured GHI",
    predicted_label="Predicted GHI",
    error_unit="W/m²",
    colorbar_label="Measured GHI (W/m²)",
    error_vs_title="Error vs Incoming Radiation",
):
    plt.style.use(STYLE)
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # 1 — Scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(actual_, predicted_, alpha=0.5, color="dodgerblue",
                edgecolor="k", s=30)
    lo = min(actual_.min(), predicted_.min())
    hi = max(actual_.max(), predicted_.max())
    ax1.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect (y=x)")
    ax1.set_title("Predicted vs Measured", fontsize=13, fontweight="bold")
    ax1.set_xlabel(measured_label, fontsize=11)
    ax1.set_ylabel(predicted_label, fontsize=11)
    ax1.legend(fontsize=10)

    # 2 — Error histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(errors_, bins=30, color="coral", edgecolor="black", alpha=0.8)
    ax2.axvline(0, color="r", linestyle="--", linewidth=2, label="Zero Error")
    mean_err = np.mean(errors_)
    ax2.axvline(mean_err, color="k", linewidth=2,
                label=f"Mean Error ({mean_err:.3f} {error_unit})")
    ax2.set_title("Error Distribution (Pred − Meas)", fontsize=13, fontweight="bold")
    ax2.set_xlabel(f"Error ({error_unit})", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.legend(fontsize=10)

    # 3 — Error vs measured
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(actual_, errors_, alpha=0.5, color="purple", edgecolor="w", s=30)
    ax3.axhline(0, color="r", linestyle="dotted", linewidth=2)
    ax3.set_title(error_vs_title, fontsize=13, fontweight="bold")
    ax3.set_xlabel(measured_label, fontsize=11)
    ax3.set_ylabel(f"Error ({error_unit})", fontsize=11)

    # 4 — Bias by hour
    ax4 = fig.add_subplot(gs[1, 1])
    sc = ax4.scatter(hours_, errors_, c=actual_, cmap="viridis",
                     alpha=0.7, s=30, edgecolor="w")
    ax4.axhline(0, color="r", linestyle="dotted", linewidth=2)
    ax4.set_title("Prediction Bias by Hour of Day", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Hour of Day", fontsize=11)
    ax4.set_ylabel(f"Error ({error_unit})", fontsize=11)
    cbar = plt.colorbar(sc, ax=ax4)
    cbar.set_label(colorbar_label, fontsize=10)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_persistence_pdf(csv_path=CSV, out_pdf=OUT_PDF):
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df_day = df[df["GHI_true"] > 20].copy()

    actual = df_day["GHI_true"].values
    pred_caf = df_day["GHI_pred_caf"].values
    hours = df_day["hour"].values
    times = df_day["datetime"].values
    errors_caf = pred_caf - actual

    df_direct = df_day[df_day["GHI_pred_direct"].notna()].copy()
    actual_d = df_direct["GHI_true"].values
    pred_direct = df_direct["GHI_pred_direct"].values
    hours_d = df_direct["hour"].values
    times_d = df_direct["datetime"].values
    errors_d = pred_direct - actual_d

    metrics_caf = calculate_metrics(actual, pred_caf)
    metrics_direct = calculate_metrics(actual_d, pred_direct)

    with PdfPages(out_pdf) as pdf:
        _cover_page(pdf)
        _time_series_page(pdf, times, actual, pred_caf, "Method A (CAF-based) — Actual vs Predicted GHI")
        _four_panel_page(pdf, actual, pred_caf, errors_caf, hours, "Method A (CAF-based) — 4-Panel Evaluation")
        _time_series_page(pdf, times_d, actual_d, pred_direct, "Method B (Direct GHI) — Actual vs Predicted GHI")
        _four_panel_page(pdf, actual_d, pred_direct, errors_d, hours_d, "Method B (Direct GHI) — 4-Panel Evaluation")

        meta = pdf.infodict()
        meta["Title"] = "Persistence Model — GHI Results Report"
        meta["Author"] = "Persistence Pipeline"
        meta["Subject"] = "Solar Irradiance Persistence Baseline Evaluation"

    print(f"PDF saved → {out_pdf}")


def build_cloudmapper_pdf(results_dir):
    results_dir = os.path.abspath(results_dir)
    report_csv_path = os.path.join(results_dir, "validation_report_data.csv")
    csv_path = report_csv_path if os.path.exists(report_csv_path) else os.path.join(results_dir, "validation_predictions.csv")
    config_path = os.path.join(BASE, "phase1_cloud_mapper", "checkpoints", "cloud_mapper_v6_config.json")
    out_pdf = os.path.join(results_dir, "cloudmapper_validation_report.pdf")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    actual = df["actual_cloud_cover"].values
    predicted = df["predicted_cloud_cover"].values
    if "baseline_total_cloud_cover" not in df.columns:
        raise KeyError(
            f"'baseline_total_cloud_cover' not found in {csv_path}. "
            "Re-run Phase 1 training to regenerate validation_report_data.csv."
        )
    baseline = df["baseline_total_cloud_cover"].values
    hours = df["hour"].values if "hour" in df.columns else df["datetime"].dt.hour.values
    times = df["datetime"].values
    station_id = df["station_id"].iloc[0] if "station_id" in df.columns and len(df) else "unknown"

    metrics_model = calculate_metrics(actual, predicted, mape_threshold=0.05)
    metrics_baseline = calculate_metrics(actual, baseline, mape_threshold=0.05)
    output_label = "CloudMapper"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="ascii") as fh:
            cfg = json.load(fh)
        station_id = cfg.get("selected_station", station_id)
        output_label = cfg.get("version", output_label)

    with PdfPages(out_pdf) as pdf:
        _cloudmapper_cover_page(
            pdf,
            metrics_model=metrics_model,
            metrics_baseline=metrics_baseline,
            station_id=station_id,
            row_count=len(df),
            generated_at=datetime.now().strftime("%d %b %Y  %H:%M"),
            output_label=output_label,
        )
        _time_series_page(
            pdf,
            times,
            actual,
            predicted,
            "CloudMapper — Actual vs Predicted Cloud Cover",
            actual_label="Actual Cloud Cover",
            predicted_label="Predicted Cloud Cover",
            y_label="Cloud Cover Fraction",
        )
        _four_panel_page(
            pdf,
            actual,
            predicted,
            predicted - actual,
            hours,
            "CloudMapper — 4-Panel Evaluation",
            measured_label="Actual Cloud Cover",
            predicted_label="Predicted Cloud Cover",
            error_unit="fraction",
            colorbar_label="Actual Cloud Cover",
            error_vs_title="Error vs Actual Cloud Cover",
        )
        _time_series_page(
            pdf,
            times,
            actual,
            baseline,
            "ERA5 Baseline — Actual vs Baseline Cloud Cover",
            actual_label="Actual Cloud Cover",
            predicted_label="ERA5 Baseline Cloud Cover",
            y_label="Cloud Cover Fraction",
        )
        _four_panel_page(
            pdf,
            actual,
            baseline,
            baseline - actual,
            hours,
            "ERA5 Baseline — 4-Panel Evaluation",
            measured_label="Actual Cloud Cover",
            predicted_label="ERA5 Baseline Cloud Cover",
            error_unit="fraction",
            colorbar_label="Actual Cloud Cover",
            error_vs_title="Error vs Actual Cloud Cover",
        )

        meta = pdf.infodict()
        meta["Title"] = "CloudMapper Validation Report"
        meta["Author"] = "CloudMapper Pipeline"
        meta["Subject"] = "Phase 1 Cloud Cover Mapping Validation"

    print(f"PDF saved → {out_pdf}")


def build_finetuned_pdf(results_dir):
    results_dir = os.path.abspath(results_dir)
    csv_path = os.path.join(results_dir, "validation_report_data.csv")
    out_pdf = os.path.join(results_dir, "finetuned_validation_report.pdf")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df_day = df[df["GHI_true"] > 20].copy()
    if df_day.empty:
        raise ValueError("No rows passed the validation filter (GHI_true > 20 W/m²).")

    actual = df_day["GHI_true"].values
    predicted = df_day["GHI_pred"].values
    hours = df_day["hour"].values
    times = df_day["datetime"].values
    errors = predicted - actual
    metrics_model = calculate_metrics(actual, predicted, mape_threshold=20.0)

    with PdfPages(out_pdf) as pdf:
        _finetuned_cover_page(
            pdf,
            metrics_model=metrics_model,
            station_id=df_day["station_id"].iloc[0] if "station_id" in df_day.columns else "unknown",
            row_count=len(df_day),
            generated_at=datetime.now().strftime("%d %b %Y  %H:%M"),
        )
        _time_series_page(pdf, times, actual, predicted, "Fine-tuned Moirai — Actual vs Predicted GHI")
        _four_panel_page(pdf, actual, predicted, errors, hours, "Fine-tuned Moirai — 4-Panel Evaluation")

        meta = pdf.infodict()
        meta["Title"] = "Fine-tuned Moirai Validation Report"
        meta["Author"] = "Moirai Fine-tuning Pipeline"
        meta["Subject"] = "Phase 2 GHI Validation"

    print(f"PDF saved → {out_pdf}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cloudmapper":
        target_dir = (
            sys.argv[2]
            if len(sys.argv) > 2
            else os.path.join(BASE, "phase1_cloud_mapper", "results", "cloud_mapper_v6_validation")
        )
        build_cloudmapper_pdf(target_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "finetuned":
        target_dir = (
            sys.argv[2]
            if len(sys.argv) > 2
            else os.path.join(BASE, "phase2_finetuning", "results")
        )
        build_finetuned_pdf(target_dir)
    else:
        build_persistence_pdf()
