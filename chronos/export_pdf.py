import os
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
ONE_WEEK_DAYS = 7
RANDOM_WEEK_SEED = 42


def _style_window_date_axis(ax, times_):
    times_ = pd.to_datetime(np.asarray(times_))
    if len(times_) == 0:
        return

    span_days = max((times_.max() - times_.min()) / np.timedelta64(1, "D"), 1)
    if span_days <= 10:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.tick_params(axis="x", labelsize=10, pad=6)
    for label in ax.get_xticklabels():
        label.set_rotation(28)
        label.set_horizontalalignment("right")
        label.set_verticalalignment("top")


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


def _finetuned_cover_page(
    pdf,
    metrics_model,
    station_label,
    row_count,
    generated_at,
    ghi_filter_wm2=20.0,
    cover_title="Amazon Chronos Fine-Tuned Validation Report",
    metric_label="Predicted GHI",
):
    plt.style.use(STYLE)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#f7f9fc")

    fig.text(0.5, 0.76, cover_title,
             ha="center", fontsize=28, fontweight="bold", color="#1a2e4a")
    fig.text(0.5, 0.69, f"Train: 2017-2018  |  Validation: 2019  |  Scope: {station_label}",
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
    col_labels = ["Metric", metric_label]
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

    fig.text(0.5, 0.12, f"Filter: GHI_true > {ghi_filter_wm2:.0f} W/m²  |  Validation rows: {row_count:,}",
             ha="center", fontsize=11, color="#555555")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _resolve_chronos_pdf_branding(metrics_json, out_pdf_filename):
    pipeline = str(metrics_json.get("pipeline", "")).lower()
    model_name = str(metrics_json.get("model", "")).lower()
    out_name = str(out_pdf_filename).lower()

    is_zero_shot = (
        "zero_shot" in pipeline
        or "zero-shot" in pipeline
        or "zero_shot" in out_name
        or "zero-shot" in out_name
        or ("amazon_chronos_zero_shot" in model_name)
    )

    if is_zero_shot:
        return {
            "cover_title": "Amazon Chronos Zero-Shot Validation Report",
            "metric_label": "Predicted GHI",
            "page_prefix": "Amazon Chronos Zero-Shot",
            "meta_title": "Amazon Chronos Zero-Shot Validation Report",
            "meta_author": "Chronos Pipeline",
            "meta_subject": "Direct GHI Zero-Shot Validation",
        }

    return {
        "cover_title": "Amazon Chronos Fine-Tuned Validation Report",
        "metric_label": "Predicted GHI",
        "page_prefix": "Amazon Chronos Fine-Tuned",
        "meta_title": "Amazon Chronos Fine-Tuned Validation Report",
        "meta_author": "Chronos Fine-Tuning Pipeline",
        "meta_subject": "Direct GHI Fine-Tuned Validation",
    }


def _metrics_table_page(pdf, title, rows, col_labels):
    plt.style.use(STYLE)
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#f7f9fc")

    fig.text(0.5, 0.92, title, ha="center", fontsize=22, fontweight="bold", color="#1a2e4a")
    ax_t = fig.add_axes([0.08, 0.12, 0.84, 0.72])
    ax_t.axis("off")
    tbl = ax_t.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a2e4a")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#dde8f5" if r % 2 == 0 else "#ffffff")

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
    n_points=None,
):
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times_, actual_,    label=actual_label,
            color="dodgerblue", linewidth=1.5, alpha=0.85)
    ax.plot(times_, predicted_, label=predicted_label,
            color="coral",      linewidth=1.5, linestyle="dashed")
    count = len(actual_) if n_points is None else n_points
    ax.set_title(f"{title}\nN={count}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(fontsize=11)
    _style_window_date_axis(ax, times_)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _two_week_comparison_page(
    pdf,
    times_,
    actual_,
    predicted_,
    title,
    window_days=14,
    actual_label="Measured (Ground Truth)",
    predicted_label="Predicted GHI",
    y_label="Irradiance (W/m²)",
    metrics=None,
    ghi_threshold=None,
):
    plt.style.use(STYLE)
    times_ = pd.to_datetime(times_)
    actual_ = np.asarray(actual_, dtype=float)
    predicted_ = np.asarray(predicted_, dtype=float)

    if len(times_) == 0:
        raise ValueError("No time points available for the comparison PDF page.")

    order = np.argsort(times_)
    times_ = times_[order]
    actual_ = actual_[order]
    predicted_ = predicted_[order]

    start_ts = times_[0]
    end_ts = start_ts + pd.Timedelta(days=window_days)
    mask = (times_ >= start_ts) & (times_ < end_ts)
    times_ = times_[mask]
    actual_ = actual_[mask]
    predicted_ = predicted_[mask]

    if len(times_) == 0:
        raise ValueError("No rows fell inside the comparison PDF window.")

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(times_, actual_, label=actual_label, color="dodgerblue", linewidth=1.8, alpha=0.9)
    ax.plot(times_, predicted_, label=predicted_label, color="coral", linewidth=1.8, linestyle="dashed")
    if ghi_threshold is not None:
        ax.axhline(
            ghi_threshold,
            label=f"{ghi_threshold:.0f} W/m² threshold",
            color="black",
            linewidth=2.5,
            linestyle=(0, (7, 2, 2, 2)),
            alpha=0.95,
        )

    under_mask = predicted_ < actual_
    if np.any(under_mask):
        ax.fill_between(
            times_,
            predicted_,
            actual_,
            where=under_mask,
            interpolate=True,
            color="crimson",
            alpha=0.18,
        )

    over_mask = predicted_ > actual_
    if np.any(over_mask):
        ax.fill_between(
            times_,
            actual_,
            predicted_,
            where=over_mask,
            interpolate=True,
            color="seagreen",
            alpha=0.10,
        )

    ax.set_title(f"{title}\nN={len(times_)}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(fontsize=11, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()

    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis("off")
    if metrics is not None:
        rows = [
            ("RMSE", f"{metrics['RMSE']:.2f}"),
            ("nRMSE", f"{metrics['nRMSE']:.2f} %"),
            ("MAE", f"{metrics['MAE']:.2f}"),
            ("MAPE", f"{metrics['MAPE']:.2f} %"),
            ("N", f"{metrics['N']:,}"),
        ]
        tbl = ax_tbl.table(
            cellText=rows,
            colLabels=["Metric", "Window"],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#1a2e4a")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#dde8f5" if r % 2 == 0 else "#ffffff")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _prediction_interval_page(
    pdf,
    times_,
    actual_,
    predicted_,
    predicted_p10_,
    predicted_p90_,
    title,
    window_days=7,
    y_label="Irradiance (W/m²)",
    metrics=None,
    ghi_threshold=None,
):
    plt.style.use(STYLE)
    times_ = pd.to_datetime(times_)
    actual_ = np.asarray(actual_, dtype=float)
    predicted_ = np.asarray(predicted_, dtype=float)
    predicted_p10_ = np.asarray(predicted_p10_, dtype=float)
    predicted_p90_ = np.asarray(predicted_p90_, dtype=float)

    if len(times_) == 0:
        raise ValueError("No time points available for the prediction interval PDF page.")

    order = np.argsort(times_)
    times_ = times_[order]
    actual_ = actual_[order]
    predicted_ = predicted_[order]
    predicted_p10_ = predicted_p10_[order]
    predicted_p90_ = predicted_p90_[order]

    start_ts = times_[0]
    end_ts = start_ts + pd.Timedelta(days=window_days)
    mask = (times_ >= start_ts) & (times_ < end_ts)
    times_ = times_[mask]
    actual_ = actual_[mask]
    predicted_ = predicted_[mask]
    predicted_p10_ = predicted_p10_[mask]
    predicted_p90_ = predicted_p90_[mask]

    if len(times_) == 0:
        raise ValueError("No rows fell inside the prediction interval PDF window.")

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(times_, predicted_p10_, label="Predicted GHI (P10)", color="#6a3d9a", linewidth=1.5, linestyle=":")
    ax.plot(times_, predicted_p90_, label="Predicted GHI (P90)", color="#1b9e77", linewidth=1.5, linestyle=":")
    ax.fill_between(
        times_,
        predicted_p10_,
        predicted_p90_,
        color="gray",
        alpha=0.18,
        label="Prediction Interval (P10-P90)",
    )
    if ghi_threshold is not None:
        ax.axhline(
            ghi_threshold,
            label=f"{ghi_threshold:.0f} W/m² threshold",
            color="black",
            linewidth=2.5,
            linestyle=(0, (7, 2, 2, 2)),
            alpha=0.95,
        )

    ax.set_title(f"{title}\nN={len(times_)}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.25)
    _style_window_date_axis(ax, times_)

    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis("off")
    if metrics is not None:
        rows = [
            ("RMSE", f"{metrics['RMSE']:.2f}"),
            ("nRMSE", f"{metrics['nRMSE']:.2f} %"),
            ("MAE", f"{metrics['MAE']:.2f}"),
            ("MAPE", f"{metrics['MAPE']:.2f} %"),
        ]
        table = ax_tbl.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.4)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1a2e4a")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#dde8f5" if row % 2 == 0 else "#ffffff")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _slice_window(df, start, days=ONE_WEEK_DAYS):
    end = pd.to_datetime(start) + pd.Timedelta(days=days)
    return df[(df["datetime"] >= start) & (df["datetime"] < end)].copy()


def _choose_random_later_week_start(df, first_start, days=ONE_WEEK_DAYS, seed=RANDOM_WEEK_SEED):
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
    n_points=None,
):
    plt.style.use(STYLE)
    fig = plt.figure(figsize=(14, 10))
    count = len(actual_) if n_points is None else n_points
    fig.suptitle(f"{title}\nN={count}", fontsize=16, fontweight="bold", y=0.99)
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
        _time_series_page(
            pdf, times, actual, pred_caf, "Method A (CAF-based) — Actual vs Predicted GHI", n_points=len(actual)
        )
        _four_panel_page(
            pdf, actual, pred_caf, errors_caf, hours, "Method A (CAF-based) — 4-Panel Evaluation", n_points=len(actual)
        )
        _time_series_page(
            pdf, times_d, actual_d, pred_direct, "Method B (Direct GHI) — Actual vs Predicted GHI", n_points=len(actual_d)
        )
        _four_panel_page(
            pdf, actual_d, pred_direct, errors_d, hours_d, "Method B (Direct GHI) — 4-Panel Evaluation", n_points=len(actual_d)
        )

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
            n_points=len(actual),
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
            n_points=len(actual),
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
            n_points=len(actual),
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
            n_points=len(actual),
        )

        meta = pdf.infodict()
        meta["Title"] = "CloudMapper Validation Report"
        meta["Author"] = "CloudMapper Pipeline"
        meta["Subject"] = "Phase 1 Cloud Cover Mapping Validation"

    print(f"PDF saved → {out_pdf}")


def build_finetuned_pdf(
    results_dir,
    ghi_filter_wm2=20.0,
    csv_filename="validation_report_data.csv",
    out_pdf_filename="finetuned_validation_report.pdf",
    metrics_filename="evaluation_metrics.json",
):
    results_dir = os.path.abspath(results_dir)
    csv_path = os.path.join(results_dir, csv_filename)
    out_pdf = os.path.join(results_dir, out_pdf_filename)
    metrics_path = os.path.join(results_dir, metrics_filename)

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df_metrics = df[df["GHI_true"] > ghi_filter_wm2].copy()
    if df_metrics.empty:
        raise ValueError(f"No rows passed the validation filter (GHI_true > {ghi_filter_wm2:.0f} W/m²).")

    metrics_json = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as fh:
            metrics_json = json.load(fh)
    branding = _resolve_chronos_pdf_branding(metrics_json, out_pdf_filename)

    actual = df_metrics["GHI_true"].values
    predicted = df_metrics["GHI_pred"].values
    metrics_model = calculate_metrics(actual, predicted, mape_threshold=ghi_filter_wm2)
    station_ids = sorted(df_metrics["station_id"].dropna().unique().tolist()) if "station_id" in df_metrics.columns else []
    station_label = f"{len(station_ids)} stations ({', '.join(station_ids)})" if station_ids else "unknown"
    plot_station_id = metrics_json.get("plot_station_id")
    available_station_ids = (
        set(df_metrics["station_id"].dropna().astype(str).unique())
        if "station_id" in df_metrics.columns
        else set()
    )
    # Reject a stale/foreign plot_station_id (e.g. left over in evaluation_metrics.json
    # from a previous run with a different station layout) and fall back to the
    # majority station actually present in the current validation CSV.
    if plot_station_id is not None and str(plot_station_id) not in available_station_ids:
        print(
            f"  build_finetuned_pdf: metrics JSON plot_station_id='{plot_station_id}' is not "
            f"present in the validation CSV; falling back to majority station."
        )
        plot_station_id = None
    if not plot_station_id and available_station_ids:
        plot_station_id = df_metrics["station_id"].value_counts().sort_values(ascending=False).index[0]

    plot_df = df[df["station_id"] == plot_station_id].copy() if plot_station_id else df.copy()
    plot_metrics_df = df_metrics[df_metrics["station_id"] == plot_station_id].copy() if plot_station_id else df_metrics.copy()
    if plot_df.empty:
        # Extreme safety net: no rows matched the resolved plot_station_id even after
        # the fallback above. Plot against the unfiltered dataframe to keep the PDF
        # generation resilient instead of crashing the whole evaluation.
        print(
            f"  build_finetuned_pdf: no rows for plot_station_id='{plot_station_id}' in the "
            f"validation CSV; using the full dataset for weekly windows."
        )
        plot_df = df.copy()
        plot_metrics_df = df_metrics.copy()

    first_week_start = plot_df["datetime"].min()
    first_week_df = _slice_window(plot_df, first_week_start)
    first_week_metrics_df = _slice_window(plot_metrics_df, first_week_start)
    first_week_metrics = None
    if not first_week_metrics_df.empty:
        first_week_metrics = calculate_metrics(
            first_week_metrics_df["GHI_true"].values,
            first_week_metrics_df["GHI_pred"].values,
            mape_threshold=ghi_filter_wm2,
        )

    random_week_start = _choose_random_later_week_start(plot_metrics_df, first_week_start)
    random_week_df = _slice_window(plot_df, random_week_start)
    random_week_metrics_df = _slice_window(plot_metrics_df, random_week_start)
    random_week_metrics = None
    if not random_week_metrics_df.empty:
        random_week_metrics = calculate_metrics(
            random_week_metrics_df["GHI_true"].values,
            random_week_metrics_df["GHI_pred"].values,
            mape_threshold=ghi_filter_wm2,
        )

    station_metric_rows = []
    if "station_id" in df_metrics.columns:
        for station_id, station_df in df_metrics.groupby("station_id", sort=True):
            station_metrics = calculate_metrics(
                station_df["GHI_true"].values,
                station_df["GHI_pred"].values,
                mape_threshold=ghi_filter_wm2,
            )
            station_metric_rows.append([
                station_id,
                f"{station_metrics['RMSE']:.2f}",
                f"{station_metrics['nRMSE']:.2f} %",
                f"{station_metrics['MAE']:.2f}",
                f"{station_metrics['MAPE']:.2f} %",
                f"{len(station_df):,}",
            ])

    with PdfPages(out_pdf) as pdf:
        _finetuned_cover_page(
            pdf,
            metrics_model=metrics_model,
            station_label=station_label,
            row_count=len(df_metrics),
            generated_at=datetime.now().strftime("%d %b %Y  %H:%M"),
            ghi_filter_wm2=ghi_filter_wm2,
            cover_title=branding["cover_title"],
            metric_label=branding["metric_label"],
        )
        if station_metric_rows:
            _metrics_table_page(
                pdf,
                title=f"Per-Station GHI Metrics (Filter > {ghi_filter_wm2:.0f} W/m²)",
                rows=station_metric_rows,
                col_labels=["Station", "RMSE", "nRMSE", "MAE", "MAPE", "Rows"],
            )
        _two_week_comparison_page(
            pdf,
            first_week_df["datetime"].values if not first_week_df.empty else plot_df["datetime"].values,
            first_week_df["GHI_true"].values if not first_week_df.empty else plot_df["GHI_true"].values,
            first_week_df["GHI_pred"].values if not first_week_df.empty else plot_df["GHI_pred"].values,
            f"{branding['page_prefix']} — One-Week Actual vs Predicted GHI ({plot_station_id})",
            window_days=ONE_WEEK_DAYS,
            metrics=first_week_metrics,
            ghi_threshold=ghi_filter_wm2,
        )
        _two_week_comparison_page(
            pdf,
            random_week_df["datetime"].values if not random_week_df.empty else plot_df["datetime"].values,
            random_week_df["GHI_true"].values if not random_week_df.empty else plot_df["GHI_true"].values,
            random_week_df["GHI_pred"].values if not random_week_df.empty else plot_df["GHI_pred"].values,
            f"{branding['page_prefix']} — Random Later One-Week Actual vs Predicted GHI ({plot_station_id})",
            window_days=ONE_WEEK_DAYS,
            metrics=random_week_metrics,
            ghi_threshold=ghi_filter_wm2,
        )
        if {"GHI_pred_p10", "GHI_pred_p90"}.issubset(first_week_df.columns):
            _prediction_interval_page(
                pdf,
                first_week_df["datetime"].values,
                first_week_df["GHI_true"].values,
                first_week_df["GHI_pred"].values,
                first_week_df["GHI_pred_p10"].values,
                first_week_df["GHI_pred_p90"].values,
                f"{branding['page_prefix']} — One-Week GHI Prediction Interval ({plot_station_id})",
                window_days=ONE_WEEK_DAYS,
                metrics=first_week_metrics,
                ghi_threshold=ghi_filter_wm2,
            )
        if {"GHI_pred_p10", "GHI_pred_p90"}.issubset(random_week_df.columns):
            _prediction_interval_page(
                pdf,
                random_week_df["datetime"].values,
                random_week_df["GHI_true"].values,
                random_week_df["GHI_pred"].values,
                random_week_df["GHI_pred_p10"].values,
                random_week_df["GHI_pred_p90"].values,
                f"{branding['page_prefix']} — Random Later One-Week GHI Prediction Interval ({plot_station_id})",
                window_days=ONE_WEEK_DAYS,
                metrics=random_week_metrics,
                ghi_threshold=ghi_filter_wm2,
            )
        _four_panel_page(
            pdf,
            df["GHI_true"].values,
            df["GHI_pred"].values,
            df["GHI_pred"].values - df["GHI_true"].values,
            df["hour"].values,
            f"{branding['page_prefix']} — 4-Panel Evaluation",
            n_points=len(df),
        )

        meta = pdf.infodict()
        meta["Title"] = branding["meta_title"]
        meta["Author"] = branding["meta_author"]
        meta["Subject"] = branding["meta_subject"]

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
