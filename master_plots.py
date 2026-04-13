import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

def set_style():
    """Set the master plot style."""
    plt.style.use('seaborn-v0_8-darkgrid')


def style_window_date_axis(ax, times):
    """Apply cleaner date styling for short comparison windows."""
    times = pd.to_datetime(np.asarray(times))
    if len(times) == 0:
        return

    span_days = max((times.max() - times.min()) / np.timedelta64(1, "D"), 1)
    if span_days <= 10:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    ax.tick_params(axis="x", labelsize=10, pad=6)
    for label in ax.get_xticklabels():
        label.set_rotation(28)
        label.set_horizontalalignment("right")
        label.set_verticalalignment("top")

def plot_time_series(
    time_array,
    actual,
    predicted,
    title="Actual vs Predicted GHI",
    save_path=None,
    n_points=None,
    ghi_threshold=None,
):
    """
    Creates a standard line graph over time comparing actual vs predicted.
    
    Parameters:
    - time_array: datetime array or generic sequence for the X-axis
    - actual: array-like of ground truth values
    - predicted: array-like of predicted values
    - title: String title for the plot
    - save_path: Absolute path to save the generated image
    """
    set_style()
    plt.figure(figsize=(15, 6))
    
    plt.plot(time_array, actual, label='Measured (Ground Truth)', color='dodgerblue', linewidth=2, alpha=0.8)
    plt.plot(time_array, predicted, label='Predicted GHI', color='coral', linewidth=2, linestyle='dashed')
    if ghi_threshold is not None:
        plt.axhline(
            ghi_threshold,
            label=f"{ghi_threshold:.0f} W/m² threshold",
            color="black",
            linewidth=2.5,
            linestyle=(0, (7, 2, 2, 2)),
            alpha=0.95,
        )
    
    count = len(np.asarray(actual)) if n_points is None else n_points
    plt.title(f"{title}\nN={count}", fontsize=16, weight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Irradiance (W/m²)', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_two_week_comparison(
    time_array,
    actual,
    predicted,
    start=None,
    end=None,
    window_days=14,
    title="Windowed Actual vs Predicted",
    save_path=None,
    n_points=None,
    metrics=None,
    ghi_threshold=None,
):
    """
    Plot a windowed slice of actual vs predicted values.

    This is useful for spotting sustained underprediction or overprediction.

    Parameters:
    - time_array: datetime-like sequence for the X-axis
    - actual: array-like of ground truth values
    - predicted: array-like of predicted values
    - start: optional start timestamp for the slice
    - end: optional end timestamp for the slice
    - window_days: length of the slice when only start is provided or when no slice is provided
    - title: plot title
    - save_path: absolute path to save the generated image
    """
    set_style()

    times = pd.to_datetime(np.asarray(time_array))
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(times) == 0:
        raise ValueError("time_array is empty.")

    order = np.argsort(times)
    times = times[order]
    actual = actual[order]
    predicted = predicted[order]

    if start is not None:
        start_ts = pd.to_datetime(start)
        if end is not None:
            end_ts = pd.to_datetime(end)
        else:
            end_ts = start_ts + pd.Timedelta(days=window_days)
        mask = (times >= start_ts) & (times < end_ts)
    else:
        start_ts = times[0]
        end_ts = start_ts + pd.Timedelta(days=window_days)
        mask = (times >= start_ts) & (times < end_ts)

    times = times[mask]
    actual = actual[mask]
    predicted = predicted[mask]

    if len(times) == 0:
        raise ValueError("No data found in the requested comparison window.")

    if metrics is None:
        fig, ax = plt.subplots(figsize=(16, 6))
        ax_tbl = None
    else:
        fig, (ax, ax_tbl) = plt.subplots(
            2,
            1,
            figsize=(16, 8),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.22},
        )
    ax.plot(times, actual, label="Measured (Ground Truth)", color="dodgerblue", linewidth=2)
    ax.plot(times, predicted, label="Predicted", color="coral", linewidth=2, linestyle="--")
    if ghi_threshold is not None:
        ax.axhline(
            ghi_threshold,
            label=f"{ghi_threshold:.0f} W/m² threshold",
            color="black",
            linewidth=2.5,
            linestyle=(0, (7, 2, 2, 2)),
            alpha=0.95,
        )

    under_mask = predicted < actual
    if np.any(under_mask):
        ax.fill_between(
            times,
            predicted,
            actual,
            where=under_mask,
            interpolate=True,
            color="crimson",
            alpha=0.18,
            label="Underprediction",
        )

    over_mask = predicted > actual
    if np.any(over_mask):
        ax.fill_between(
            times,
            actual,
            predicted,
            where=over_mask,
            interpolate=True,
            color="seagreen",
            alpha=0.10,
            label="Overprediction",
        )

    count = len(times) if n_points is None else n_points
    ax.set_title(f"{title}\nN={count}", fontsize=16, weight='bold')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Irradiance (W/m²)", fontsize=12)
    ax.legend(fontsize=11, ncol=2)
    ax.grid(True, alpha=0.25)
    style_window_date_axis(ax, times)
    if ax_tbl is not None:
        ax_tbl.axis("off")
        rows = [
            ("N", f"{int(metrics.get('N', len(times)))}"),
            ("RMSE", f"{metrics['RMSE']:.2f} W/m²"),
            ("nRMSE", f"{metrics.get('nRMSE', metrics.get('nRMSE_pct', 0)):.2f} %"),
            ("MAE", f"{metrics['MAE']:.2f} W/m²"),
            ("MAPE", f"{metrics.get('MAPE', metrics.get('MAPE_pct', 0)):.2f} %"),
        ]
        table = ax_tbl.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.35)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1a2e4a")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#dde8f5" if row % 2 == 0 else "#ffffff")
    if ax_tbl is None:
        plt.tight_layout()
    else:
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.08, hspace=0.28)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_4panel_evaluation(
    actual,
    predicted,
    hour_array=None,
    title="Model Evaluation Analysis",
    save_path=None,
    n_points=None,
):
    """
    Generates the comprehensive 4-panel analysis grid used for clear sky verification.
    Includes: Scatter, Error Histogram, Error vs Measured, Error vs Time of Day.
    
    Parameters:
    - actual: array of true values
    - predicted: array of predicted values
    - hour_array: optional array of hour of day (float 0-24) mapped to the data points.
    - title: Main figure title
    - save_path: Absolute path to save the generated image
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    errors = predicted - actual
    
    set_style()
    fig = plt.figure(figsize=(18, 12))
    count = len(actual) if n_points is None else n_points
    fig.suptitle(f"{title}\nN={count}", fontsize=20, weight='bold', y=0.98)
    
    # 1. Scatter Plot: Predicted vs Actual
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(actual, predicted, alpha=0.6, color='dodgerblue', edgecolor='k', s=60)
    
    max_val = max(np.max(actual) if len(actual) else 0, np.max(predicted) if len(predicted) else 0)
    min_val = min(np.min(actual) if len(actual) else 0, np.min(predicted) if len(predicted) else 0)
    
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    ax1.set_title('Predicted vs Measured Ground Truth', fontsize=14, weight='bold')
    ax1.set_xlabel('Measured GHI (W/m²)', fontsize=12)
    ax1.set_ylabel('Predicted GHI (W/m²)', fontsize=12)
    ax1.legend()

    # 2. Histogram of Errors
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(errors, bins=30, color='coral', edgecolor='black', alpha=0.8)
    ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    if len(errors) > 0:
        mean_err = np.mean(errors)
        ax2.axvline(mean_err, color='k', linestyle='-', linewidth=2, label=f"Mean Error ({mean_err:.1f} W/m²)")
    ax2.set_title('Error Distribution (Predicted - Measured)', fontsize=14, weight='bold')
    ax2.set_xlabel('Error (W/m²)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()

    # 3. Error vs Initial Measured Value
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(actual, errors, alpha=0.6, color='purple', edgecolor='w', s=60)
    ax3.axhline(0, color='r', linestyle='dotted', linewidth=2)
    ax3.set_title('Error Magnitude vs Incoming Radiation', fontsize=14, weight='bold')
    ax3.set_xlabel('Measured Ground Truth GHI (W/m²)', fontsize=12)
    ax3.set_ylabel('Error (W/m²)', fontsize=12)

    # 4. Error vs Time of Day
    ax4 = plt.subplot(2, 2, 4)
    if hour_array is not None and len(hour_array) == len(actual):
        scatter = ax4.scatter(hour_array, errors, c=actual, cmap='viridis', alpha=0.8, s=60, edgecolor='w')
        ax4.axhline(0, color='r', linestyle='dotted', linewidth=2)
        ax4.set_title('Prediction Bias by Time of Day', fontsize=14, weight='bold')
        ax4.set_xlabel('Hour of Day', fontsize=12)
        ax4.set_ylabel('Error (W/m²)', fontsize=12)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Measured GHI (W/m²)')
    else:
        ax4.text(0.5, 0.5, 'Time of Day Mapping\nNot Provided or Mismatched', horizontalalignment='center', verticalalignment='center', fontsize=14, color='grey')
        ax4.set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"4-Panel Evaluation Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()
