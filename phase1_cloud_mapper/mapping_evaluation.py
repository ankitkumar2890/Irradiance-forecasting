from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from master_metrics import calculate_metrics, print_metrics
from master_plots import set_style
from time_utils import to_ist_series


def plot_mapping_time_series(
    time_array,
    actual,
    predicted,
    variable_name: str,
    save_path: str | Path,
) -> None:
    set_style()
    plt.figure(figsize=(15, 6))
    plt.plot(time_array, actual, label="ICON Ground Truth", color="dodgerblue", linewidth=2, alpha=0.8)
    plt.plot(time_array, predicted, label="Mapped Prediction", color="coral", linewidth=2, linestyle="dashed")
    plt.title(f"CloudMapper Validation: {variable_name}", fontsize=16, weight="bold")
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Cloud Fraction", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=12)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_mapping_4panel(
    actual,
    predicted,
    hour_array,
    variable_name: str,
    save_path: str | Path,
) -> None:
    actual = np.array(actual)
    predicted = np.array(predicted)
    errors = predicted - actual

    set_style()
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"CloudMapper Evaluation: {variable_name}", fontsize=20, weight="bold", y=0.98)

    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(actual, predicted, alpha=0.6, color="dodgerblue", edgecolor="k", s=60)
    ax1.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Prediction (y=x)")
    ax1.set_title("Predicted vs ICON Ground Truth", fontsize=14, weight="bold")
    ax1.set_xlabel("ICON Ground Truth", fontsize=12)
    ax1.set_ylabel("Mapped Prediction", fontsize=12)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(errors, bins=30, color="coral", edgecolor="black", alpha=0.8)
    ax2.axvline(0, color="r", linestyle="--", linewidth=2, label="Zero Error")
    if len(errors) > 0:
        mean_err = np.mean(errors)
        ax2.axvline(mean_err, color="k", linestyle="-", linewidth=2, label=f"Mean Error ({mean_err:.3f})")
    ax2.set_title("Error Distribution", fontsize=14, weight="bold")
    ax2.set_xlabel("Prediction Error", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(actual, errors, alpha=0.6, color="purple", edgecolor="w", s=60)
    ax3.axhline(0, color="r", linestyle="dotted", linewidth=2)
    ax3.set_title("Error vs ICON Ground Truth", fontsize=14, weight="bold")
    ax3.set_xlabel("ICON Ground Truth", fontsize=12)
    ax3.set_ylabel("Prediction Error", fontsize=12)

    ax4 = plt.subplot(2, 2, 4)
    if hour_array is not None and len(hour_array) == len(actual):
        scatter = ax4.scatter(hour_array, errors, c=actual, cmap="viridis", alpha=0.8, s=60, edgecolor="w")
        ax4.axhline(0, color="r", linestyle="dotted", linewidth=2)
        ax4.set_title("Prediction Bias by Hour", fontsize=14, weight="bold")
        ax4.set_xlabel("Hour of Day (IST)", fontsize=12)
        ax4.set_ylabel("Prediction Error", fontsize=12)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label("ICON Ground Truth")
    else:
        ax4.text(
            0.5,
            0.5,
            "Hour Mapping\nNot Provided or Mismatched",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
            color="grey",
        )
        ax4.set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_and_save_mapping_results(
    datetimes,
    actual,
    predicted,
    variable_names: list[str],
    output_dir: str | Path,
    station_ids=None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datetimes = to_ist_series(pd.Series(datetimes))
    station_ids = pd.Series(station_ids).reset_index(drop=True) if station_ids is not None else None
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    overall_metrics = calculate_metrics(actual.reshape(-1), predicted.reshape(-1), mape_threshold=0.05)
    print_metrics(overall_metrics, title="CLOUDMAPPER OVERALL VALIDATION", unit="cloud fraction")

    metrics_summary = {"overall": overall_metrics, "per_variable": {}}
    hours = datetimes.dt.hour.to_numpy()

    for idx, variable_name in enumerate(variable_names):
        channel_metrics = calculate_metrics(actual[:, idx], predicted[:, idx], mape_threshold=0.05)
        print_metrics(channel_metrics, title=f"{variable_name.upper()} VALIDATION", unit="cloud fraction")
        metrics_summary["per_variable"][variable_name] = channel_metrics

        channel_data = {
            "datetime": datetimes,
            "actual": actual[:, idx],
            "predicted": predicted[:, idx],
            "error": predicted[:, idx] - actual[:, idx],
        }
        if station_ids is not None:
            channel_data["station_id"] = station_ids
        channel_frame = pd.DataFrame(channel_data)
        channel_frame.to_csv(output_dir / f"{variable_name}_validation.csv", index=False)

        plot_mapping_time_series(
            datetimes,
            actual[:, idx],
            predicted[:, idx],
            variable_name=variable_name,
            save_path=output_dir / f"{variable_name}_timeseries.png",
        )
        plot_mapping_4panel(
            actual[:, idx],
            predicted[:, idx],
            hour_array=hours,
            variable_name=variable_name,
            save_path=output_dir / f"{variable_name}_evaluation.png",
        )

    summary_data = {
        "datetime": datetimes,
        **{f"actual_{name}": actual[:, idx] for idx, name in enumerate(variable_names)},
        **{f"predicted_{name}": predicted[:, idx] for idx, name in enumerate(variable_names)},
    }
    if station_ids is not None:
        summary_data["station_id"] = station_ids
    summary_frame = pd.DataFrame(summary_data)
    summary_frame.to_csv(output_dir / "validation_predictions.csv", index=False)

    with open(output_dir / "metrics_summary.json", "w", encoding="ascii") as fh:
        json.dump(metrics_summary, fh, indent=2)

    return metrics_summary
