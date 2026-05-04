"""Inference + evaluation for standalone Chronos Method 3."""

from __future__ import annotations

import importlib.util
import json
import site
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

from export_pdf import build_finetuned_pdf
from master_metrics import calculate_metrics
from master_plots import plot_4panel_evaluation, plot_two_week_comparison


def import_installed_chronos_package():
    for root_str in list(site.getsitepackages()) + [site.getusersitepackages()]:
        if not root_str:
            continue
        root = Path(root_str)
        init_path = root / "chronos" / "__init__.py"
        if not init_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            "_installed_chronos_pkg",
            init_path,
            submodule_search_locations=[str(init_path.parent)],
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError("Could not locate pip-installed chronos package.")


def _compute_metrics(y_true, y_pred, label=""):
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
    return {"label": label, "MAE": float(mae), "RMSE": float(rmse), "nRMSE_pct": float(nrmse), "MAPE_pct": float(mape), "N": int(len(y_true))}


def _slice_window(df, start, days=7):
    end = pd.to_datetime(start) + pd.Timedelta(days=days)
    return df[(df["datetime"] >= start) & (df["datetime"] < end)].copy()


def _choose_random_later_week_start(df, first_start, days=7, seed=42):
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


def _infer_cluster_name(station_ids) -> str:
    station_ids = sorted({str(station_id) for station_id in station_ids})
    if station_ids and all(station_id.startswith("box") for station_id in station_ids):
        prefix = station_ids[0].split("_r", 1)[0]
        return f"{prefix}_grid_{len(station_ids)}station"
    return "chronos_direct_ghi"


def run_inference_method3(
    *,
    cfg,
    dataset_dir,
    checkpoint_dir,
    results_dir,
    final_csv_dir,
    model_id,
    device,
    smoke_test=False,
):
    from peft import PeftModel

    dataset_dir = Path(dataset_dir)
    checkpoint_dir = Path(checkpoint_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    x_val = np.load(dataset_dir / "X_past_val.npy")
    y_val = np.load(dataset_dir / "y_future_val.npy")
    times_val = np.load(dataset_dir / "times_val.npy", allow_pickle=True).astype("datetime64[ns]")
    station_ids_val = np.load(dataset_dir / "station_ids_val.npy", allow_pickle=True)

    chronos_pkg = import_installed_chronos_package()
    pipeline = chronos_pkg.ChronosBoltPipeline.from_pretrained(model_id, device_map=str(device))
    adapter_name = "chronos_bolt_lora_adapter_smoke" if smoke_test else "chronos_bolt_lora_adapter"
    pipeline.model = PeftModel.from_pretrained(pipeline.model, checkpoint_dir / adapter_name)
    pipeline.model.to(device)
    pipeline.model.eval()

    quantiles = pipeline.model.quantiles.detach().cpu().numpy().tolist()
    q10_idx = quantiles.index(0.1) if 0.1 in quantiles else 0
    q50_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    q90_idx = quantiles.index(0.9) if 0.9 in quantiles else len(quantiles) - 1

    csv_rows = []
    all_preds = []
    all_true = []
    batch_size = int(getattr(cfg, "FT_BATCH_SIZE", 64))

    for start in range(0, len(x_val), batch_size):
        end = min(start + batch_size, len(x_val))
        context = torch.tensor(x_val[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            out = pipeline.model(context=context)
        preds = out.quantile_preds.detach().cpu().numpy()

        for batch_idx, window_idx in enumerate(range(start, end)):
            p10 = preds[batch_idx, q10_idx, : y_val.shape[1]]
            p50 = preds[batch_idx, q50_idx, : y_val.shape[1]]
            p90 = preds[batch_idx, q90_idx, : y_val.shape[1]]
            truth = y_val[window_idx]
            forecast_start = pd.Timestamp(times_val[window_idx, 0]) - pd.Timedelta(hours=1)

            all_preds.append(p50)
            all_true.append(truth)
            for h in range(len(truth)):
                ts = pd.Timestamp(times_val[window_idx, h])
                lead_time_h = int((ts - forecast_start) / pd.Timedelta(hours=1))
                csv_rows.append(
                    {
                        "station_id": str(station_ids_val[window_idx]),
                        "datetime": ts,
                        "hour": ts.hour,
                        "lead_time_h": lead_time_h,
                        "forecast_start": forecast_start,
                        "GHI_true": float(truth[h]),
                        "GHI_pred_p10": float(p10[h]),
                        "GHI_pred": float(p50[h]),
                        "GHI_pred_p90": float(p90[h]),
                    }
                )
        print(f"  Forecasted windows: {end}/{len(x_val)}")

    pred_df = pd.DataFrame(csv_rows).sort_values(["station_id", "datetime", "lead_time_h"]).reset_index(drop=True)
    pred_df.to_csv(results_dir / "finetuned_predictions.csv", index=False)

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    metrics = {
        "cluster": _infer_cluster_name(station_ids_val),
        "model": model_id,
        "GHI_RMSE_wm2": float(np.sqrt(np.mean((all_preds - all_true) ** 2))),
        "GHI_MAE_wm2": float(np.mean(np.abs(all_preds - all_true))),
    }
    with open(results_dir / "finetuned_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return pred_df


def build_report_method3(
    *,
    cfg,
    final_csv_dir,
    results_dir,
    validation_ghi_filter_wm2: float = 20.0,
):
    final_csv_dir = Path(final_csv_dir)
    results_dir = Path(results_dir)
    csvs = sorted(final_csv_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No source CSV found in {final_csv_dir}")
    source_df = pd.read_csv(csvs[0], parse_dates=["datetime"])

    pred_path = results_dir / "finetuned_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} not found. Run inference first.")

    pred_df = pd.read_csv(pred_path, parse_dates=["datetime", "forecast_start"])
    merge_cols = ["station_id", "datetime", cfg.ZENITH_COL]
    if cfg.CLEARSKY_GHI_COL in source_df.columns:
        merge_cols.append(cfg.CLEARSKY_GHI_COL)
    merged = pred_df.merge(
        source_df[merge_cols].drop_duplicates(subset=["station_id", "datetime"]),
        on=["station_id", "datetime"],
        how="left",
    )
    merged = merged.dropna(subset=["GHI_true", "GHI_pred"])
    merged["GHI_pred"] = merged["GHI_pred"].clip(lower=0)
    merged["GHI_pred_p10"] = merged["GHI_pred_p10"].clip(lower=0)
    merged["GHI_pred_p90"] = np.maximum(merged["GHI_pred_p90"].clip(lower=0), merged["GHI_pred_p10"])

    all_hours_df = merged.copy()
    solar_day_df = merged[merged[cfg.ZENITH_COL] < cfg.DAYLIGHT_ZENITH_DEG].copy()
    validation_df = merged[merged["GHI_true"] > float(validation_ghi_filter_wm2)].copy()
    if validation_df.empty:
        raise RuntimeError("No validation rows passed the GHI filter.")

    source_sorted = source_df.sort_values(["station_id", "datetime"]).copy()
    source_sorted["persist_24h"] = source_sorted.groupby("station_id")[cfg.MEASURED_GHI_COL].shift(cfg.PREDICTION_LENGTH)
    persist_eval = source_sorted[
        (source_sorted[cfg.MEASURED_GHI_COL] > float(validation_ghi_filter_wm2))
        & source_sorted["persist_24h"].notna()
    ].copy()
    persist_rmse = float(np.sqrt(np.mean((persist_eval["persist_24h"] - persist_eval[cfg.MEASURED_GHI_COL]) ** 2))) if not persist_eval.empty else float("nan")

    all_hours_m = _compute_metrics(all_hours_df["GHI_true"].values, all_hours_df["GHI_pred"].values, "GHI_all_hours")
    all_hours_master = calculate_metrics(all_hours_df["GHI_true"].to_numpy(), all_hours_df["GHI_pred"].to_numpy(), mape_threshold=validation_ghi_filter_wm2)
    solar_day_m = _compute_metrics(solar_day_df["GHI_true"].values, solar_day_df["GHI_pred"].values, "GHI_solar_day")
    solar_day_master = calculate_metrics(solar_day_df["GHI_true"].to_numpy(), solar_day_df["GHI_pred"].to_numpy(), mape_threshold=validation_ghi_filter_wm2)
    ghi_m = _compute_metrics(validation_df["GHI_true"].values, validation_df["GHI_pred"].values, "GHI_direct")
    ghi_m["Skill_vs_persist"] = 1.0 - ghi_m["RMSE"] / persist_rmse if persist_rmse > 0 else float("nan")
    master_ghi_m = calculate_metrics(validation_df["GHI_true"].to_numpy(), validation_df["GHI_pred"].to_numpy(), mape_threshold=validation_ghi_filter_wm2)

    station_metrics = {}
    for sid, group in validation_df.groupby("station_id"):
        station_metrics[sid] = _compute_metrics(group["GHI_true"].values, group["GHI_pred"].values, sid)

    horizon_rmse = {}
    for lead_time, group in validation_df.groupby("lead_time_h"):
        horizon_rmse[int(lead_time)] = float(np.sqrt(np.mean((group["GHI_pred"] - group["GHI_true"]) ** 2)))

    plot_station_id = validation_df["station_id"].value_counts().sort_values(ascending=False).index[0]
    plot_df = merged[merged["station_id"] == plot_station_id].copy()
    plot_validation_df = validation_df[validation_df["station_id"] == plot_station_id].copy()
    first_week_start = plot_df["datetime"].min()
    first_week_plot_df = _slice_window(plot_df, first_week_start)
    first_week_metrics_df = _slice_window(plot_validation_df, first_week_start)
    random_week_start = _choose_random_later_week_start(plot_validation_df, first_week_start)
    random_week_plot_df = _slice_window(plot_df, random_week_start)
    random_week_metrics_df = _slice_window(plot_validation_df, random_week_start)
    first_week_ghi_m = _compute_metrics(first_week_metrics_df["GHI_true"].values, first_week_metrics_df["GHI_pred"].values, "GHI_week_1")
    random_week_ghi_m = _compute_metrics(random_week_metrics_df["GHI_true"].values, random_week_metrics_df["GHI_pred"].values, "GHI_random_week")

    all_metrics = {
        "pipeline": "chronos_finetuned_method3",
        "cluster": _infer_cluster_name(validation_df["station_id"].tolist()),
        "filters": {"ghi_true_gt_wm2": float(validation_ghi_filter_wm2)},
        "ghi_all_hours": all_hours_m,
        "ghi_all_hours_master_metrics": {k: float(v) for k, v in all_hours_master.items()},
        "ghi_solar_day": solar_day_m,
        "ghi_solar_day_master_metrics": {k: float(v) for k, v in solar_day_master.items()},
        "ghi": ghi_m,
        "ghi_one_week": first_week_ghi_m,
        "ghi_random_week": random_week_ghi_m,
        "ghi_master_metrics": {k: float(v) for k, v in master_ghi_m.items()},
        "persistence_rmse": float(persist_rmse),
        "per_station": station_metrics,
        "horizon_rmse": horizon_rmse,
        "plot_station_id": plot_station_id,
    }
    with open(results_dir / "evaluation_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2, default=str)

    report_lines = [
        f"DIRECT GHI FORECASTING EVALUATION — Cluster: {all_metrics['cluster']}",
        "Model: Amazon Chronos fine-tuned",
        f"Filter: GHI_true > {validation_ghi_filter_wm2:.0f} W/m²",
        f"Rows total: {len(all_hours_df)}",
        f"Rows solar-day: {len(solar_day_df)}",
        f"Rows filtered validation: {len(validation_df)}",
        "",
        "Filtered Validation Metrics (GHI_true > 20 W/m²):",
        f"  RMSE: {ghi_m['RMSE']:.2f}",
        f"  MAE: {ghi_m['MAE']:.2f}",
        f"  nRMSE_pct: {ghi_m['nRMSE_pct']:.2f}",
        f"  MAPE_pct: {ghi_m['MAPE_pct']:.2f}",
        f"  Skill_vs_persist: {ghi_m['Skill_vs_persist']:.4f}",
        "",
        "Master Metrics:",
        f"  RMSE: {master_ghi_m['RMSE']:.2f}",
        f"  nRMSE: {master_ghi_m['nRMSE']:.2f}",
        f"  MAE: {master_ghi_m['MAE']:.2f}",
        f"  MAPE: {master_ghi_m['MAPE']:.2f}",
        "",
    ]
    with open(results_dir / "evaluation_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines) + "\n")

    merged.to_csv(results_dir / "validation_report_data.csv", index=False)

    plot_two_week_comparison(
        first_week_plot_df["datetime"].to_numpy(),
        first_week_plot_df["GHI_true"].to_numpy(),
        first_week_plot_df["GHI_pred"].to_numpy(),
        start=first_week_start,
        window_days=7,
        title=f"Chronos Fine-Tuned — One-Week Measured vs Predicted ({plot_station_id})",
        save_path=str(results_dir / "finetuned_one_week.png"),
        n_points=len(first_week_plot_df),
        metrics=first_week_ghi_m,
        ghi_threshold=validation_ghi_filter_wm2,
    )
    plot_two_week_comparison(
        random_week_plot_df["datetime"].to_numpy(),
        random_week_plot_df["GHI_true"].to_numpy(),
        random_week_plot_df["GHI_pred"].to_numpy(),
        start=random_week_start,
        window_days=7,
        title=f"Chronos Fine-Tuned — Random Later One-Week ({plot_station_id})",
        save_path=str(results_dir / "finetuned_random_week.png"),
        n_points=len(random_week_plot_df),
        metrics=random_week_ghi_m,
        ghi_threshold=validation_ghi_filter_wm2,
    )
    plot_4panel_evaluation(
        validation_df["GHI_true"].to_numpy(),
        validation_df["GHI_pred"].to_numpy(),
        hour_array=validation_df["hour"].to_numpy(),
        title=f"Chronos Fine-Tuned — Cluster Evaluation ({all_metrics['cluster']})",
        save_path=str(results_dir / "finetuned_4panel.png"),
        n_points=len(validation_df),
    )
    build_finetuned_pdf(str(results_dir), ghi_filter_wm2=validation_ghi_filter_wm2)
