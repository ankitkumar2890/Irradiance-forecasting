"""Standalone zero-shot Amazon Chronos baseline for direct GHI forecasting."""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.util
import json
import site
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from config import (  # noqa: E402
    BASE_DIR,
    CHECKPOINT_DIR,
    CLUSTER_NAME,
    DATASET_DIR,
    GHI_SCALE_FACTOR,
    MULTI_STATION_DOWNLOADS_DIR,
    PREDICTION_LENGTH,
    RESULTS_DIR,
    STATIONS,
    YEARS,
)
from export_pdf import build_finetuned_pdf  # noqa: E402
from master_metrics import calculate_metrics  # noqa: E402
from master_plots import plot_4panel_evaluation, plot_two_week_comparison  # noqa: E402

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GHI_FILTER_WM2 = 20.0
DEFAULT_MODEL_ID = "amazon/chronos-bolt-mini"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_SAMPLES = 64
ONE_WEEK_DAYS = 7
RANDOM_WEEK_SEED = 42


def import_installed_chronos_package():
    """Import the pip-installed Chronos package, not this local folder."""
    candidate_roots = []
    for path in site.getsitepackages():
        candidate_roots.append(Path(path))
    user_site = site.getusersitepackages()
    if user_site:
        candidate_roots.append(Path(user_site))

    for root in candidate_roots:
        init_path = root / "chronos" / "__init__.py"
        if not init_path.exists():
            continue
        module_name = "_installed_chronos_pkg"
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_path,
            submodule_search_locations=[str(init_path.parent)],
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    raise ImportError("Could not locate pip-installed chronos package in site-packages.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument(
        "--device-map",
        default="cuda" if _torch_cuda_available() else "cpu",
        help="Device map passed to Chronos from_pretrained.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype passed to Chronos from_pretrained when supported.",
    )
    parser.add_argument(
        "--ghi-filter-wm2",
        type=float,
        default=DEFAULT_GHI_FILTER_WM2,
        help="Only evaluate rows with measured GHI above this threshold.",
    )
    return parser.parse_args()


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "auto":
        return None
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def normalize_ghi_to_hour_grid(df: pd.DataFrame) -> pd.DataFrame:
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
    mape = (
        np.mean(np.abs((y_pred[day] - y_true[day]) / (y_true[day] + 1e-8))) * 100
        if day.any()
        else float("nan")
    )
    return {
        "label": label,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "nRMSE_pct": float(nrmse),
        "MAPE_pct": float(mape),
        "N": int(len(y_true)),
    }


def summarize_subset(df, label, mape_threshold=DEFAULT_GHI_FILTER_WM2):
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


def choose_random_later_week_start(
    df,
    first_start,
    days=ONE_WEEK_DAYS,
    seed=RANDOM_WEEK_SEED,
):
    latest_start = df["datetime"].max() - pd.Timedelta(days=days)
    earliest_start = pd.to_datetime(first_start) + pd.Timedelta(days=days * 2)
    candidate_days = (
        df.loc[
            (df["datetime"] >= earliest_start) & (df["datetime"] <= latest_start),
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
    proc_path = DATASET_DIR / "processed_data_2017_2019.csv"
    if proc_path.exists():
        proc = pd.read_csv(proc_path, parse_dates=["datetime"])
        required = {"station_id", "datetime", "w_ghr"}
        if required.issubset(proc.columns):
            ghi = proc[list(required)].copy()
            return ghi.drop_duplicates(subset=["station_id", "datetime"]).sort_values(
                ["station_id", "datetime"]
            )

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
    return ghi.drop_duplicates(subset=["station_id", "datetime"]).sort_values(
        ["station_id", "datetime"]
    )


def infer_cluster_name(station_ids) -> str:
    station_ids = sorted({str(station_id) for station_id in station_ids})
    if station_ids and all(station_id.startswith("box") for station_id in station_ids):
        prefix = station_ids[0].split("_r", 1)[0]
        return f"{prefix}_grid_{len(station_ids)}station"
    return CLUSTER_NAME


def load_chronos_pipeline(model_id: str, device_map: str, torch_dtype_name: str):
    try:
        chronos = import_installed_chronos_package()
    except ImportError as exc:
        raise RuntimeError(
            "chronos-forecasting is not installed in ./venv311. "
            "Install it in the Python 3.11 environment, then rerun this script."
        ) from exc

    dtype = resolve_torch_dtype(torch_dtype_name)
    load_kwargs = {"device_map": device_map}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    if "chronos-bolt" in model_id and hasattr(chronos, "ChronosBoltPipeline"):
        pipeline_cls = chronos.ChronosBoltPipeline
    elif "chronos-2" in model_id and hasattr(chronos, "Chronos2Pipeline"):
        pipeline_cls = chronos.Chronos2Pipeline
    elif hasattr(chronos, "ChronosPipeline"):
        pipeline_cls = chronos.ChronosPipeline
    elif hasattr(chronos, "BaseChronosPipeline"):
        pipeline_cls = chronos.BaseChronosPipeline
    else:
        raise RuntimeError(
            "Installed chronos-forecasting package does not expose a supported pipeline class."
        )

    return pipeline_cls.from_pretrained(model_id, **load_kwargs)


def forecast_batch(pipeline, context_batch: np.ndarray, prediction_length: int, num_samples: int):
    import torch

    context_tensor = torch.tensor(context_batch, dtype=torch.float32)
    with torch.inference_mode():
        if pipeline.__class__.__name__ == "ChronosBoltPipeline":
            forecast = pipeline.predict(
                context_tensor,
                prediction_length=prediction_length,
            )
        else:
            forecast = pipeline.predict(
                context_tensor,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )

    if hasattr(forecast, "detach"):
        forecast = forecast.detach().cpu()
    forecast = np.asarray(forecast, dtype=np.float32)

    if forecast.ndim == 3:
        if pipeline.__class__.__name__ == "ChronosBoltPipeline":
            quantiles = np.asarray(getattr(pipeline, "quantiles", []), dtype=np.float32)
            if quantiles.size == 0:
                raise RuntimeError("ChronosBoltPipeline did not expose quantile levels.")
            point = forecast[:, int(np.argmin(np.abs(quantiles - 0.5))), :]
            p10 = forecast[:, int(np.argmin(np.abs(quantiles - 0.1))), :]
            p90 = forecast[:, int(np.argmin(np.abs(quantiles - 0.9))), :]
        else:
            point = np.quantile(forecast, 0.5, axis=1)
            p10 = np.quantile(forecast, 0.1, axis=1)
            p90 = np.quantile(forecast, 0.9, axis=1)
    elif forecast.ndim == 2:
        point = forecast
        p10 = forecast
        p90 = forecast
    else:
        raise RuntimeError(f"Unexpected forecast shape from Chronos: {forecast.shape}")

    point = np.clip(point, 0.0, None)
    p10 = np.clip(p10, 0.0, None)
    p90 = np.maximum(np.clip(p90, 0.0, None), p10)
    return point, p10, p90


def evaluate_prediction_frame(df_pred: pd.DataFrame, ghi_filter_wm2: float, cluster_name: str):
    ghi = load_measured_ghi()
    df = df_pred.copy()
    df = df.merge(
        ghi.rename(columns={"w_ghr": "w_ghr_measured"}),
        on=["station_id", "datetime"],
        how="left",
    )

    proc_path = DATASET_DIR / "processed_data_2017_2019.csv"
    if proc_path.exists() and "zenith_angle" not in df.columns:
        proc = pd.read_csv(proc_path, parse_dates=["datetime"])
        proc_cols = ["station_id", "datetime", "zenith_angle"]
        if "azimuth_angle" in proc.columns:
            proc_cols.append("azimuth_angle")
        proc = proc[proc_cols].drop_duplicates(subset=["station_id", "datetime"])
        df = df.merge(proc, on=["station_id", "datetime"], how="left")

    df = df.dropna(subset=["GHI_true", "GHI_pred"])
    df["GHI_pred"] = df["GHI_pred"].clip(lower=0)
    if "GHI_p10" in df.columns:
        df["GHI_p10"] = df["GHI_p10"].clip(lower=0)
    if "GHI_p90" in df.columns:
        df["GHI_p90"] = df["GHI_p90"].clip(lower=0)

    all_hours_df = df.copy()
    solar_day_df = (
        df[df["zenith_angle"] < 90].copy() if "zenith_angle" in df.columns else pd.DataFrame()
    )
    validation_df = df[df["GHI_true"] > ghi_filter_wm2].copy()
    if validation_df.empty:
        raise RuntimeError("No validation rows passed the GHI filter.")

    persist_eval = ghi.copy()
    persist_eval["GHI_persist_24h"] = persist_eval.groupby("station_id")["w_ghr"].shift(
        PREDICTION_LENGTH
    )
    persist_eval = persist_eval[
        (persist_eval["w_ghr"] > ghi_filter_wm2) & persist_eval["GHI_persist_24h"].notna()
    ].copy()
    persist_rmse = float("nan")
    if not persist_eval.empty:
        persist_rmse = float(
            np.sqrt(np.mean((persist_eval["GHI_persist_24h"] - persist_eval["w_ghr"]) ** 2))
        )

    all_hours_m, all_hours_master = summarize_subset(
        all_hours_df, "GHI_all_hours", ghi_filter_wm2
    )
    solar_day_m, solar_day_master = summarize_subset(
        solar_day_df, "GHI_solar_day", ghi_filter_wm2
    )
    ghi_m = compute_metrics(validation_df["GHI_true"].values, validation_df["GHI_pred"].values, "GHI_direct")
    ghi_m["Skill_vs_persist"] = (
        1.0 - ghi_m["RMSE"] / persist_rmse if persist_rmse > 0 else float("nan")
    )

    plot_station_id = validation_df["station_id"].value_counts().sort_values(
        ascending=False
    ).index[0]
    plot_df = df[df["station_id"] == plot_station_id].copy()
    plot_validation_df = validation_df[validation_df["station_id"] == plot_station_id].copy()
    first_week_start = plot_df["datetime"].min()
    first_week_plot_df = slice_window(plot_df, first_week_start)
    first_week_metrics_df = slice_window(plot_validation_df, first_week_start)
    random_week_start = choose_random_later_week_start(plot_validation_df, first_week_start)
    random_week_plot_df = slice_window(plot_df, random_week_start)
    random_week_metrics_df = slice_window(plot_validation_df, random_week_start)

    first_week_ghi_m = compute_metrics(
        first_week_metrics_df["GHI_true"].values,
        first_week_metrics_df["GHI_pred"].values,
        "GHI_week_1",
    )
    random_week_ghi_m = compute_metrics(
        random_week_metrics_df["GHI_true"].values,
        random_week_metrics_df["GHI_pred"].values,
        "GHI_random_week",
    )
    master_ghi_m = calculate_metrics(
        validation_df["GHI_true"].to_numpy(),
        validation_df["GHI_pred"].to_numpy(),
        mape_threshold=ghi_filter_wm2,
    )

    station_metrics = {}
    for station_id, group in validation_df.groupby("station_id"):
        station_metrics[station_id] = compute_metrics(
            group["GHI_true"].values,
            group["GHI_pred"].values,
            station_id,
        )

    horizon_rmse = {}
    for lead_time, group in validation_df.groupby("lead_time_h"):
        t = group["GHI_true"].to_numpy()
        p = group["GHI_pred"].to_numpy()
        horizon_rmse[int(lead_time)] = float(np.sqrt(np.mean((p - t) ** 2)))

    all_metrics = {
        "pipeline": "chronos_zero_shot",
        "cluster": cluster_name,
        "model": "amazon_chronos_zero_shot",
        "plot_station_id": plot_station_id,
        "filters": {"ghi_true_gt_wm2": ghi_filter_wm2},
        "ghi_all_hours": all_hours_m,
        "ghi_all_hours_master_metrics": (
            {k: float(v) for k, v in all_hours_master.items()}
            if all_hours_master is not None
            else None
        ),
        "ghi_solar_day": solar_day_m,
        "ghi_solar_day_master_metrics": (
            {k: float(v) for k, v in solar_day_master.items()}
            if solar_day_master is not None
            else None
        ),
        "ghi": ghi_m,
        "ghi_one_week": first_week_ghi_m,
        "ghi_random_week": random_week_ghi_m,
        "ghi_master_metrics": {k: float(v) for k, v in master_ghi_m.items()},
        "persistence_rmse": float(persist_rmse),
        "per_station": station_metrics,
        "horizon_rmse": horizon_rmse,
    }

    report_lines = [
        f"DIRECT GHI FORECASTING EVALUATION — Cluster: {cluster_name}",
        "Model: Amazon Chronos zero-shot",
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
        report_lines.extend(
            [
                "All-Hours Master Metrics:",
                f"  RMSE: {all_hours_master['RMSE']:.2f}",
                f"  nRMSE: {all_hours_master['nRMSE']:.2f}",
                f"  MAE: {all_hours_master['MAE']:.2f}",
                f"  MAPE: {all_hours_master['MAPE']:.2f}",
                "",
            ]
        )
    if solar_day_m:
        report_lines.extend(
            [
                "Solar-Day Metrics (zenith_angle < 90°):",
                f"  RMSE: {solar_day_m['RMSE']:.2f}",
                f"  MAE: {solar_day_m['MAE']:.2f}",
                f"  nRMSE_pct: {solar_day_m['nRMSE_pct']:.2f}",
                f"  MAPE_pct: {solar_day_m['MAPE_pct']:.2f}",
                "",
            ]
        )
    if solar_day_master is not None:
        report_lines.extend(
            [
                "Solar-Day Master Metrics:",
                f"  RMSE: {solar_day_master['RMSE']:.2f}",
                f"  nRMSE: {solar_day_master['nRMSE']:.2f}",
                f"  MAE: {solar_day_master['MAE']:.2f}",
                f"  MAPE: {solar_day_master['MAPE']:.2f}",
                "",
            ]
        )
    report_lines.extend(
        [
            f"Filtered Validation Metrics (GHI_true > {ghi_filter_wm2:.0f} W/m²):",
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
        ]
    )
    for sid, sm in station_metrics.items():
        report_lines.append(f"  {sid}: RMSE={sm['RMSE']:.2f}  MAE={sm['MAE']:.2f}  N={sm['N']}")
    if horizon_rmse:
        report_lines.extend(["", "Horizon RMSE (W/m²):"])
        report_lines.extend([f"  +{lead:02d}h: {rmse:.2f}" for lead, rmse in sorted(horizon_rmse.items())])

    return (
        df,
        all_metrics,
        report_lines,
        plot_station_id,
        first_week_start,
        first_week_plot_df,
        first_week_ghi_m,
        random_week_start,
        random_week_plot_df,
        random_week_ghi_m,
        validation_df,
        station_metrics,
    )


def save_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def main():
    args = parse_args()

    X_past = np.load(DATASET_DIR / "X_past_val.npy")
    y_future = np.load(DATASET_DIR / "y_future_val.npy")
    times = np.load(DATASET_DIR / "times_val.npy", allow_pickle=True).astype("datetime64[ns]")
    station_ids = np.load(DATASET_DIR / "station_ids_val.npy", allow_pickle=True)
    cluster_name = infer_cluster_name(station_ids)

    context_ghi = X_past[:, :, 0].astype(np.float32)
    y_true_all = y_future.astype(np.float32) * GHI_SCALE_FACTOR

    print(f"Loading Amazon Chronos zero-shot model: {args.model_id}")
    print(f"Device map: {args.device_map}  dtype: {args.torch_dtype}")
    pipeline = load_chronos_pipeline(args.model_id, args.device_map, args.torch_dtype)

    csv_rows = []
    all_preds = []
    all_true = []
    total = len(context_ghi)

    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch_context = context_ghi[start:end]
        point_batch, p10_batch, p90_batch = forecast_batch(
            pipeline,
            batch_context,
            prediction_length=PREDICTION_LENGTH,
            num_samples=args.num_samples,
        )

        for batch_idx, window_idx in enumerate(range(start, end)):
            y_true_wm2 = y_true_all[window_idx]
            forecast_start = pd.Timestamp(times[window_idx, 0]) - pd.Timedelta(hours=1)

            all_preds.append(point_batch[batch_idx])
            all_true.append(y_true_wm2)

            for h in range(PREDICTION_LENGTH):
                ts = pd.Timestamp(times[window_idx, h])
                lead_time_h = int((ts - forecast_start) / pd.Timedelta(hours=1))
                csv_rows.append(
                    {
                        "station_id": str(station_ids[window_idx]),
                        "datetime": ts,
                        "hour": ts.hour,
                        "lead_time_h": lead_time_h,
                        "forecast_start": forecast_start,
                        "GHI_true": float(y_true_wm2[h]),
                        "GHI_p10": float(p10_batch[batch_idx, h]),
                        "GHI_pred": float(point_batch[batch_idx, h]),
                        "GHI_p90": float(p90_batch[batch_idx, h]),
                    }
                )

        print(f"  Forecasted windows: {end}/{total}")

    pred_df = pd.DataFrame(csv_rows).sort_values(
        ["station_id", "datetime", "lead_time_h"]
    ).reset_index(drop=True)

    predictions_path = RESULTS_DIR / "chronos_zero_shot_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    rmse = float(np.sqrt(np.mean((all_preds - all_true) ** 2)))
    mae = float(np.mean(np.abs(all_preds - all_true)))
    mean_true = float(np.mean(all_true[all_true > 0]))
    nrmse = float((rmse / mean_true * 100) if mean_true > 0 else float("nan"))

    raw_metrics = {
        "cluster": cluster_name,
        "model": args.model_id,
        "GHI_RMSE_wm2": rmse,
        "GHI_MAE_wm2": mae,
        "GHI_nRMSE_pct": nrmse,
    }
    save_json(RESULTS_DIR / "chronos_zero_shot_metrics.json", raw_metrics)

    (
        evaluated_df,
        all_metrics,
        report_lines,
        plot_station_id,
        first_week_start,
        first_week_plot_df,
        first_week_ghi_m,
        random_week_start,
        random_week_plot_df,
        random_week_ghi_m,
        validation_df,
        station_metrics,
    ) = evaluate_prediction_frame(pred_df, args.ghi_filter_wm2, cluster_name)
    all_metrics["model_id"] = args.model_id
    all_metrics["num_samples"] = int(args.num_samples)

    prefixed_csv = RESULTS_DIR / "chronos_zero_shot_validation_report_data.csv"
    generic_csv = RESULTS_DIR / "validation_report_data.csv"
    evaluated_df.to_csv(prefixed_csv, index=False)
    evaluated_df.to_csv(generic_csv, index=False)

    prefixed_metrics = RESULTS_DIR / "chronos_zero_shot_evaluation_metrics.json"
    generic_metrics = RESULTS_DIR / "evaluation_metrics.json"
    save_json(prefixed_metrics, all_metrics)
    save_json(generic_metrics, all_metrics)

    report_text = "\n".join(report_lines) + "\n"
    prefixed_report = RESULTS_DIR / "chronos_zero_shot_evaluation_report.txt"
    generic_report = RESULTS_DIR / "evaluation_report.txt"
    prefixed_report.write_text(report_text, encoding="utf-8")
    generic_report.write_text(report_text, encoding="utf-8")

    try:
        plot_two_week_comparison(
            first_week_plot_df["datetime"].to_numpy(),
            first_week_plot_df["GHI_true"].to_numpy(),
            first_week_plot_df["GHI_pred"].to_numpy(),
            start=first_week_start,
            window_days=ONE_WEEK_DAYS,
            title=f"Chronos Zero-Shot — One-Week Measured vs Predicted ({plot_station_id})",
            save_path=str(RESULTS_DIR / "chronos_zero_shot_one_week.png"),
            n_points=len(first_week_plot_df),
            metrics=first_week_ghi_m,
            ghi_threshold=args.ghi_filter_wm2,
        )
    except Exception as exc:
        print(f"  Warning: Could not generate one-week plot: {exc}")

    try:
        plot_two_week_comparison(
            random_week_plot_df["datetime"].to_numpy(),
            random_week_plot_df["GHI_true"].to_numpy(),
            random_week_plot_df["GHI_pred"].to_numpy(),
            start=random_week_start,
            window_days=ONE_WEEK_DAYS,
            title=f"Chronos Zero-Shot — Random Later One-Week ({plot_station_id})",
            save_path=str(RESULTS_DIR / "chronos_zero_shot_random_week.png"),
            n_points=len(random_week_plot_df),
            metrics=random_week_ghi_m,
            ghi_threshold=args.ghi_filter_wm2,
        )
    except Exception as exc:
        print(f"  Warning: Could not generate random-week plot: {exc}")

    try:
        plot_4panel_evaluation(
            validation_df["GHI_true"].to_numpy(),
            validation_df["GHI_pred"].to_numpy(),
            hour_array=validation_df["hour"].to_numpy(),
            title=f"Chronos Zero-Shot — Cluster Evaluation ({cluster_name})",
            save_path=str(RESULTS_DIR / "chronos_zero_shot_4panel.png"),
            n_points=len(validation_df),
        )
    except Exception as exc:
        print(f"  Warning: Could not generate 4-panel plot: {exc}")

    for sid in station_metrics:
        try:
            sdf = validation_df[validation_df["station_id"] == sid]
            if len(sdf) < 10:
                continue
            plot_4panel_evaluation(
                sdf["GHI_true"].to_numpy(),
                sdf["GHI_pred"].to_numpy(),
                hour_array=sdf["hour"].to_numpy(),
                title=f"Chronos Zero-Shot — {sid}",
                save_path=str(RESULTS_DIR / f"chronos_zero_shot_4panel_{sid}.png"),
                n_points=len(sdf),
            )
        except Exception as exc:
            print(f"  Warning: Could not generate 4-panel for {sid}: {exc}")

    try:
        build_finetuned_pdf(
            str(RESULTS_DIR),
            ghi_filter_wm2=args.ghi_filter_wm2,
            csv_filename="validation_report_data.csv",
            out_pdf_filename="chronos_zero_shot_validation_report.pdf",
            metrics_filename="evaluation_metrics.json",
        )
    except Exception as exc:
        print(f"  Warning: Could not generate PDF report: {exc}")

    print(
        f"\nChronos zero-shot validation complete — RMSE: {rmse:.2f} W/m²  "
        f"MAE: {mae:.2f} W/m²  nRMSE: {nrmse:.2f}%"
    )
    print(f"Predictions: {predictions_path}")
    print(f"Validation CSV: {prefixed_csv}")
    print(f"Metrics: {prefixed_metrics}")
    print(f"Report: {prefixed_report}")


if __name__ == "__main__":
    main()
