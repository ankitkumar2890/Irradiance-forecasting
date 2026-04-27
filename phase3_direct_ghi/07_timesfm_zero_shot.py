"""Zero-shot Google TimesFM baseline on the Phase 3 Direct GHI validation set.

This script uses the same validation windows as the Moirai fine-tuned pipeline
and writes predictions/metrics in the same overall structure under results/.

Expected outputs:
  - results/timesfm_zero_shot_predictions.csv
  - results/timesfm_zero_shot_metrics.json
  - results/timesfm_zero_shot_validation_report_data.csv
  - results/timesfm_zero_shot_evaluation_metrics.json
  - results/timesfm_zero_shot_evaluation_report.txt

Notes:
  - The official TimesFM package currently targets Python >=3.10,<3.12.
  - Run this script with python3.11 after installing timesfm in that env.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (  # noqa: E402
    DATASET_DIR,
    MULTI_STATION_DOWNLOADS_DIR,
    RESULTS_DIR,
    CLUSTER_NAME,
    PREDICTION_LENGTH,
    GHI_SCALE_FACTOR,
    STATIONS,
    YEARS,
)
from master_metrics import calculate_metrics  # noqa: E402

DEFAULT_GHI_FILTER_WM2 = 20.0
ONE_WEEK_DAYS = 7
RANDOM_WEEK_SEED = 42
TIMESFM_MODEL_NAME = "google/timesfm-2.5-200m-pytorch"
TIMESFM_MAX_CONTEXT = 1024
TIMESFM_MAX_HORIZON = 256
TIMESFM_REPO_URL = "https://github.com/google-research/timesfm"
LEGACY_TIMESFM_MODEL_NAME = "google/timesfm-1.0-200m-pytorch"
LEGACY_TIMESFM_CONTEXT = 512
LEGACY_TIMESFM_HORIZON = 128


def normalize_ghi_to_hour_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Average :30 timestamps to the hour grid if needed."""
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


def load_timesfm(max_context: int | None = None, max_horizon: int | None = None):
    try:
        import timesfm  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "timesfm is not installed in this environment. "
            "Use a Python 3.11 environment and install the official package/repo first."
        ) from exc

    model_cls = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
    forecast_config_cls = getattr(timesfm, "ForecastConfig", None)
    if model_cls is not None and forecast_config_cls is not None:
        compile_context = int(max_context or TIMESFM_MAX_CONTEXT)
        compile_horizon = int(max_horizon or TIMESFM_MAX_HORIZON)
        model = model_cls.from_pretrained(TIMESFM_MODEL_NAME)
        model.compile(
            forecast_config_cls(
                max_context=compile_context,
                max_horizon=compile_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        model._phase3_timesfm_api = "v2p5"
        return model

    legacy_api = all(
        hasattr(timesfm, attr)
        for attr in ("TimesFm", "TimesFmHparams", "TimesFmCheckpoint")
    )
    if legacy_api:
        print(
            "Detected legacy TimesFM package API; attempting compatibility load "
            f"with legacy checkpoint {LEGACY_TIMESFM_MODEL_NAME}."
        )
        try:
            model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    context_len=LEGACY_TIMESFM_CONTEXT,
                    horizon_len=max(PREDICTION_LENGTH, LEGACY_TIMESFM_HORIZON),
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=20,
                    num_heads=16,
                    model_dims=1280,
                    per_core_batch_size=1,
                    backend="cpu",
                    point_forecast_mode="mean",
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    version="torch",
                    huggingface_repo_id=LEGACY_TIMESFM_MODEL_NAME,
                ),
            )
            model._phase3_timesfm_api = "legacy"
            model._phase3_timesfm_model_name = LEGACY_TIMESFM_MODEL_NAME
            return model
        except Exception as exc:
            exc_name = type(exc).__name__
            exc_msg = str(exc)
            if "torch_model.ckpt" in exc_msg and "model.safetensors" not in exc_msg:
                raise RuntimeError(
                    "Installed timesfm package exposes only the legacy API, but "
                    f"{TIMESFM_MODEL_NAME} is a newer Hugging Face checkpoint layout "
                    "that ships `model.safetensors` instead of the legacy "
                    "`torch_model.ckpt`. This means your local timesfm package is too "
                    "old for the 2.5 checkpoint format. Upgrade to the current "
                    f"official TimesFM 2.5 code from {TIMESFM_REPO_URL} in this "
                    "Python 3.11 environment, then rerun the script."
                ) from exc
            if (
                "ConnectError" in exc_msg
                or "LocalEntryNotFoundError" in exc_name
                or "internet connection" in exc_msg.lower()
            ):
                raise RuntimeError(
                    "Installed timesfm package exposes only the legacy API, and "
                    f"the compatibility loader could not download {TIMESFM_MODEL_NAME} "
                    "from Hugging Face in this environment. If you are offline, retry "
                    "with internet access or pre-download the checkpoint. If you are "
                    "online, also upgrade to the current official TimesFM 2.5 code "
                    f"from {TIMESFM_REPO_URL} because the package in this environment "
                    "still uses the legacy interface."
                ) from exc
            raise RuntimeError(
                "Installed timesfm package exposes only the legacy API and could "
                f"not load {TIMESFM_MODEL_NAME}. This usually means the environment "
                "has an older pre-2.5 package build. Install the current official "
                f"TimesFM 2.5 code from {TIMESFM_REPO_URL} in your Python 3.11 "
                "environment, then rerun this script."
            ) from exc

    raise RuntimeError(
        "Installed timesfm package does not expose a supported inference API for "
        f"{TIMESFM_MODEL_NAME}. Install the current official TimesFM 2.5 code from "
        f"{TIMESFM_REPO_URL} in your Python 3.11 environment."
    )


def run_timesfm_forecast(model, context):
    context_arr = context.astype(np.float32)
    if getattr(model, "_phase3_timesfm_api", "v2p5") == "legacy":
        context_len = getattr(model, "context_len", LEGACY_TIMESFM_CONTEXT)
        context_arr = context_arr[-context_len:]

    forecast_kwargs = {"inputs": [context_arr]}
    if getattr(model, "_phase3_timesfm_api", "v2p5") == "v2p5":
        forecast_kwargs["horizon"] = PREDICTION_LENGTH
    else:
        forecast_kwargs["normalize"] = True
    point_forecast, quantile_forecast = model.forecast(**forecast_kwargs)

    point = np.asarray(point_forecast, dtype=np.float32)[0, :PREDICTION_LENGTH]
    p10 = point.copy()
    p90 = point.copy()

    if quantile_forecast is not None:
        q = np.asarray(quantile_forecast, dtype=np.float32)
        if q.ndim == 3 and q.shape[0] >= 1 and q.shape[1] >= PREDICTION_LENGTH:
            if q.shape[2] >= 10:
                p10 = q[0, :PREDICTION_LENGTH, 1]
                p90 = q[0, :PREDICTION_LENGTH, 9]
            elif q.shape[2] >= 3:
                p10 = q[0, :PREDICTION_LENGTH, 1]
                p90 = q[0, :PREDICTION_LENGTH, -1]

    point = np.clip(point, 0.0, None)
    p10 = np.clip(p10, 0.0, None)
    p90 = np.maximum(np.clip(p90, 0.0, None), p10)
    if not np.isfinite(point).any():
        model_name = getattr(model, "_phase3_timesfm_model_name", TIMESFM_MODEL_NAME)
        raise RuntimeError(
            f"TimesFM produced only NaN forecasts for checkpoint {model_name}. "
            "This usually indicates an incompatible package/checkpoint combination."
        )
    return point, p10, p90


def evaluate_prediction_frame(df_pred: pd.DataFrame, ghi_filter_wm2: float):
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
    solar_day_df = df[df["zenith_angle"] < 90].copy() if "zenith_angle" in df.columns else pd.DataFrame()
    validation_df = df[df["GHI_true"] > ghi_filter_wm2].copy()
    if validation_df.empty:
        raise RuntimeError("No validation rows passed the GHI filter.")

    persist_eval = ghi.copy()
    persist_eval["GHI_persist_24h"] = persist_eval.groupby("station_id")["w_ghr"].shift(PREDICTION_LENGTH)
    persist_eval = persist_eval[
        (persist_eval["w_ghr"] > ghi_filter_wm2) &
        persist_eval["GHI_persist_24h"].notna()
    ].copy()
    persist_rmse = float("nan")
    if not persist_eval.empty:
        persist_rmse = np.sqrt(np.mean((persist_eval["GHI_persist_24h"] - persist_eval["w_ghr"]) ** 2))

    all_hours_m, all_hours_master = summarize_subset(all_hours_df, "GHI_all_hours", ghi_filter_wm2)
    solar_day_m, solar_day_master = summarize_subset(solar_day_df, "GHI_solar_day", ghi_filter_wm2)
    ghi_m = compute_metrics(validation_df["GHI_true"].values, validation_df["GHI_pred"].values, "GHI_direct")
    ghi_m["Skill_vs_persist"] = 1.0 - ghi_m["RMSE"] / persist_rmse if persist_rmse > 0 else float("nan")

    plot_station_id = validation_df["station_id"].value_counts().sort_values(ascending=False).index[0]
    plot_df = df[df["station_id"] == plot_station_id].copy()
    plot_validation_df = validation_df[validation_df["station_id"] == plot_station_id].copy()
    first_week_start = plot_df["datetime"].min()
    first_week_metrics_df = slice_window(plot_validation_df, first_week_start)
    random_week_start = choose_random_later_week_start(plot_validation_df, first_week_start)
    random_week_metrics_df = slice_window(plot_validation_df, random_week_start)
    first_week_ghi_m = compute_metrics(first_week_metrics_df["GHI_true"].values, first_week_metrics_df["GHI_pred"].values, "GHI_week_1")
    random_week_ghi_m = compute_metrics(random_week_metrics_df["GHI_true"].values, random_week_metrics_df["GHI_pred"].values, "GHI_random_week")
    master_ghi_m = calculate_metrics(
        validation_df["GHI_true"].to_numpy(),
        validation_df["GHI_pred"].to_numpy(),
        mape_threshold=ghi_filter_wm2,
    )

    station_metrics = {}
    for station_id, group in validation_df.groupby("station_id"):
        station_metrics[station_id] = compute_metrics(group["GHI_true"].values, group["GHI_pred"].values, station_id)

    horizon_rmse = {}
    for lead_time, group in validation_df.groupby("lead_time_h"):
        t = group["GHI_true"].to_numpy()
        p = group["GHI_pred"].to_numpy()
        horizon_rmse[int(lead_time)] = float(np.sqrt(np.mean((p - t) ** 2)))

    all_metrics = {
        "pipeline": "phase3_direct_ghi_timesfm_zero_shot",
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
        "per_station": station_metrics,
        "horizon_rmse": horizon_rmse,
    }

    report_lines = [
        f"DIRECT GHI FORECASTING EVALUATION — Cluster: {CLUSTER_NAME}",
        "Model: Google TimesFM zero-shot",
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
        report_lines.extend([f"  +{lead:02d}h: {rmse:.2f}" for lead, rmse in sorted(horizon_rmse.items())])

    return df, all_metrics, report_lines


def main():
    ghi_filter_wm2 = DEFAULT_GHI_FILTER_WM2
    X_past = np.load(DATASET_DIR / "X_past_val.npy")
    y_future = np.load(DATASET_DIR / "y_future_val.npy")
    times = np.load(DATASET_DIR / "times_val.npy", allow_pickle=True).astype("datetime64[ns]")
    station_ids = np.load(DATASET_DIR / "station_ids_val.npy", allow_pickle=True)
    context_window = int(X_past.shape[1])
    forecast_horizon = int(y_future.shape[1])

    print(f"Loading TimesFM zero-shot baseline: {TIMESFM_MODEL_NAME}")
    print(f"Compiling TimesFM for context={context_window}, horizon={forecast_horizon}")
    model = load_timesfm(max_context=context_window, max_horizon=forecast_horizon)

    csv_rows = []
    all_preds = []
    all_true = []

    print(f"Validation windows: {len(X_past)}")
    for i in range(len(X_past)):
        context = X_past[i, :, 0].astype(np.float32)
        pred, pred_p10, pred_p90 = run_timesfm_forecast(model, context)
        y_true_wm2 = y_future[i].astype(np.float32) * GHI_SCALE_FACTOR
        forecast_start = pd.Timestamp(times[i, 0]) - pd.Timedelta(hours=1)

        all_preds.append(pred)
        all_true.append(y_true_wm2)

        for h in range(PREDICTION_LENGTH):
            ts = pd.Timestamp(times[i, h])
            lead_time_h = int((ts - forecast_start) / pd.Timedelta(hours=1))
            csv_rows.append({
                "station_id": str(station_ids[i]),
                "datetime": ts,
                "hour": ts.hour,
                "lead_time_h": lead_time_h,
                "forecast_start": forecast_start,
                "GHI_true": float(y_true_wm2[h]),
                "GHI_p10": float(pred_p10[h]),
                "GHI_pred": float(pred[h]),
                "GHI_p90": float(pred_p90[h]),
            })

        if i % 20 == 0:
            print(f"  {i}/{len(X_past)}")

    pred_df = pd.DataFrame(csv_rows).sort_values(["station_id", "datetime", "lead_time_h"]).reset_index(drop=True)
    pred_path = RESULTS_DIR / "timesfm_zero_shot_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    rmse = np.sqrt(np.mean((all_preds - all_true) ** 2))
    mae = np.mean(np.abs(all_preds - all_true))
    mean_true = np.mean(all_true[all_true > 0])
    nrmse = (rmse / mean_true * 100) if mean_true > 0 else float("nan")

    raw_metrics = {
        "GHI_RMSE_wm2": float(rmse),
        "GHI_MAE_wm2": float(mae),
        "GHI_nRMSE_pct": float(nrmse),
        "cluster": CLUSTER_NAME,
        "model": "google_timesfm_zero_shot",
    }
    raw_metrics_path = RESULTS_DIR / "timesfm_zero_shot_metrics.json"
    with open(raw_metrics_path, "w") as f:
        json.dump(raw_metrics, f, indent=2)

    evaluated_df, all_metrics, report_lines = evaluate_prediction_frame(pred_df, ghi_filter_wm2)

    report_csv_path = RESULTS_DIR / "timesfm_zero_shot_validation_report_data.csv"
    evaluated_df.to_csv(report_csv_path, index=False)

    metrics_path = RESULTS_DIR / "timesfm_zero_shot_evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    report_path = RESULTS_DIR / "timesfm_zero_shot_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\n  TimesFM Zero-Shot Validation — RMSE: {rmse:.2f} W/m²  MAE: {mae:.2f} W/m²  nRMSE: {nrmse:.1f}%")
    print(f"  Saved → {pred_path}")
    print(f"  Saved → {raw_metrics_path}")
    print(f"  Saved → {report_csv_path}")
    print(f"  Saved → {report_path}")
    print(f"  Saved → {metrics_path}")


if __name__ == "__main__":
    main()
