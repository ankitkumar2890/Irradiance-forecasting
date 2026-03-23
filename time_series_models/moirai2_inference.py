#!/usr/bin/env python3
"""Minimal Moirai2-only inference script."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_eval_freq(freq: str) -> str:
    f = str(freq).strip()
    if f.upper() == "H":
        return "h"
    return f


def parse_sequence(sequence_arg: str) -> np.ndarray:
    parts = [p.strip() for p in sequence_arg.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("Sequence must contain at least 2 numeric values.")
    return np.asarray([float(x) for x in parts], dtype=np.float32)


def load_sequence_from_csv(csv_path: str, value_col: str | None) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if value_col:
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found in {csv_path}.")
        series = df[value_col]
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("CSV has no numeric columns; pass --value-col explicitly.")
        series = df[numeric_cols[0]]

    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float32)
    if len(values) < 2:
        raise ValueError("Need at least 2 numeric values after dropping NaNs.")
    return values


def configure_hf_auth(hf_token: str | None) -> None:
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token


_MODEL_CACHE = {}

def get_moirai_predictor(model_name, prediction_length, context_length):
    key = (model_name, prediction_length, context_length)
    if key not in _MODEL_CACHE:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(model_name),
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        _MODEL_CACHE[key] = model.create_predictor(batch_size=1)
    return _MODEL_CACHE[key]

def run_moirai2(
    sequence: np.ndarray,
    horizon: int,
    context_length: int,
    freq: str,
    start: str,
    model_id: str | None,
) -> np.ndarray:
    from gluonts.dataset.common import ListDataset

    model_name = model_id or "Salesforce/moirai-2.0-R-small"
    predictor = get_moirai_predictor(model_name, horizon, context_length)

    context = sequence[-context_length:].astype(np.float32)
    item = {
        "start": pd.Period(pd.Timestamp(start), freq=freq),
        "target": context,
    }
    ds = ListDataset([item], freq=freq)
    forecast = next(iter(predictor.predict(ds)))

    if hasattr(forecast, "quantile"):
        pred = np.asarray(forecast.quantile(0.5), dtype=np.float32)
    elif hasattr(forecast, "mean"):
        pred = np.asarray(forecast.mean, dtype=np.float32)
    else:
        raise RuntimeError("Could not extract point forecast from Moirai2 output.")
    return pred[:horizon]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mape_eps: float) -> dict:
    if y_true.size == 0:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
    err = y_pred - y_true
    mse = float(np.mean(np.square(err)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(y_true), float(mape_eps))
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}


def load_hourly_series_from_ground_truth_csv(csv_path: str, target_col: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Missing required target column: {target_col}")

    if "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], errors="coerce")
    elif {"day", "hour", "minute"}.issubset(set(df.columns)):
        day = pd.to_datetime(df["day"], errors="coerce")
        hour = pd.to_numeric(df["hour"], errors="coerce")
        minute = pd.to_numeric(df["minute"], errors="coerce")
        ts = day + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    else:
        raise ValueError("Missing timestamp columns. Need either 'datetime' or ('day', 'hour', 'minute')")

    values = pd.to_numeric(df[target_col], errors="coerce")

    data = pd.DataFrame({"timestamp": ts, "value": values}, copy=False).dropna(subset=["timestamp"])
    data = data.sort_values("timestamp")
    series = data.groupby("timestamp", as_index=True)["value"].mean().sort_index()
    series = series.clip(lower=0.0)

    minute_index = pd.date_range(
        series.index.min().floor("min"),
        series.index.max().ceil("min"),
        freq="1min",
    )
    series = series.reindex(minute_index)
    series = series.interpolate(method="time", limit=5, limit_direction="both")
    return series.resample("h").mean().astype(np.float32)


def evaluate_from_ground_truth_csv(args: argparse.Namespace) -> dict:
    series = load_hourly_series_from_ground_truth_csv(args.eval_ground_truth_csv, args.target_col)
    values = series.to_numpy(dtype=np.float32)
    index = series.index

    candidate_pos = np.where((index.hour == args.origin_hour) & (index.minute == 0))[0]
    rows = []

    for pos in candidate_pos:
        if pos - args.context_length < 0 or pos + args.horizon > values.shape[0]:
            continue
        context = values[pos - args.context_length : pos]
        if np.isnan(context).any():
            continue
        pred = run_moirai2(
            sequence=context,
            horizon=args.horizon,
            context_length=args.context_length,
            freq=normalize_eval_freq(args.eval_freq),
            start=str(index[pos - args.context_length]),
            model_id=args.model_id,
        )
        true = values[pos : pos + args.horizon]
        if np.isnan(true).any():
            continue
        for step, (ts, y_true, y_pred) in enumerate(zip(index[pos : pos + args.horizon], true, pred), start=1):
            rows.append(
                {
                    "origin": index[pos],
                    "target_timestamp": ts,
                    "step": int(step),
                    "target": float(y_true),
                    "prediction": float(max(0.0, y_pred)),
                }
            )

    if not rows:
        metrics = compute_metrics(np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), args.mape_eps)
        return {"overall_metrics": metrics, "n_points": 0}

    pred_df = pd.DataFrame(rows).sort_values(["target_timestamp", "origin"]).drop_duplicates(
        subset=["target_timestamp"], keep="last"
    )

    ts = pd.DatetimeIndex(pred_df["target_timestamp"])
    mask = (ts >= pd.Timestamp(args.filter_start_date)) & (
        ts <= pd.Timestamp(args.filter_end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    )
    mask &= (ts.hour >= args.filter_start_hour) & (ts.hour <= args.filter_end_hour)

    y_true = pred_df.loc[mask, "target"].to_numpy(dtype=np.float32)
    y_pred = pred_df.loc[mask, "prediction"].to_numpy(dtype=np.float32)
    metrics = compute_metrics(y_true, y_pred, args.mape_eps)
    return {"overall_metrics": metrics, "n_points": int(y_true.size)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal Moirai2 inference.")
    p.add_argument("--sequence", type=str, help="Comma-separated values.")
    p.add_argument("--csv", type=str, help="CSV path with history values.")
    p.add_argument("--value-col", type=str, default=None, help="Value column in CSV.")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--context-length", type=int, default=168)
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--start", type=str, default="2020-01-01 00:00:00")
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--eval-ground-truth-csv", type=str, default=None)
    p.add_argument("--target-col", type=str, default="w_ghr")
    p.add_argument("--eval-freq", type=str, default="h")
    p.add_argument("--origin-hour", type=int, default=6)
    p.add_argument("--filter-start-date", type=str, default="2024-10-01")
    p.add_argument("--filter-end-date", type=str, default="2024-10-07")
    p.add_argument("--filter-start-hour", type=int, default=6)
    p.add_argument("--filter-end-hour", type=int, default=18)
    p.add_argument("--mape-eps", type=float, default=1.0)
    return p


def get_sequence(args: argparse.Namespace) -> np.ndarray:
    if bool(args.sequence) == bool(args.csv):
        raise ValueError("Provide exactly one of --sequence or --csv.")
    if args.sequence:
        return parse_sequence(args.sequence)
    return load_sequence_from_csv(args.csv, args.value_col)


def main() -> int:
    args = build_parser().parse_args()
    configure_hf_auth(args.hf_token)

    if args.eval_ground_truth_csv:
        result = evaluate_from_ground_truth_csv(args)
        print(json.dumps(result, indent=2, default=str))
        metrics = result["overall_metrics"]
        print(f"MSE  : {metrics['mse']}")
        print(f"RMSE : {metrics['rmse']}")
        print(f"MAE  : {metrics['mae']}")
        print(f"MAPE : {metrics['mape']}")
        if args.output_json:
            out_path = Path(args.output_json)
            out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return 0

    sequence = get_sequence(args)
    context_length = min(args.context_length, len(sequence))
    forecast = run_moirai2(
        sequence=sequence,
        horizon=args.horizon,
        context_length=context_length,
        freq=args.freq,
        start=args.start,
        model_id=args.model_id,
    )

    result = {
        "backend": "moirai2",
        "horizon": int(args.horizon),
        "input_length": int(len(sequence)),
        "forecast": [float(x) for x in forecast.tolist()],
    }
    print(json.dumps(result, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
