#!/usr/bin/env python3
"""
Moirai-2 daytime-only site evaluation using the features available in:
option 5_ GHI_20 - Sheet1.csv

This script uses:
1. feat_dynamic_real (9 features): known for both context and forecast horizon
   - sin(day_of_year), cos(day_of_year)
   - sin(hour), cos(hour)
   - solar_declination, eccentricity_factor, hour_angle, zenith_angle, clear_sky_ghi
2. past_feat_dynamic_real (2 features): only known over the context window
   - pressure, cloud_cover

Daytime only: 8 AM - 4 PM (hours 8-16, 9 points/day)

Important: because only daytime rows are evaluated, the target passed to Moirai uses a
synthetic regular hourly index rather than the real wall-clock timestamps. The original
timestamps are still used to build the temporal covariates.
"""

from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_ID = "Salesforce/moirai-2.0-R-small"
SITES = [
    "option 5_ GHI_20 - Sheet1.csv"
]
CONTEXT_DAYS = [1, 2, 3, 4]
DAYTIME_START_HOUR = 8
DAYTIME_END_HOUR = 17
DAYTIME_POINTS_PER_DAY = DAYTIME_END_HOUR - DAYTIME_START_HOUR

KNOWN_COLS = ["solar_declination", "eccentricity_factor", "hour_angle", "zenith_angle", "clear_sky_ghi"]
PAST_COLS = ["pressure", "cloud_cover"]

N_FEAT = 4 + len(KNOWN_COLS)  # 9 dynamic features (past and future)
N_PAST_FEAT = len(PAST_COLS)  # 2 past-only features


def load_site(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    
    # Convert all needed columns
    needed_cols = ["w_ghr", "day_of_year"] + KNOWN_COLS + PAST_COLS
    if "GHI_ICON_Calibrated" in df.columns:
        needed_cols.append("GHI_ICON_Calibrated")
        
    for col in needed_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df = df.dropna(subset=["datetime", "w_ghr"])
    df = df[(df["datetime"].dt.hour >= DAYTIME_START_HOUR) &
            (df["datetime"].dt.hour < DAYTIME_END_HOUR)]
    df = df.sort_values("datetime").set_index("datetime")
    
    # Resample to ensure hourly grid, but this file is already hourly
    hourly = df[needed_cols].resample("h").mean()
    
    # Interpolate lightly for any missing covariates
    for col in KNOWN_COLS + PAST_COLS:
        if col in hourly.columns:
            hourly[col] = hourly[col].interpolate(limit_direction='both')
    hourly = hourly.dropna(subset=["w_ghr"])
    
    return hourly


def get_day(df, date):
    s, e = date.normalize(), date.normalize() + pd.Timedelta(days=1)
    return df[(df.index >= s) & (df.index < e)]


def get_ctx(df, date, n_days):
    e = date.normalize()
    s = e - pd.Timedelta(days=n_days)
    return df[(df.index >= s) & (df.index < e)]


def compute_feature_ranges(df):
    ranges = {}
    for col in KNOWN_COLS + PAST_COLS:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            ranges[col] = (0.0, 0.0)
            continue
        ranges[col] = (float(np.min(finite)), float(np.max(finite)))
    return ranges


def normalize_with_ranges(values_df, col, feature_ranges):
    fmin, fmax = feature_ranges[col]
    vals = values_df[col].values.astype(np.float32)
    if fmax > fmin:
        vals = (vals - fmin) / (fmax - fmin)
        return np.clip(vals, 0.0, 1.0)
    return np.zeros_like(vals, dtype=np.float32)


def build_features(ctx_df, tgt_df, feature_ranges):
    combined = pd.concat([ctx_df, tgt_df])
    h = combined.index.hour.values.astype(np.float32)
    doy = combined["day_of_year"].values.astype(np.float32)
    doy = np.nan_to_num(doy, nan=1.0)
    
    feats = [
        np.sin(2 * np.pi * doy / 365.0).astype(np.float32),
        np.cos(2 * np.pi * doy / 365.0).astype(np.float32),
        np.sin(2 * np.pi * h / 24.0).astype(np.float32),
        np.cos(2 * np.pi * h / 24.0).astype(np.float32),
    ]
    
    for col in KNOWN_COLS:
        v = normalize_with_ranges(combined, col, feature_ranges)
        feats.append(np.nan_to_num(v, nan=0.0).astype(np.float32))
        
    past_feats = []
    for col in PAST_COLS:
        v = normalize_with_ranges(ctx_df, col, feature_ranges)
        past_feats.append(np.nan_to_num(v, nan=0.0).astype(np.float32))
        
    # Shape: (9, context + horizon), (2, context)
    return np.array(feats, dtype=np.float32), np.array(past_feats, dtype=np.float32)


_CACHE = {}
def _get_pred(pl, cl):
    k = (MODEL_ID, pl, cl)
    if k not in _CACHE:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        print(f"  [load model: pl={pl}, cl={cl}, feat={N_FEAT}, past_feat={N_PAST_FEAT}]")
        m = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(MODEL_ID),
            prediction_length=pl, context_length=cl,
            target_dim=1, 
            feat_dynamic_real_dim=N_FEAT,
            past_feat_dynamic_real_dim=N_PAST_FEAT,
        )
        _CACHE[k] = m.create_predictor(batch_size=1)
    return _CACHE[k]


def predict(ctx_vals, feats, past_feats, horizon):
    from gluonts.dataset.common import ListDataset
    pred = _get_pred(horizon, len(ctx_vals))

    # Daytime-only evaluation removes overnight timestamps, so build a synthetic
    # regular index for the target sequence while keeping real-time covariates.
    item = {
        "start": pd.Period("2000-01-01 00:00:00", freq="h"),
        "target": ctx_vals.astype(np.float32),
        "feat_dynamic_real": feats,
        "past_feat_dynamic_real": past_feats,
    }
    
    ds = ListDataset([item], freq="h")
    fc = next(iter(pred.predict(ds)))
    
    if hasattr(fc, "quantile"):
        p = np.asarray(fc.quantile(0.5), dtype=np.float32)
    elif hasattr(fc, "mean"):
        p = np.asarray(fc.mean, dtype=np.float32)
    else:
        raise RuntimeError("No forecast output")
    return np.clip(p[:horizon], 0, None)


def metrics(yt, yp):
    if yt.size == 0:
        return {"MSE": float("nan"), "RMSE": float("nan"), "nRMSE": float("nan"),
                "MAE": float("nan"), "MAPE": float("nan")}
    e = yp - yt
    mse = float(np.mean(e**2))
    rmse = float(np.sqrt(mse))
    mean_yt = float(np.mean(yt))
    nrmse = (rmse / mean_yt * 100) if mean_yt != 0 else 0.0
    return {"MSE": mse, "RMSE": rmse, "nRMSE": nrmse,
            "MAE": float(np.mean(np.abs(e))),
            "MAPE": float(np.mean(np.abs(e)/np.maximum(np.abs(yt),1.0))*100)}


def summarize_grouped_records(records, group_key):
    grouped = {}
    for record in records:
        key = record[group_key]
        bucket = grouped.setdefault(key, {"t": [], "p": []})
        bucket["t"].append(record["true"])
        bucket["p"].append(record["pred"])

    summary = {}
    for key in sorted(grouped):
        yt = np.concatenate(grouped[key]["t"])
        yp = np.concatenate(grouped[key]["p"])
        summary[key] = {
            "Moirai": metrics(yt, yp),
            "n_days": len(grouped[key]["t"]),
            "n_hours": int(yt.size),
        }
    return summary


def build_hourly_summary(records):
    exploded = []
    for record in records:
        for ts, yt, yp in zip(record["timestamps"], record["true"], record["pred"]):
            exploded.append(
                {
                    "hour": int(pd.Timestamp(ts).hour),
                    "true": np.asarray([yt], dtype=np.float32),
                    "pred": np.asarray([yp], dtype=np.float32),
                }
            )

    grouped = summarize_grouped_records(exploded, "hour")
    hourly_rows = []
    for hour, hour_data in grouped.items():
        hour_metrics = hour_data["Moirai"]
        hourly_rows.append(
            {
                "hour": int(hour),
                "n_points": int(hour_data["n_hours"]),
                "mse": float(hour_metrics["MSE"]),
                "rmse": float(hour_metrics["RMSE"]),
                "nrmse": float(hour_metrics["nRMSE"]),
                "mae": float(hour_metrics["MAE"]),
                "mape": float(hour_metrics["MAPE"]),
            }
        )
    return grouped, pd.DataFrame(hourly_rows)


def build_prediction_rows(records, context_days):
    rows = []
    for record in records:
        for ts, yt, yp in zip(record["timestamps"], record["true"], record["pred"]):
            stamp = pd.Timestamp(ts)
            rows.append(
                {
                    "context_days": int(context_days),
                    "forecast_date": record["date"],
                    "target_timestamp": stamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "month": record["month"],
                    "hour": int(stamp.hour),
                    "target": float(yt),
                    "prediction": float(yp),
                }
            )
    return pd.DataFrame(rows)


def evaluate(csv_path, n_days):
    df = load_site(csv_path)
    feature_ranges = compute_feature_ranges(df)
    dates = sorted(df.index.normalize().unique())
    at, ap, dr, skip = [], [], [], 0
    t0 = time.time()
    for i, d in enumerate(dates):
        if i % 50 == 0 and i > 0:
            el = time.time()-t0; r = i/el; eta = (len(dates)-i)/r if r>0 else 0
            print(f"    {i}/{len(dates)} ({el:.0f}s, ~{eta:.0f}s left)")
        
        gt = get_day(df, d)
        if gt.empty or len(gt) != DAYTIME_POINTS_PER_DAY:
            skip += 1
            continue
            
        ctx = get_ctx(df, d, n_days)
        if ctx.empty or len(ctx) != n_days * DAYTIME_POINTS_PER_DAY:
            skip += 1
            continue
            
        feats, past_feats = build_features(ctx, gt, feature_ranges)
        p = predict(ctx["w_ghr"].values, feats, past_feats, len(gt))
        t = gt["w_ghr"].values.astype(np.float32)

        at.append(t); ap.append(p)

        dr.append(
            {
                "date": str(d.date()),
                "month": str(d.date())[:7],
                "nctx": len(ctx),
                "nhrs": len(gt),
                "timestamps": gt.index.to_numpy(),
                "true": t,
                "pred": p,
            }
        )
        
    el = time.time()-t0
    print(f"    Done: {len(dr)} days, {skip} skipped, {el:.1f}s")
    ov = metrics(np.concatenate(at), np.concatenate(ap)) if at else metrics(np.array([]),np.array([]))
    monthly = summarize_grouped_records(dr, "month")
    hourly, hourly_df = build_hourly_summary(dr)
    predictions_df = build_prediction_rows(dr, n_days)

    return {
        "overall": ov,
        "monthly": monthly,
        "hourly": hourly,
        "hourly_df": hourly_df,
        "predictions_df": predictions_df,
        "n_total_days": len(dr),
        "n_total_hours": int(sum(len(t) for t in at)),
        "n_skipped": skip,
    }


def main():
    print("="*80)
    print("  Moirai-2 Daytime Site Eval")
    print(f"  Model: {MODEL_ID}")
    print("  Source: option 5_ GHI_20 - Sheet1.csv")
    print("  Daytime: 8AM-4PM")
    print(f"  feat_dynamic_real (9): {', '.join(['sin_doy', 'cos_doy', 'sin_hr', 'cos_hr'] + KNOWN_COLS)}")
    print(f"  past_feat_dynamic_real (2): {', '.join(PAST_COLS)}")
    print("="*80)
    R = {}
    for s in SITES:
        sn = Path(s).stem.replace("_model_inputs","")
        print(f"\n{'━'*70}\n  📍  {sn}\n{'━'*70}")
        R[sn] = {}
        for nd in CONTEXT_DAYS:
            print(f"\n  ▸ {nd} day(s) context ...")
            r = evaluate(s, nd)
            R[sn][nd] = r
            print(f"\n    {'Month':<10} {'Model':<10} {'Days':>5} {'Hrs':>5}  {'MSE':>10}  {'RMSE':>7}  {'nRMSE':>7}  {'MAE':>7}  {'MAPE':>7}")
            print(f"    {'─'*85}")
            for mo, mm_data in r["monthly"].items():
                m_moirai = mm_data["Moirai"]
                print(f"    {mo:<10} {'Moirai':<10} {mm_data['n_days']:>5} {mm_data['n_hours']:>5}  "
                      f"{m_moirai['MSE']:>10.1f}  {m_moirai['RMSE']:>7.1f}  {m_moirai['nRMSE']:>6.1f}%  {m_moirai['MAE']:>7.1f}  {m_moirai['MAPE']:>6.1f}%")
            
            m = r["overall"]
            print(f"    {'─'*85}")
            print(f"    {'OVERALL':<10} {'Moirai':<10} {r['n_total_days']:>5} {r['n_total_hours']:>5}  "
                  f"{m['MSE']:>10.1f}  {m['RMSE']:>7.1f}  {m['nRMSE']:>6.1f}%  {m['MAE']:>7.1f}  {m['MAPE']:>6.1f}%")

            print(f"\n    {'Hour':<10} {'Model':<10} {'Pts':>5}  {'MSE':>10}  {'RMSE':>7}  {'nRMSE':>7}  {'MAE':>7}  {'MAPE':>7}")
            print(f"    {'─'*79}")
            for hour, hh_data in r["hourly"].items():
                h_moirai = hh_data["Moirai"]
                print(f"    {hour:02d}:00{'':<5} {'Moirai':<10} {hh_data['n_hours']:>5}  "
                      f"{h_moirai['MSE']:>10.1f}  {h_moirai['RMSE']:>7.1f}  {h_moirai['nRMSE']:>6.1f}%  {h_moirai['MAE']:>7.1f}  {h_moirai['MAPE']:>6.1f}%")

            hourly_csv = Path(f"time_series_models/{sn}_ctx{nd}_hourly_rmse.csv")
            r["hourly_df"].to_csv(hourly_csv, index=False)
            print(f"\n    Saved hourly RMSE CSV to: {hourly_csv}")
            predictions_csv = Path(f"time_series_models/{sn}_ctx{nd}_per_timestamp_predictions.csv")
            r["predictions_df"].to_csv(predictions_csv, index=False)
            print(f"    Saved per-timestamp predictions CSV to: {predictions_csv}")
            del r["hourly_df"]
            del r["predictions_df"]
    
    print("\n\n"+"="*80)
    print("  SUMMARY TABLE")
    print("="*80)
    print(f"{'Site':<40} {'Ctx':<6} {'Days':>5} {'Hrs':>5}  {'MSE':>10}  {'RMSE':>7}  {'nRMSE':>7}  {'MAE':>7}  {'MAPE':>7}")
    print("─"*105)
    for sn in R:
        for nd in R[sn]:
            m = R[sn][nd]["overall"]
            print(f"{sn[:40]:<40} {nd} day  {R[sn][nd]['n_total_days']:>5} {R[sn][nd]['n_total_hours']:>5}  "
                  f"{m['MSE']:>10.2f}  {m['RMSE']:>7.2f}  {m['nRMSE']:>6.1f}%  {m['MAE']:>7.2f}  {m['MAPE']:>6.1f}%")
        print()
        
    op = Path("time_series_models/moirai_eval_all_features.json")
    def cv(o):
        if isinstance(o,(np.integer,)): return int(o)
        if isinstance(o,(np.floating,)): return float(o)
        if isinstance(o,np.ndarray): return o.tolist()
        return str(o)
    op.write_text(json.dumps(R, indent=2, default=cv), encoding="utf-8")
    print(f"\n✅ Saved to: {op}")


if __name__ == "__main__":
    main()
