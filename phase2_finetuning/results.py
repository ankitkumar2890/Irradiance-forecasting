"""Recover GHI from CAF predictions and evaluate against measured w_ghr."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, DOWNLOADS_DIR, RESULTS_DIR, YEARS
from master_metrics import calculate_metrics, print_metrics


def load_predictions():
    pred_file = RESULTS_DIR / "finetuned_predictions.csv"
    if not pred_file.exists():
        raise FileNotFoundError(
            f"{pred_file} not found. Run 05_finetuned_inference.py first."
        )

    df = pd.read_csv(pred_file)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_processed():
    proc_file = DATASET_DIR / "processed_data_2017_2019.csv"
    if not proc_file.exists():
        raise FileNotFoundError(
            f"{proc_file} not found. Run 02_build_features.py first."
        )

    proc = pd.read_csv(proc_file)
    proc["datetime"] = pd.to_datetime(proc["datetime"])
    return proc[["datetime", "clear_sky_ghi", "zenith_angle"]].drop_duplicates(
        subset=["datetime"]
    )


def load_measured_ghi():
    frames = []
    for year in YEARS:
        ghi_file = DOWNLOADS_DIR / f"ghi_{year}.csv"
        if not ghi_file.exists():
            raise FileNotFoundError(f"Missing measured GHI file: {ghi_file}")

        df = pd.read_csv(ghi_file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        frames.append(df[["datetime", "w_ghr"]])

    ghi = pd.concat(frames, ignore_index=True)
    return ghi.drop_duplicates(subset=["datetime"]).sort_values("datetime")


def main():
    pred = load_predictions()
    proc = load_processed()
    ghi = load_measured_ghi()

    df = (
        pred.merge(proc, on="datetime", how="left")
        .merge(ghi, on="datetime", how="left")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    df = df.dropna(subset=["clear_sky_ghi", "w_ghr"])

    # Physical reconstruction used in the repo: GHI = CAF * clear_sky_ghi
    df["GHI_pred"] = df["CAF_pred"] * df["clear_sky_ghi"]
    df["GHI_true"] = df["w_ghr"]

    daytime = df[df["zenith_angle"] < 85].copy()

    overall_metrics = calculate_metrics(
        df["GHI_true"].to_numpy(),
        df["GHI_pred"].to_numpy(),
    )
    daytime_metrics = calculate_metrics(
        daytime["GHI_true"].to_numpy(),
        daytime["GHI_pred"].to_numpy(),
    )

    out_cols = [
        "datetime",
        "hour",
        "CAF_true",
        "CAF_pred",
        "clear_sky_ghi",
        "zenith_angle",
        "w_ghr",
        "GHI_true",
        "GHI_pred",
    ]
    df[out_cols].to_csv(RESULTS_DIR / "finetuned_ghi_from_caf.csv", index=False)

    metrics_payload = {
        "overall": {k: float(v) for k, v in overall_metrics.items()},
        "daytime_zenith_lt_85": {k: float(v) for k, v in daytime_metrics.items()},
        "rows": int(len(df)),
        "daytime_rows": int(len(daytime)),
    }
    with open(RESULTS_DIR / "finetuned_ghi_metrics_w_ghr.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("Using measured GHI truth from column: w_ghr")
    print(f"Rows matched: {len(df)}")
    print(f"Daytime rows (zenith < 85): {len(daytime)}")

    print_metrics(overall_metrics, title="GHI METRICS vs w_ghr (ALL HOURS)")
    print_metrics(daytime_metrics, title="GHI METRICS vs w_ghr (DAYTIME, zenith < 85)")

    print(f"\nSaved: {RESULTS_DIR / 'finetuned_ghi_from_caf.csv'}")
    print(f"Saved: {RESULTS_DIR / 'finetuned_ghi_metrics_w_ghr.json'}")


if __name__ == "__main__":
    main()
