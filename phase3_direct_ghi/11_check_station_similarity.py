"""Quick similarity audit for fetched station GHI series.

Usage:
  python 11_check_station_similarity.py --year 2019

This helps decide whether a chosen cluster is too spatially redundant before
running feature building and model training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import MULTI_STATION_DOWNLOADS_DIR, STATIONS


def load_year_matrix(year: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for station in STATIONS:
        path = MULTI_STATION_DOWNLOADS_DIR / station["id"] / f"ghi_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path, usecols=["datetime", "w_ghr"])
        frames.append(df.rename(columns={"w_ghr": station["id"]}))

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="datetime", how="inner")
    return merged


def summarize_similarity(df: pd.DataFrame, daylight_threshold: float) -> None:
    values = df.drop(columns=["datetime"])
    daylight_mask = values.max(axis=1) > daylight_threshold
    day_values = values.loc[daylight_mask]
    night_values = values.loc[~daylight_mask]

    print(f"rows_total={len(values)} rows_day={len(day_values)} rows_night={len(night_values)}")
    print(f"fraction_all_exact_total={(values.nunique(axis=1) == 1).mean():.4f}")
    print(f"fraction_all_exact_day={(day_values.nunique(axis=1) == 1).mean():.4f}")
    print(f"fraction_all_exact_night={(night_values.nunique(axis=1) == 1).mean():.4f}")

    rows = []
    cols = list(values.columns)
    for idx, left in enumerate(cols):
        for right in cols[idx + 1:]:
            lhs = day_values[left]
            rhs = day_values[right]
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "day_exact_fraction": float((lhs == rhs).mean()),
                    "day_mae": float((lhs - rhs).abs().mean()),
                    "day_corr": float(lhs.corr(rhs)),
                }
            )

    pair_df = pd.DataFrame(rows).sort_values(["day_mae", "day_exact_fraction"], ascending=[True, False])
    print("\nClosest daytime pairs:")
    print(pair_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    spread = pd.DataFrame(
        {
            "range": day_values.max(axis=1) - day_values.min(axis=1),
            "std": day_values.std(axis=1),
        }
    )
    print("\nDaytime spread summary:")
    print(f"mean_range={spread['range'].mean():.2f}")
    print(f"median_range={spread['range'].median():.2f}")
    print(f"p90_range={spread['range'].quantile(0.9):.2f}")
    print(f"p99_range={spread['range'].quantile(0.99):.2f}")
    print(f"mean_std={spread['std'].mean():.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--daylight-threshold", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merged = load_year_matrix(args.year)
    summarize_similarity(merged, daylight_threshold=args.daylight_threshold)
