"""Build a Moirai-style processed test CSV from persistence predictions.

This adapts `persistence_ghi_predictions.csv` into the same column layout used
by `dataset/processed_data_2017_2019.csv`, so it can be used as model input for
test-time window generation.

Assumptions:
- `CAF` is taken from `phase1_cloud_mapper/downloads/merged_data.csv`.
- The downstream Phase 2 dataset builder uses a single synthetic station id, `site_1`.
- `GHI` is taken from `w_ghr` in `phase1_cloud_mapper/downloads/merged_data.csv`.
- `zenith_angle`, `hour_sin`, `hour_cos`, `doy_sin`, and `doy_cos` are reused
  from `phase1_cloud_mapper/downloads/merged_data.csv` for matching timestamps.
- `cloud_cover` is derived from Chennai rows in
  `phase1_cloud_mapper/downloads/icon_2024.csv` by time interpolation onto the
  persistence timeline. For 2025 timestamps, the script falls back to the
  corresponding 2024 month-day-hour value when available.
"""

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

PERSISTENCE_CSV = BASE_DIR / "persistence_ghi_predictions.csv"
MERGED_DATA_CSV = PROJECT_ROOT / "phase1_cloud_mapper" / "downloads" / "merged_data.csv"
ICON_2024_CSV = PROJECT_ROOT / "phase1_cloud_mapper" / "downloads" / "icon_2024.csv"
OUTPUT_CSV = BASE_DIR / "dataset" / "processed_test_from_persistence.csv"

STATION_ID = "chennai"
OUTPUT_STATION_ID = "site_1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloud-cover-csv",
        type=Path,
        default=None,
        help="Optional CSV with at least datetime and cloud_cover columns.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="Where to write the processed test CSV.",
    )
    return parser.parse_args()


def load_time_features() -> pd.DataFrame:
    df = pd.read_csv(MERGED_DATA_CSV, parse_dates=["datetime"])
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    cols = ["datetime", "CAF", "w_ghr", "zenith_angle", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    return df[cols].drop_duplicates(subset=["datetime"])


def build_cloud_cover(target_times: pd.Series) -> pd.DataFrame:
    icon = pd.read_csv(ICON_2024_CSV, parse_dates=["datetime"])
    icon = icon[icon["station_id"] == STATION_ID].copy()
    if getattr(icon["datetime"].dt, "tz", None) is not None:
        icon["datetime"] = icon["datetime"].dt.tz_localize(None)

    # Interpolate the Chennai 2024 archive onto an hourly grid.
    base_2024 = (
        icon[["datetime", "cloud_cover"]]
        .dropna(subset=["datetime"])
        .drop_duplicates(subset=["datetime"])
        .set_index("datetime")
        .sort_index()
    )
    hourly_2024_index = pd.date_range(
        base_2024.index.min().floor("h"),
        base_2024.index.max().ceil("h"),
        freq="h",
    )
    hourly_2024 = (
        base_2024.reindex(base_2024.index.union(hourly_2024_index))
        .sort_index()
        .interpolate(method="time", limit_direction="both")
        .reindex(hourly_2024_index)
        .rename_axis("datetime")
        .reset_index()
    )

    target = pd.DataFrame({"datetime": pd.to_datetime(target_times).sort_values().unique()})
    cloud = target.merge(hourly_2024, on="datetime", how="left")

    # 2025 fallback: reuse the corresponding 2024 month-day-hour value.
    hourly_2024["month"] = hourly_2024["datetime"].dt.month
    hourly_2024["day"] = hourly_2024["datetime"].dt.day
    hourly_2024["hour"] = hourly_2024["datetime"].dt.hour
    climatology = hourly_2024.drop_duplicates(subset=["month", "day", "hour"])[
        ["month", "day", "hour", "cloud_cover"]
    ].rename(columns={"cloud_cover": "cloud_cover_fallback"})

    cloud["month"] = cloud["datetime"].dt.month
    cloud["day"] = cloud["datetime"].dt.day
    cloud["hour"] = cloud["datetime"].dt.hour
    cloud = cloud.merge(climatology, on=["month", "day", "hour"], how="left")
    cloud["cloud_cover"] = cloud["cloud_cover"].fillna(cloud["cloud_cover_fallback"])
    cloud["cloud_cover"] = (
        cloud.sort_values("datetime")
        .set_index("datetime")["cloud_cover"]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
        .reset_index(drop=True)
    )
    cloud["cloud_cover"] = cloud["cloud_cover"].clip(lower=0.0, upper=1.0)

    return cloud[["datetime", "cloud_cover"]]


def load_cloud_cover_from_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    if "cloud_cover" not in df.columns:
        raise KeyError(f"{path} must contain a cloud_cover column.")
    return df[["datetime", "cloud_cover"]].drop_duplicates(subset=["datetime"])


def validate_exact_cloud_cover_timestamps(target_times: pd.Series, cloud_cover_df: pd.DataFrame, path: Path) -> None:
    target_index = pd.Index(pd.to_datetime(target_times).sort_values().unique(), name="datetime")
    cloud_index = pd.Index(pd.to_datetime(cloud_cover_df["datetime"]).sort_values().unique(), name="datetime")
    missing = target_index.difference(cloud_index)
    if len(missing) > 0:
        preview = ", ".join(str(ts) for ts in missing[:10])
        raise ValueError(
            f"{path} is missing {len(missing)} required datetime rows. "
            f"First missing timestamps: {preview}"
        )


def main() -> None:
    args = parse_args()
    persistence = pd.read_csv(PERSISTENCE_CSV, parse_dates=["datetime"])
    persistence["datetime"] = pd.to_datetime(persistence["datetime"])

    out = pd.DataFrame({
        "station_id": OUTPUT_STATION_ID,
        "datetime": persistence["datetime"],
        "clear_sky_ghi": persistence["clear_sky_ghi"],
    })

    out = out.merge(load_time_features(), on="datetime", how="left")
    if args.cloud_cover_csv is not None:
        cloud_cover = load_cloud_cover_from_file(args.cloud_cover_csv)
        validate_exact_cloud_cover_timestamps(out["datetime"], cloud_cover, args.cloud_cover_csv)
    else:
        cloud_cover = build_cloud_cover(out["datetime"])
    out = out.merge(cloud_cover, on="datetime", how="left")

    cols = [
        "station_id",
        "datetime",
        "CAF",
        "w_ghr",
        "clear_sky_ghi",
        "cloud_cover",
        "zenith_angle",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    out = out[cols].sort_values(["station_id", "datetime"]).reset_index(drop=True)
    out = out.rename(columns={"w_ghr": "GHI"})

    missing = out.isna().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        raise ValueError(f"Output still contains missing values: {missing_cols.to_dict()}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv}")
    print(f"Shape: {out.shape}")
    print(out.head(5).to_string())


if __name__ == "__main__":
    main()
