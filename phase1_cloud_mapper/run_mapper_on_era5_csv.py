"""Run the frozen CloudMapper on an arbitrary ERA5 CSV."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from config import (
    CHECKPOINT_DIR,
    DROPOUT,
    HIDDEN_DIM,
    ICON_COLS,
    NUM_RES_BLOCKS,
    STATIONS,
    ERA5_FRACTION_COLS,
    VERSION,
)
from features import add_time_features, add_lag_features
from generate_synthetic import load_frozen_mapper
from model_architecture import TAFResNet
from time_utils import to_ist_series


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--era5-csv",
        type=Path,
        required=True,
        help="ERA5 CSV with datetime, station_id, and the expected ERA5 fraction columns.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to write datetime/station_id/cloud_cover predictions.",
    )
    parser.add_argument(
        "--default-station-id",
        type=str,
        default="site_1",
        help="Used only when the ERA5 CSV has no station_id column.",
    )
    parser.add_argument("--lat", type=float, default=None, help="Latitude for custom single-site features.")
    parser.add_argument("--lon", type=float, default=None, help="Longitude for custom single-site features.")
    parser.add_argument(
        "--alt-m",
        type=float,
        default=None,
        help="Altitude in meters for custom single-site features. Defaults to the nearest configured station.",
    )
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_station_id(df: pd.DataFrame, default_station_id: str) -> pd.DataFrame:
    df = df.copy()
    if "station_id" not in df.columns:
        df["station_id"] = default_station_id
    return df


def nearest_station(lat: float, lon: float) -> dict:
    return min(
        STATIONS,
        key=lambda station: (station["lat"] - lat) ** 2 + (station["lon"] - lon) ** 2,
    )


def build_mapper_features(df: pd.DataFrame, lat: float, lon: float, alt_m: float) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    for col in ERA5_FRACTION_COLS:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)
    df = add_time_features(df)
    df = add_lag_features(df)
    df["lat_norm"] = lat / 90.0
    df["lon_norm"] = lon / 180.0
    df["alt_norm"] = alt_m / 1000.0

    input_cols = list(ERA5_FRACTION_COLS)
    input_cols += [f"{col}_lag1" for col in ERA5_FRACTION_COLS]
    input_cols += ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "lat_norm", "lon_norm", "alt_norm"]
    return df, input_cols


def main():
    args = parse_args()
    device = choose_device()
    print(f"Device: {device}")

    era5 = pd.read_csv(args.era5_csv)
    era5["datetime"] = to_ist_series(era5["datetime"])
    era5 = ensure_station_id(era5, args.default_station_id)
    era5 = era5.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    if args.lat is None or args.lon is None:
        first_station_id = era5["station_id"].dropna().iloc[0]
        match = next((station for station in STATIONS if station["id"] == first_station_id), None)
        if match is None:
            raise ValueError(
                "Custom station_id requires --lat and --lon so mapper station features can be built."
            )
        lat = match["lat"]
        lon = match["lon"]
        alt_m = args.alt_m if args.alt_m is not None else match["alt_m"]
    else:
        lat = args.lat
        lon = args.lon
        nearest = nearest_station(lat, lon)
        alt_m = args.alt_m if args.alt_m is not None else nearest["alt_m"]
        print(
            f"Using custom site features lat={lat}, lon={lon}, alt_m={alt_m} "
            f"(nearest configured station: {nearest['id']})"
        )

    era5, input_cols = build_mapper_features(era5, lat=lat, lon=lon, alt_m=alt_m)
    era5 = era5.dropna(subset=input_cols).sort_values(["station_id", "datetime"]).reset_index(drop=True)
    print(f"Rows after feature prep: {len(era5)}")
    print(f"Using {len(input_cols)} input features")

    mapper: TAFResNet = load_frozen_mapper(device, input_size=len(input_cols))
    scaler = joblib.load(CHECKPOINT_DIR / f"era5_scaler_{VERSION}.pkl")

    iso_path = CHECKPOINT_DIR / f"isotonic_calibrators_{VERSION}.pkl"
    calibrators = joblib.load(iso_path) if iso_path.exists() else None

    features_scaled = scaler.transform(era5[input_cols].values.astype(np.float32))
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    era5_raw = torch.tensor(era5[["total_cloud_cover"]].values.astype(np.float32), dtype=torch.float32)

    mapper.eval()
    preds = []
    batch_size = 4096
    with torch.no_grad():
        for start in range(0, len(era5), batch_size):
            batch_feat = features_tensor[start:start + batch_size].to(device)
            batch_raw = era5_raw[start:start + batch_size].to(device)
            batch_pred = mapper(batch_feat, era5_raw=batch_raw)
            preds.append(batch_pred.cpu().numpy())

    cloud_cover = np.concatenate(preds).flatten()
    if calibrators is not None:
        cloud_cover = calibrators[0].predict(cloud_cover)

    out = pd.DataFrame({
        # Keep IST wall-clock timestamps when writing CSV; using `.values` on a
        # tz-aware Series converts to UTC-normalized naive datetimes.
        "datetime": era5["datetime"].dt.tz_localize(None),
        "station_id": era5["station_id"].values,
        "cloud_cover": np.clip(cloud_cover, 0.0, 1.0),
    })
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved → {args.output_csv}")
    print(f"Shape: {out.shape}")


if __name__ == "__main__":
    main()
