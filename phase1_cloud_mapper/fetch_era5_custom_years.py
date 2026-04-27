"""Fetch and extract ERA5 for arbitrary years and an optional single site."""

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from fetch_data import download_era5_year, extract_era5_to_csv, ERA5_OUTPUT_COLS
from time_utils import to_ist_series


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Years to fetch, e.g. --years 2024 2025",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="era5_custom.csv",
        help="Output CSV filename written into phase1_cloud_mapper/downloads/",
    )
    parser.add_argument("--lat", type=float, default=None, help="Optional single-site latitude.")
    parser.add_argument("--lon", type=float, default=None, help="Optional single-site longitude.")
    parser.add_argument(
        "--station-id",
        type=str,
        default="site_1",
        help="Station id to use for single-site extraction.",
    )
    return parser.parse_args()


def extract_era5_single_site(years: list[int], output_filename: str, lat: float, lon: float, station_id: str) -> None:
    base_dir = Path(__file__).resolve().parent
    era5_dir = base_dir / "era5"
    downloads_dir = base_dir / "downloads"

    frames = []
    print(f"  Extracting ERA5 {years} for {station_id} at ({lat}, {lon}) → {output_filename}")
    for year in years:
        year_dir = era5_dir / str(year)
        for month in range(1, 13):
            nc_file = year_dir / f"{year}_{month:02d}.nc"
            if not nc_file.exists():
                print(f"    Missing: {nc_file}")
                continue

            ds = xr.open_dataset(nc_file, engine="netcdf4")
            lat_d = "latitude" if "latitude" in ds.dims else "lat"
            lon_d = "longitude" if "longitude" in ds.dims else "lon"
            ds_pt = ds.sel({lat_d: lat, lon_d: lon}, method="nearest").squeeze(drop=True)
            df = ds_pt.to_dataframe().reset_index()
            ds.close()

            for tc in ("valid_time", "time", "forecast_time"):
                if tc in df.columns:
                    df = df.rename(columns={tc: "datetime"})
                    break
            else:
                raise RuntimeError(f"No time column found in {nc_file}")

            df = df.rename(columns={
                "tcc": "total_cloud_cover",
                "lcc": "low_cloud_cover",
                "mcc": "medium_cloud_cover",
                "hcc": "high_cloud_cover",
                "tclw": "cloud_liquid_water",
                "tciw": "cloud_ice_water",
                "tcwv": "water_vapour",
                "total_column_cloud_liquid_water": "cloud_liquid_water",
                "total_column_cloud_ice_water": "cloud_ice_water",
                "total_column_water_vapour": "water_vapour",
            })
            df["station_id"] = station_id
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No ERA5 data extracted for years={years}")

    out_df = pd.concat(frames, ignore_index=True)
    out_df["datetime"] = to_ist_series(out_df["datetime"])
    output_cols = ["datetime", "station_id"] + [col for col in ERA5_OUTPUT_COLS if col in out_df.columns]
    out_df = (
        out_df[output_cols]
        .dropna(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    out_path = downloads_dir / output_filename
    out_df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(f"Shape: {out_df.shape}")
    print(f"Range: {out_df['datetime'].iloc[0]} → {out_df['datetime'].iloc[-1]}")


def main():
    args = parse_args()
    years = sorted(set(args.years))
    print(f"Fetching ERA5 years: {years}")
    for year in years:
        download_era5_year(year)
    if args.lat is not None or args.lon is not None:
        if args.lat is None or args.lon is None:
            raise ValueError("Provide both --lat and --lon for single-site extraction.")
        extract_era5_single_site(years, args.output_filename, args.lat, args.lon, args.station_id)
    else:
        extract_era5_to_csv(years, args.output_filename)
        out_path = Path(__file__).resolve().parent / "downloads" / args.output_filename
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
