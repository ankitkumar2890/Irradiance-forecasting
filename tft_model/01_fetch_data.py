"""Fetch and prepare multi-station ERA5, GHI, and clear-sky inputs for TFT."""
import calendar
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CDSAPI_KEY,
    ERA5_CANONICAL_COLUMNS,
    ERA5_SOURCE_FILE,
    MULTI_STATION_DOWNLOADS_DIR,
    NREL_API_KEY,
    NREL_EMAIL,
    STATIONS,
    YEARS,
)

ERA5_RAW_DIR = MULTI_STATION_DOWNLOADS_DIR / "_era5_raw"
ERA5_ENRICHED_SOURCE_FILE = MULTI_STATION_DOWNLOADS_DIR / "era5_2017_2019_tft.csv"
ERA5_VARIABLES = [
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
AREA = [17.0, 72.5, 8.0, 81.5]
GRID = [0.25, 0.25]
TARGET_HOURS = [f"{h:02d}:00" for h in range(24)]
LOCAL_TIMEZONE = "Asia/Kolkata"


def centered_hourly_average(ghi_df):
    """Convert :30 samples into top-of-hour values using adjacent averages."""
    out = ghi_df.sort_values("datetime").reset_index(drop=True).copy()
    minutes = sorted(out["datetime"].dt.minute.dropna().unique().tolist())
    if minutes != [30]:
        return out

    values = pd.to_numeric(out["w_ghr"], errors="coerce")
    averaged = values.rolling(window=2).mean()
    if len(averaged) > 1:
        averaged.iloc[0] = averaged.iloc[1]
    else:
        averaged.iloc[0] = values.iloc[0]

    out["datetime"] = out["datetime"] - pd.Timedelta(minutes=30)
    out["w_ghr"] = averaged.fillna(values).clip(lower=0)
    return out


def fetch_nrel_ghi(year, lat, lon, station_id):
    """Download hourly GHI from NREL NSRDB for one station-year."""
    assert NREL_API_KEY, "Set NREL_API_KEY env var."
    station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
    station_dir.mkdir(parents=True, exist_ok=True)

    print(f"  NREL GHI {year} for {station_id} ({lat}, {lon})...")
    url = "https://developer.nrel.gov/api/nsrdb/v2/solar/msg-iodc-download.csv"
    for interval in ("60", "30"):
        payload = {
            "api_key": NREL_API_KEY,
            "full_name": "Research User",
            "email": NREL_EMAIL,
            "affiliation": "Research",
            "reason": "Academic",
            "wkt": f"POINT({lon} {lat})",
            "names": str(year),
            "attributes": "ghi",
            "interval": interval,
            "utc": "false",
            "leap_day": "false",
        }
        response = requests.get(url, params=payload, timeout=180)
        if response.status_code == 200 and "Error" not in response.text[:200]:
            df = pd.read_csv(io.StringIO(response.text), skiprows=2)
            df.columns = [c.strip() for c in df.columns]
            df["datetime"] = pd.to_datetime(
                df[["Year", "Month", "Day", "Hour", "Minute"]].rename(
                    columns={
                        "Year": "year",
                        "Month": "month",
                        "Day": "day",
                        "Hour": "hour",
                        "Minute": "minute",
                    }
                )
            )
            ghi_col = next(c for c in df.columns if c.upper() == "GHI")
            df = df[["datetime", ghi_col]].rename(columns={ghi_col: "w_ghr"})
            df["w_ghr"] = pd.to_numeric(df["w_ghr"], errors="coerce").clip(lower=0)
            if df["datetime"].diff().median().total_seconds() < 3500:
                df = centered_hourly_average(df)
            out = station_dir / f"ghi_{year}.csv"
            df.to_csv(out, index=False)
            print(f"    Saved {out.name} shape={df.shape}")
            return
        print(f"    interval={interval} failed for {station_id}, retrying...")
        time.sleep(3)

    raise RuntimeError(f"NREL fetch failed for station={station_id}, year={year}")


def normalize_era5_columns(df):
    """Map alias columns onto the required TFT ERA5 feature names."""
    rename_map = {}
    missing = []
    for canonical, aliases in ERA5_CANONICAL_COLUMNS.items():
        found = next((alias for alias in aliases if alias in df.columns), None)
        if found is None:
            missing.append(f"{canonical} <- {aliases}")
        else:
            rename_map[found] = canonical

    return df.rename(columns=rename_map), missing


def to_ist_naive(values):
    """Return timestamps as IST wall-clock time without timezone metadata."""
    dt = pd.Series(pd.to_datetime(values))
    if getattr(dt.dt, "tz", None) is None:
        return dt.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
    return dt.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)


def source_has_required_era5_columns(path):
    """Quick check to see whether a CSV already satisfies TFT ERA5 requirements."""
    if not path.exists():
        return False
    df = pd.read_csv(path, nrows=2)
    _, missing = normalize_era5_columns(df)
    return not missing and "station_id" in df.columns and "datetime" in df.columns


def localized_time_index(values):
    """Return an Asia/Kolkata-aware index preserving IST wall-clock values."""
    dt = pd.Series(pd.to_datetime(values))
    if getattr(dt.dt, "tz", None) is None:
        return pd.DatetimeIndex(dt).tz_localize(LOCAL_TIMEZONE)
    return pd.DatetimeIndex(dt.dt.tz_convert(LOCAL_TIMEZONE))


def clearsky_time_grid(year, station_dir):
    """Use the ERA5 timestamp grid so clear-sky aligns with UTC to IST :30 rows."""
    era5_path = station_dir / f"era5_{year}.csv"
    if era5_path.exists():
        era5 = pd.read_csv(era5_path, usecols=["datetime"])
        return localized_time_index(era5["datetime"])
    return pd.date_range(
        f"{year}-01-01",
        f"{year}-12-31 23:00",
        freq="1h",
        tz="UTC",
    ).tz_convert(LOCAL_TIMEZONE)


def clearsky_matches_era5_grid(clearsky_path, era5_path):
    """Return True when cached clear-sky timestamps exactly match ERA5."""
    if not clearsky_path.exists() or not era5_path.exists():
        return False
    cs = pd.read_csv(clearsky_path, usecols=["datetime"])
    era5 = pd.read_csv(era5_path, usecols=["datetime"])
    return pd.to_datetime(cs["datetime"]).equals(pd.to_datetime(era5["datetime"]))


def ensure_cdsapirc():
    """Write a CDS API config for this session if a key is available."""
    if not CDSAPI_KEY:
        raise RuntimeError(
            "The ERA5 source is missing u10/v10, and CDSAPI_KEY is not set.\n"
            "Set CDSAPI_KEY so TFT can download the missing ERA5 wind fields."
        )
    cdsapirc = Path.home() / ".cdsapirc"
    cdsapirc.write_text(f"url: https://cds.climate.copernicus.eu/api\nkey: {CDSAPI_KEY}\n")


def download_era5_year(year):
    """Download monthly ERA5 NetCDF files including TFT wind variables."""
    import cdsapi

    ensure_cdsapirc()
    client = cdsapi.Client()
    year_dir = ERA5_RAW_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    for month in range(1, 13):
        out_file = year_dir / f"{year}_{month:02d}.nc"
        if out_file.exists():
            print(f"    Skip ERA5 {year}-{month:02d} (cached)")
            continue

        ndays = calendar.monthrange(year, month)[1]
        request = {
            "product_type": "reanalysis",
            "variable": ERA5_VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, ndays + 1)],
            "time": TARGET_HOURS,
            "area": AREA,
            "grid": GRID,
            "data_format": "netcdf",
        }
        print(f"    Downloading ERA5 {year}-{month:02d} with wind fields...")
        client.retrieve("reanalysis-era5-single-levels", request, str(out_file))
        print(f"      Done: {out_file.name}")


def build_enriched_era5_source():
    """Create a TFT-specific multi-station ERA5 CSV that includes u10/v10."""
    import xarray as xr

    print("  Building TFT ERA5 source with cloud + wind fields...")
    ERA5_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        download_era5_year(year)

    all_dfs = []
    for year in YEARS:
        year_dir = ERA5_RAW_DIR / str(year)
        for month in range(1, 13):
            nc_file = year_dir / f"{year}_{month:02d}.nc"
            if not nc_file.exists():
                raise FileNotFoundError(f"Missing ERA5 file after download: {nc_file}")

            ds = xr.open_dataset(nc_file)
            lat_d = "latitude" if "latitude" in ds.dims else "lat"
            lon_d = "longitude" if "longitude" in ds.dims else "lon"

            for station in STATIONS:
                ds_pt = ds.sel(
                    {lat_d: station["lat"], lon_d: station["lon"]},
                    method="nearest",
                ).squeeze(drop=True)
                df = ds_pt.to_dataframe().reset_index()
                df["station_id"] = station["id"]

                for tc in ("valid_time", "time", "forecast_time"):
                    if tc in df.columns:
                        df = df.rename(columns={tc: "datetime"})
                        break
                else:
                    raise RuntimeError(f"No time column found in {nc_file}")

                drop_cols = {
                    lat_d,
                    lon_d,
                    "level",
                    "number",
                    "step",
                    "surface",
                    "expver",
                    "pressure_level",
                }
                df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
                all_dfs.append(df)
            ds.close()

    if not all_dfs:
        raise RuntimeError("No ERA5 data was extracted for the TFT source.")

    out_df = pd.concat(all_dfs, ignore_index=True)
    out_df["datetime"] = to_ist_naive(out_df["datetime"])

    out_df = out_df.rename(
        columns={
            "tcc": "tcc",
            "lcc": "lcc",
            "mcc": "mcc",
            "hcc": "hcc",
            "u10": "u10",
            "v10": "v10",
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
        }
    )
    keep_cols = ["datetime", "station_id", "tcc", "lcc", "mcc", "hcc", "u10", "v10"]
    missing_cols = [col for col in keep_cols if col not in out_df.columns]
    if missing_cols:
        raise RuntimeError(f"Extracted ERA5 data is still missing columns: {missing_cols}")

    out_df = (
        out_df[keep_cols]
        .dropna(subset=["datetime", "station_id"])
        .sort_values(["station_id", "datetime"])
        .reset_index(drop=True)
    )
    out_df.to_csv(ERA5_ENRICHED_SOURCE_FILE, index=False)
    print(f"    Saved enriched ERA5 source → {ERA5_ENRICHED_SOURCE_FILE}")
    return ERA5_ENRICHED_SOURCE_FILE


def get_or_build_era5_source():
    """Return a CSV source that satisfies TFT's required ERA5 columns."""
    if source_has_required_era5_columns(ERA5_SOURCE_FILE):
        return ERA5_SOURCE_FILE
    if source_has_required_era5_columns(ERA5_ENRICHED_SOURCE_FILE):
        return ERA5_ENRICHED_SOURCE_FILE

    print(
        "  Existing ERA5 source is missing some TFT-required columns.\n"
        f"  Requested source: {ERA5_SOURCE_FILE}"
    )
    return build_enriched_era5_source()


def load_direct_era5():
    """Split a raw multi-station ERA5 CSV into station-year TFT inputs."""
    source_path = get_or_build_era5_source()
    print(f"  Loading ERA5 source: {source_path}")
    df = pd.read_csv(source_path)
    if "station_id" not in df.columns:
        raise ValueError(f"{source_path} must contain a station_id column for multi-station TFT training.")

    df, missing = normalize_era5_columns(df)
    if missing:
        raise ValueError(
            "ERA5 source is missing required columns for TFT:\n  - "
            + "\n  - ".join(missing)
            + f"\nSource file: {source_path}"
        )

    df["datetime"] = to_ist_naive(df["datetime"])

    df = df[df["station_id"].isin([station["id"] for station in STATIONS])].copy()
    keep_cols = ["datetime", "station_id", *ERA5_CANONICAL_COLUMNS.keys()]
    df = df[keep_cols].sort_values(["station_id", "datetime"]).reset_index(drop=True)

    for station in STATIONS:
        station_id = station["id"]
        station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
        station_dir.mkdir(parents=True, exist_ok=True)
        station_df = df[df["station_id"] == station_id].copy()
        if station_df.empty:
            print(f"    WARNING: No ERA5 rows for station={station_id}")
            continue

        for year in YEARS:
            year_df = station_df[station_df["datetime"].dt.year == year].copy()
            if year_df.empty:
                print(f"    WARNING: No ERA5 rows for station={station_id}, year={year}")
                continue
            out = station_dir / f"era5_{year}.csv"
            year_df.drop(columns=["station_id"]).to_csv(out, index=False)
            print(f"    {station_id}/{out.name} shape={year_df.shape}")


def generate_clearsky(year, lat, lon, alt_m, station_id):
    """Compute clear-sky GHI and zenith angle using PVLib."""
    import pvlib
    from pvlib.location import Location

    station_dir = MULTI_STATION_DOWNLOADS_DIR / station_id
    station_dir.mkdir(parents=True, exist_ok=True)

    print(f"  PVLib clear-sky {year} for {station_id}...")
    times = clearsky_time_grid(year, station_dir)
    site = Location(lat, lon, tz=LOCAL_TIMEZONE, altitude=alt_m)
    cs = site.get_clearsky(times, model="ineichen")
    solpos = site.get_solarposition(times)
    df = pd.DataFrame(
        {
            "datetime": times.tz_localize(None),
            "clear_sky_ghi": cs["ghi"].values.clip(0),
            "zenith_angle": solpos["apparent_zenith"].values,
        }
    )
    out = station_dir / f"clearsky_{year}.csv"
    df.to_csv(out, index=False)
    print(f"    Saved {out.name} shape={df.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"  TFT ERA5 pipeline for years {YEARS}")
    print(f"  Stations: {', '.join(station['id'] for station in STATIONS)}")
    print("=" * 60)

    print("\n=== GHI (NREL NSRDB) ===")
    for station in STATIONS:
        for year in YEARS:
            out = MULTI_STATION_DOWNLOADS_DIR / station["id"] / f"ghi_{year}.csv"
            if out.exists():
                print(f"  Skip {station['id']}/{out.name} (cached)")
            else:
                fetch_nrel_ghi(year, station["lat"], station["lon"], station["id"])

    print("\n=== ERA5 (raw source CSV) ===")
    load_direct_era5()

    print("\n=== Clear-sky (PVLib) ===")
    for station in STATIONS:
        for year in YEARS:
            out = MULTI_STATION_DOWNLOADS_DIR / station["id"] / f"clearsky_{year}.csv"
            era5 = MULTI_STATION_DOWNLOADS_DIR / station["id"] / f"era5_{year}.csv"
            if clearsky_matches_era5_grid(out, era5):
                print(f"  Skip {station['id']}/{out.name} (cached, matches ERA5 grid)")
            else:
                generate_clearsky(year, station["lat"], station["lon"], station["alt_m"], station["id"])

    print("\n=== Verification ===")
    for station in STATIONS:
        for year in YEARS:
            for filename in [f"ghi_{year}.csv", f"era5_{year}.csv", f"clearsky_{year}.csv"]:
                path = MULTI_STATION_DOWNLOADS_DIR / station["id"] / filename
                if path.exists():
                    df = pd.read_csv(path, nrows=2)
                    print(f"  OK {station['id']}/{filename:<18} cols={list(df.columns)}")
                else:
                    print(f"  MISSING {station['id']}/{filename}")

    print("\nDone. Next: python tft_model/02_build_features.py")
