"""Prepare GHI, direct ERA5, and clear-sky inputs for the ERA5-only experiment."""
import io, time, sys
import pandas as pd
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DOWNLOADS_DIR, ERA5_SOURCE_FILE,
    SELECTED_STATION, FINETUNE_STATION,
    YEARS, NREL_API_KEY, NREL_EMAIL,
)


def fetch_nrel_ghi(year, lat, lon):
    """Download hourly GHI from NREL NSRDB for one year."""
    assert NREL_API_KEY, "Set NREL_API_KEY env var"
    print(f"  NREL GHI {year} ({lat}, {lon})...")
    url = "https://developer.nrel.gov/api/nsrdb/v2/solar/msg-iodc-download.csv"
    for interval in ("60", "30"):
        payload = {
            "api_key": NREL_API_KEY, "full_name": "Research User",
            "email": NREL_EMAIL, "affiliation": "Research", "reason": "Academic",
            "wkt": f"POINT({lon} {lat})", "names": str(year),
            "attributes": "ghi", "interval": interval,
            "utc": "false", "leap_day": "false",
        }
        r = requests.get(url, params=payload, timeout=180)
        if r.status_code == 200 and "Error" not in r.text[:200]:
            df = pd.read_csv(io.StringIO(r.text), skiprows=2)
            df.columns = [c.strip() for c in df.columns]
            df["datetime"] = pd.to_datetime(
                df[["Year", "Month", "Day", "Hour", "Minute"]].rename(
                    columns={"Year": "year", "Month": "month", "Day": "day",
                             "Hour": "hour", "Minute": "minute"})
            )
            ghi_col = next(c for c in df.columns if c.upper() == "GHI")
            df = df[["datetime", ghi_col]].rename(columns={ghi_col: "w_ghr"})
            df["w_ghr"] = pd.to_numeric(df["w_ghr"], errors="coerce").clip(lower=0)
            if df["datetime"].diff().median().total_seconds() < 3500:
                df = df.set_index("datetime").resample("1h").mean().reset_index()
            out = DOWNLOADS_DIR / f"ghi_{year}.csv"
            df.to_csv(out, index=False)
            print(f"    ghi_{year}.csv  shape={df.shape}")
            return
        print(f"    interval={interval} failed, retrying...")
        time.sleep(3)
    raise RuntimeError(f"NREL fetch failed for {year}")


def load_direct_era5():
    """Load raw ERA5 for the selected station and save yearly CSVs."""
    if not ERA5_SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Phase 1 ERA5 source not found: {ERA5_SOURCE_FILE}\n"
            "Run Phase 1 fetch_data.py first."
        )

    print(f"  Loading direct ERA5 from Phase 1: {ERA5_SOURCE_FILE.name}")
    df = pd.read_csv(ERA5_SOURCE_FILE)
    # Normalize the Phase 1 timezone-aware timestamps onto the same :00 clock
    # used by the direct fine-tuning pipeline.
    dt = pd.to_datetime(df["datetime"])
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    df["datetime"] = dt

    if "station_id" in df.columns:
        df = df[df["station_id"] == FINETUNE_STATION].copy()
        print(f"    Filtered to station: {FINETUNE_STATION}")

    for year in YEARS:
        year_df = df[df["datetime"].dt.year == year].copy()
        if year_df.empty:
            print(f"    WARNING: No data for {year}")
            continue
        out_cols = [
            "datetime",
            "total_cloud_cover",
            "low_cloud_cover",
            "medium_cloud_cover",
            "high_cloud_cover",
            "cloud_liquid_water",
            "cloud_ice_water",
            "water_vapour",
        ]
        year_df = year_df[out_cols].sort_values("datetime").reset_index(drop=True)
        out = DOWNLOADS_DIR / f"era5_{year}.csv"
        year_df.to_csv(out, index=False)
        print(f"    era5_{year}.csv  shape={year_df.shape}")


def generate_clearsky(year, lat, lon, alt_m):
    """Compute clear-sky GHI and zenith angle using PVLib Ineichen model."""
    import pvlib
    from pvlib.location import Location
    print(f"  PVLib clear-sky {year}...")
    times = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00",
                          freq="1h", tz="Asia/Kolkata")
    site = Location(lat, lon, tz="Asia/Kolkata", altitude=alt_m)
    cs = site.get_clearsky(times, model="ineichen")
    solpos = site.get_solarposition(times)
    df = pd.DataFrame({
        "datetime": times.tz_localize(None),
        "clear_sky_ghi": cs["ghi"].values.clip(0),
        "zenith_angle": solpos["apparent_zenith"].values,
    })
    out = DOWNLOADS_DIR / f"clearsky_{year}.csv"
    df.to_csv(out, index=False)
    print(f"    clearsky_{year}.csv  shape={df.shape}")


if __name__ == "__main__":
    lat = SELECTED_STATION["lat"]
    lon = SELECTED_STATION["lon"]
    alt = SELECTED_STATION["alt_m"]

    print("=" * 55)
    print(f"  Phase 2 — Fetching data for {YEARS}")
    print(f"  Station: {FINETUNE_STATION} ({lat}, {lon})")
    print("=" * 55)

    # 1. GHI from NREL (one per year)
    print("\n=== GHI (NREL NSRDB) ===")
    for year in YEARS:
        out = DOWNLOADS_DIR / f"ghi_{year}.csv"
        if out.exists():
            print(f"  Skip ghi_{year}.csv (cached)")
        else:
            fetch_nrel_ghi(year, lat, lon)

    # 2. Synthetic ICON from Phase 1
    print("\n=== Direct ERA5 (Phase 1 raw source) ===")
    load_direct_era5()

    # 3. Clear-sky (local PVLib computation)
    print("\n=== Clear-sky (PVLib) ===")
    for year in YEARS:
        generate_clearsky(year, lat, lon, alt)

    # 4. Verify
    print("\n=== Verification ===")
    for year in YEARS:
        for f in [f"ghi_{year}.csv", f"era5_{year}.csv", f"clearsky_{year}.csv"]:
            p = DOWNLOADS_DIR / f
            if p.exists():
                df = pd.read_csv(p, nrows=2)
                print(f"  OK  {f:<25}  cols={list(df.columns)}")
            else:
                print(f"  MISSING  {f}")

    print("\nDone. Next: python 02_build_features.py")
