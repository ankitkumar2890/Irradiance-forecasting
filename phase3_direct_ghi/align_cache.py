import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/IRFAN/Desktop/moirai_finetuning/phase3_direct_ghi")
sys.path.insert(0, str(BASE_DIR))

from config import MULTI_STATION_DOWNLOADS_DIR, STATIONS, YEARS, ERA5_CANONICAL_COLUMNS
LOCAL_TIMEZONE = "Asia/Kolkata"


def average_half_hour_to_top_of_hour(df: pd.DataFrame, value_cols: list) -> pd.DataFrame:
    """Average 09:30 and 10:30 values to create an aligned 10:00 timestamp."""
    out = df.sort_values("datetime").reset_index(drop=True).copy()
    minutes = sorted(out["datetime"].dt.minute.dropna().unique().tolist())
    if 30 not in minutes:
        return out  # Already aligned

    for col in value_cols:
        if col in out.columns:
            values = pd.to_numeric(out[col], errors="coerce")
            averaged = values.rolling(window=2).mean()
            if len(averaged) > 1:
                averaged.iloc[0] = values.iloc[0]
            out[col] = averaged

    out["datetime"] = out["datetime"] - pd.Timedelta(minutes=30)
    return out


print("=== Forcing all raw data in cache to HH:00 top-of-hour grid ===")

for station in STATIONS:
    sid = station["id"]
    sdir = MULTI_STATION_DOWNLOADS_DIR / sid
    
    for year in YEARS:
        # 1. GHI
        ghi_path = sdir / f"ghi_{year}.csv"
        if ghi_path.exists():
            ghi = pd.read_csv(ghi_path, parse_dates=["datetime"])
            if 30 in ghi["datetime"].dt.minute.unique():
                aligned = average_half_hour_to_top_of_hour(ghi, ["w_ghr"])
                aligned.to_csv(ghi_path, index=False)
                print(f"  [GHI] Aligned {sid} {year} to :00")

        # 2. ERA5
        era5_path = sdir / f"era5_{year}.csv"
        if era5_path.exists():
            era5 = pd.read_csv(era5_path, parse_dates=["datetime"])
            if 30 in era5["datetime"].dt.minute.unique():
                cols = list(ERA5_CANONICAL_COLUMNS.keys())
                aligned = average_half_hour_to_top_of_hour(era5, cols)
                aligned.to_csv(era5_path, index=False)
                print(f"  [ERA5] Aligned {sid} {year} to :00")

        # 3. Clear-sky (REGENERATE mathematically exact HH:00)
        cs_path = sdir / f"clearsky_{year}.csv"
        if cs_path.exists():
            cs = pd.read_csv(cs_path, parse_dates=["datetime"])
            if 30 in cs["datetime"].dt.minute.unique():
                from pvlib.location import Location
                times = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="1h", tz=LOCAL_TIMEZONE)
                site = Location(station["lat"], station["lon"], tz=LOCAL_TIMEZONE, altitude=station["alt_m"])
                solpos = site.get_solarposition(times)
                df = pd.DataFrame({
                    "datetime": times.tz_localize(None),
                    "zenith_angle": solpos["apparent_zenith"].values,
                    "azimuth_angle": solpos["azimuth"].values,
                })
                df.to_csv(cs_path, index=False)
                print(f"  [PVLib] Regenerated exactly at :00 for {sid} {year}")

print("\nDone! Now run:")
print("python phase3_direct_ghi/02_build_features.py")
print("python phase3_direct_ghi/03_build_dataset.py")
