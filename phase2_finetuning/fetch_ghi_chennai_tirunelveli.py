"""
Fetch hourly NREL NSRDB GHI CSVs for both Chennai and Tirunelveli.

Outputs:
  /Users/IRFAN/Desktop/moirai_finetuning/multi_station_ghi/chennai/ghi_2017.csv
  /Users/IRFAN/Desktop/moirai_finetuning/multi_station_ghi/chennai/ghi_2018.csv
  /Users/IRFAN/Desktop/moirai_finetuning/multi_station_ghi/chennai/ghi_2019.csv
  /Users/IRFAN/Desktop/moirai_finetuning/multi_station_ghi/tirunelveli/ghi_2017.csv
  /Users/IRFAN/Desktop/moirai_finetuning/multi_station_ghi/tirunelveli/ghi_2018.csv
  /Users/IRFAN/Desktop/moirai_finetuning/multi_station_ghi/tirunelveli/ghi_2019.csv

Required env vars:
  NREL_API_KEY
  NREL_EMAIL
"""

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import NREL_API_KEY, NREL_EMAIL, PROJECT_ROOT, YEARS


TARGET_STATIONS = [
    {"id": "chennai", "lat": 13.08, "lon": 80.27},
    {"id": "tirunelveli", "lat": 9.14, "lon": 77.92},
]

OUTPUT_ROOT = PROJECT_ROOT / "multi_station_ghi"


def fetch_nrel_ghi(year: int, lat: float, lon: float, out_path: Path) -> None:
    """Download hourly GHI from NREL NSRDB for one year and one location."""
    if not NREL_API_KEY:
        raise RuntimeError("Set NREL_API_KEY before running this script.")
    if not NREL_EMAIL:
        raise RuntimeError("Set NREL_EMAIL before running this script.")

    url = "https://developer.nrel.gov/api/nsrdb/v2/solar/msg-iodc-download.csv"
    print(f"  Fetching {year} for ({lat}, {lon}) -> {out_path}")

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

            # Some responses come back at 30-minute resolution; normalize to hourly.
            if df["datetime"].diff().median().total_seconds() < 3500:
                df = df.set_index("datetime").resample("1h").mean().reset_index()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"    saved {out_path.name} shape={df.shape}")
            return

        print(f"    interval={interval} failed, retrying...")
        time.sleep(3)

    raise RuntimeError(f"NREL fetch failed for {year} at ({lat}, {lon})")


def main() -> None:
    print("=" * 72)
    print("Fetching GHI for Chennai and Tirunelveli (2017, 2018, 2019)")
    print(f"Output root: {OUTPUT_ROOT}")
    print("=" * 72)

    for station in TARGET_STATIONS:
        station_dir = OUTPUT_ROOT / station["id"]
        print(f"\n=== {station['id']} ({station['lat']}, {station['lon']}) ===")
        for year in YEARS:
            out_path = station_dir / f"ghi_{year}.csv"
            if out_path.exists():
                print(f"  Skip {out_path.name} (cached)")
                continue
            fetch_nrel_ghi(year, station["lat"], station["lon"], out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
