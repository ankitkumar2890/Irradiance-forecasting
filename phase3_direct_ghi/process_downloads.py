"""Process 15-min raw downloads into hourly data.

For GHI and clear-sky (which may be at 15-min resolution):
  - For each whole hour (0:00, 1:00, 2:00, ...),
    take the average of the value at (hour-15min), (hour), and (hour+15min).
  - This produces a smooth hourly value from the 3 surrounding 15-min samples.

ERA5 is already hourly, so it is copied as-is.

Phase 3 addition: also processes azimuth_angle in clear-sky data.

Output goes to: downloads/processed/<station>/
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import MULTI_STATION_DOWNLOADS_DIR, PROCESSED_DOWNLOADS_DIR, STATIONS, YEARS


def process_ghi(station_id: str, year: int, out_dir: Path) -> None:
    """Average 15-min GHI to hourly using ±15 min window around each whole hour."""
    src = MULTI_STATION_DOWNLOADS_DIR / station_id / f"ghi_{year}.csv"
    out = out_dir / f"ghi_{year}.csv"
    if out.exists():
        print(f"  Already exists, skipping: {out.name}")
        return

    df = pd.read_csv(src, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()

    # For each whole hour, average the 3 points: hour-15m, hour, hour+15m
    whole_hours = df.index[df.index.minute == 0]
    rows = []
    for ts in whole_hours:
        ts_before = ts - pd.Timedelta(minutes=15)
        ts_after = ts + pd.Timedelta(minutes=15)

        vals = []
        for t in [ts_before, ts, ts_after]:
            if t in df.index:
                vals.append(df.loc[t, "w_ghr"])

        avg = sum(vals) / len(vals) if vals else 0.0
        rows.append({"datetime": ts, "w_ghr": round(avg, 4)})

    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    print(f"  Saved {out.name}  ({len(result)} rows)")


def process_clearsky(station_id: str, year: int, out_dir: Path) -> None:
    """Average 15-min zenith & azimuth to hourly using ±15 min window."""
    src = MULTI_STATION_DOWNLOADS_DIR / station_id / f"clearsky_{year}.csv"
    out = out_dir / f"clearsky_{year}.csv"
    if out.exists():
        print(f"  Already exists, skipping: {out.name}")
        return

    df = pd.read_csv(src, parse_dates=["datetime"])
    has_azimuth = "azimuth_angle" in df.columns
    df = df.set_index("datetime").sort_index()

    whole_hours = df.index[df.index.minute == 0]
    rows = []
    for ts in whole_hours:
        ts_before = ts - pd.Timedelta(minutes=15)
        ts_after = ts + pd.Timedelta(minutes=15)

        zen_vals, azi_vals = [], []
        for t in [ts_before, ts, ts_after]:
            if t in df.index:
                zen_vals.append(df.loc[t, "zenith_angle"])
                if has_azimuth:
                    azi_vals.append(df.loc[t, "azimuth_angle"])

        row = {
            "datetime": ts,
            "zenith_angle": round(sum(zen_vals) / len(zen_vals), 4) if zen_vals else 90.0,
        }
        if has_azimuth:
            row["azimuth_angle"] = round(sum(azi_vals) / len(azi_vals), 4) if azi_vals else 180.0
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    print(f"  Saved {out.name}  ({len(result)} rows)")


def copy_era5(station_id: str, year: int, out_dir: Path) -> None:
    """ERA5 is already hourly — just copy it to the processed folder."""
    src = MULTI_STATION_DOWNLOADS_DIR / station_id / f"era5_{year}.csv"
    out = out_dir / f"era5_{year}.csv"
    if out.exists():
        print(f"  Already exists, skipping: {out.name}")
        return

    import shutil
    shutil.copy2(src, out)
    print(f"  Copied {out.name}")


def main() -> None:
    print("=== Phase 3: Processing downloads → hourly (±15 min average) ===\n")

    for station in STATIONS:
        sid = station["id"]
        for year in YEARS:
            print(f"[{sid} {year}]")

            out_dir = PROCESSED_DOWNLOADS_DIR / sid
            out_dir.mkdir(parents=True, exist_ok=True)

            process_ghi(sid, year, out_dir)
            process_clearsky(sid, year, out_dir)
            copy_era5(sid, year, out_dir)

    print("\n✅ Done! Processed files are in downloads/processed/<station>/")


if __name__ == "__main__":
    main()
