"""Process NSRDB raw downloads into hourly data.

For GHI fields that are sub-hourly:
  - For each whole hour (0:00, 1:00, 2:00, ...),
    take the average of the value at (hour-15min), (hour), and (hour+15min).
  - This produces a smooth hourly value from the surrounding NSRDB samples.

ERA5 is already hourly, so it is copied as-is.

Output goes to: downloads/processed/<station>/
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import MULTI_STATION_DOWNLOADS_DIR, STATIONS, YEARS

PROCESSED_GHI_REQUIRED_COLUMNS = {"datetime", "w_ghr", "nsrdb_clearsky_ghi", "zenith_angle"}


def process_ghi(station_id: str, year: int, out_dir: Path) -> None:
    """Average 15-min NSRDB GHI fields to hourly using a ±15 min window."""
    src = MULTI_STATION_DOWNLOADS_DIR / station_id / f"ghi_{year}.csv"
    out = out_dir / f"ghi_{year}.csv"
    if out.exists():
        existing = pd.read_csv(out, nrows=2)
        missing = sorted(PROCESSED_GHI_REQUIRED_COLUMNS.difference(existing.columns))
        if not missing:
            print(f"  Already exists, skipping: {out.name}")
            return
        print(f"  Regenerating {out.name} (missing {missing})")

    df = pd.read_csv(src, parse_dates=["datetime"])
    missing_raw_cols = sorted({"w_ghr", "nsrdb_clearsky_ghi", "zenith_angle"}.difference(df.columns))
    if missing_raw_cols:
        raise ValueError(
            f"{src} is missing required NSRDB columns: {missing_raw_cols}. "
            "Re-run phase2_finetuning/01_fetch_data.py to refresh the raw GHI download."
        )
    df = df.set_index("datetime").sort_index()

    # For each whole hour, average the 3 points: hour-15m, hour, hour+15m
    whole_hours = df.index[df.index.minute == 0]
    rows = []
    for ts in whole_hours:
        ts_before = ts - pd.Timedelta(minutes=15)
        ts_after = ts + pd.Timedelta(minutes=15)

        # Gather available values from the 3 timestamps
        vals = []
        cs_vals = []
        zen_vals = []
        for t in [ts_before, ts, ts_after]:
            if t in df.index:
                vals.append(df.loc[t, "w_ghr"])
                cs_vals.append(df.loc[t, "nsrdb_clearsky_ghi"])
                zen_vals.append(df.loc[t, "zenith_angle"])

        avg = sum(vals) / len(vals) if vals else 0.0
        row = {"datetime": ts, "w_ghr": round(avg, 4)}
        avg_cs = sum(cs_vals) / len(cs_vals) if cs_vals else 0.0
        row["nsrdb_clearsky_ghi"] = round(avg_cs, 4)
        row["zenith_angle"] = round(sum(zen_vals) / len(zen_vals), 4) if zen_vals else 90.0
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
    print("=== Processing 15-min downloads → hourly (±15 min average) ===\n")

    for station in STATIONS:
        sid = station["id"]
        for year in YEARS:
            print(f"[{sid} {year}]")

            out_dir = MULTI_STATION_DOWNLOADS_DIR.parent / "processed" / sid
            out_dir.mkdir(parents=True, exist_ok=True)

            process_ghi(sid, year, out_dir)
            copy_era5(sid, year, out_dir)

    print("\n✅ Done! Processed files are in downloads/processed/<station>/")


if __name__ == "__main__":
    main()
