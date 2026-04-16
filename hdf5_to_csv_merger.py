import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_TARGET_LAT = 9.14
DEFAULT_TARGET_LON = 77.92
DEFAULT_DATASET_ROOTS = [
    Path("data/3RIMG_L2C_INS"),
    Path("data_2017/3RIMG_L2C_INS"),
]
FILENAME_PATTERN = re.compile(
    r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_L2C_INS_V\d+R\d+\.h5$"
)


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [decode_attr(item) for item in value]
    if isinstance(value, tuple):
        return [decode_attr(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract point data from MOSDAC HDF5 files, convert to CSV rows, "
            "and merge them in timestamp order for a month or year."
        )
    )
    parser.add_argument("--year", type=int, required=True, help="Year folder to process, e.g. 2017")
    parser.add_argument(
        "--month",
        type=str,
        help="Optional 3-letter month code, e.g. JAN. If omitted, the full year is processed.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Optional dataset root like data_2017/3RIMG_L2C_INS. Auto-detected if omitted.",
    )
    parser.add_argument("--target-lat", type=float, default=DEFAULT_TARGET_LAT)
    parser.add_argument("--target-lon", type=float, default=DEFAULT_TARGET_LON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Folder where merged CSV and metadata JSON will be written.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Only merge when no timestamps are missing for the selected period.",
    )
    return parser.parse_args()


def resolve_dataset_root(year, root_override):
    if root_override is not None:
        year_path = root_override / str(year)
        if not year_path.exists():
            raise FileNotFoundError(f"Year folder not found: {year_path}")
        return root_override

    for root in DEFAULT_DATASET_ROOTS:
        if (root / str(year)).exists():
            return root

    searched = ", ".join(str(root) for root in DEFAULT_DATASET_ROOTS)
    raise FileNotFoundError(f"Could not find year {year} under: {searched}")


def extract_timestamp(file_path):
    match = FILENAME_PATTERN.match(file_path.name)
    if not match:
        raise ValueError(f"Unexpected filename format: {file_path.name}")
    return datetime.strptime("".join(match.groups()), "%d%b%Y%H%M")


def filter_files_for_period(year_root, month=None):
    h5_files = []
    month = month.upper() if month else None

    for day_dir in sorted(path for path in year_root.iterdir() if path.is_dir()):
        if month and not day_dir.name.endswith(month):
            continue
        h5_files.extend(day_dir.glob("*.h5"))

    h5_files = sorted(h5_files, key=extract_timestamp)
    if not h5_files:
        scope = f"{year_root.name} {month}" if month else year_root.name
        raise FileNotFoundError(f"No .h5 files found for {scope}")
    return h5_files


def build_grid_index(file_path, target_lat, target_lon):
    with h5py.File(file_path, "r") as h5_file:
        x = h5_file["X"][:]
        y = h5_file["Y"][:]
        lon_grid, lat_grid = np.meshgrid(x, y)
        dist = (lat_grid - target_lat) ** 2 + (lon_grid - target_lon) ** 2
        row_idx, col_idx = np.unravel_index(np.argmin(dist), dist.shape)
        return row_idx, col_idx, float(lat_grid[row_idx, col_idx]), float(lon_grid[row_idx, col_idx])


def collect_file_inventory(file_path):
    inventory = {}
    with h5py.File(file_path, "r") as h5_file:
        for key, dataset in h5_file.items():
            if not isinstance(dataset, h5py.Dataset):
                continue
            inventory[key] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
                "attrs": {attr_key: decode_attr(attr_val) for attr_key, attr_val in dataset.attrs.items()},
            }
    return inventory


def normalize_dataset_value(dataset, value):
    if isinstance(value, np.generic):
        value = value.item()

    fill_value = dataset.attrs.get("_FillValue")
    if fill_value is not None:
        fill_value = np.asarray(fill_value).reshape(-1)[0]
        if np.isclose(value, fill_value):
            return np.nan

    return value


def extract_point_row(file_path, row_idx, col_idx, grid_lat, grid_lon, target_lat, target_lon):
    time_utc = extract_timestamp(file_path)
    row = {
        "source_file": file_path.name,
        "source_day_folder": file_path.parent.name,
        "time_utc": time_utc,
        "time_ist": time_utc + pd.Timedelta(hours=5, minutes=30),
        "target_lat": target_lat,
        "target_lon": target_lon,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "row_index": row_idx,
        "col_index": col_idx,
    }

    with h5py.File(file_path, "r") as h5_file:
        for key, dataset in h5_file.items():
            if not isinstance(dataset, h5py.Dataset):
                continue

            data = dataset[()]
            column_name = key.lower()

            if data.ndim == 3 and data.shape[0] == 1:
                row[column_name] = normalize_dataset_value(dataset, data[0, row_idx, col_idx])
            elif data.ndim == 1 and data.shape[0] == 1:
                row[column_name] = normalize_dataset_value(dataset, data[0])

    return row


def build_output_paths(output_dir, year, month):
    period_label = f"{year}_{month.lower()}" if month else str(year)
    csv_path = output_dir / f"tirunelveli_all_fields_30min_IST_{period_label}.csv"
    metadata_path = output_dir / f"tirunelveli_hdf5_inventory_{period_label}.json"
    missing_report_path = output_dir / f"tirunelveli_missing_timestamps_{period_label}.txt"
    return csv_path, metadata_path, missing_report_path


def expected_day_timestamps(reference_timestamp):
    day_start = reference_timestamp.replace(hour=0, minute=15)
    day_end = reference_timestamp.replace(hour=23, minute=45)
    timestamps = []
    current = day_start
    while current <= day_end:
        timestamps.append(current)
        current += timedelta(minutes=30)
    return timestamps


def build_expected_filename(timestamp):
    return f"3RIMG_{timestamp.strftime('%d%b%Y_%H%M').upper()}_L2C_INS_V01R00.h5"


def detect_missing_timestamps(files):
    files_by_day = {}
    for file_path in files:
        files_by_day.setdefault(file_path.parent.name, set()).add(extract_timestamp(file_path))

    missing_entries = []
    for day_dir_name in sorted(files_by_day):
        actual_timestamps = files_by_day[day_dir_name]
        reference_timestamp = min(actual_timestamps)
        for expected_timestamp in expected_day_timestamps(reference_timestamp):
            if expected_timestamp in actual_timestamps:
                continue
            missing_entries.append(
                {
                    "day_folder": day_dir_name,
                    "time_utc": expected_timestamp,
                    "expected_file": build_expected_filename(expected_timestamp),
                }
            )
    return missing_entries


def write_missing_report(report_path, year_root, missing_entries):
    with report_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("# Missing MOSDAC HDF5 timestamps\n")
        file_obj.write(f"# dataset_root={year_root}\n")
        file_obj.write("# format=day_folder,time_utc,expected_file,expected_path\n")
        for entry in missing_entries:
            expected_path = year_root / entry["day_folder"] / entry["expected_file"]
            line = ",".join(
                [
                    entry["day_folder"],
                    entry["time_utc"].strftime("%Y-%m-%d %H:%M:%S"),
                    entry["expected_file"],
                    str(expected_path),
                ]
            )
            file_obj.write(f"{line}\n")


def reorder_columns(dataframe):
    preferred_columns = [
        "source_file",
        "source_day_folder",
        "time_utc",
        "time_ist",
        "target_lat",
        "target_lon",
        "grid_lat",
        "grid_lon",
        "row_index",
        "col_index",
        "ghi",
        "dhi",
        "dni",
        "ins",
        "time",
        "projection_information",
        "proj_dim",
    ]
    ordered_columns = [col for col in preferred_columns if col in dataframe.columns]
    ordered_columns.extend(col for col in dataframe.columns if col not in ordered_columns)
    return dataframe[ordered_columns]


def export_period(year, month, root, target_lat, target_lon, output_dir, require_complete=False):
    dataset_root = resolve_dataset_root(year, root)
    year_root = dataset_root / str(year)
    files = filter_files_for_period(year_root, month)

    output_dir.mkdir(parents=True, exist_ok=True)
    row_idx, col_idx, grid_lat, grid_lon = build_grid_index(files[0], target_lat, target_lon)
    inventory = collect_file_inventory(files[0])

    rows = []
    failed_files = []
    for file_path in files:
        try:
            rows.append(
                extract_point_row(
                    file_path=file_path,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    grid_lat=grid_lat,
                    grid_lon=grid_lon,
                    target_lat=target_lat,
                    target_lon=target_lon,
                )
            )
        except Exception as exc:
            failed_files.append({"file": str(file_path), "error": str(exc)})

    if not rows:
        raise RuntimeError("No rows could be extracted from the selected files.")

    dataframe = pd.DataFrame(rows).sort_values("time_utc").reset_index(drop=True)
    dataframe = reorder_columns(dataframe)

    missing_entries = detect_missing_timestamps(files)
    csv_path, metadata_path, missing_report_path = build_output_paths(output_dir, year, month)
    write_missing_report(missing_report_path, year_root, missing_entries)

    if require_complete and missing_entries:
        raise RuntimeError(
            f"Missing timestamps found ({len(missing_entries)}). "
            f"See report: {missing_report_path}"
        )

    dataframe.to_csv(csv_path, index=False)

    metadata = {
        "year": year,
        "month": month,
        "dataset_root": str(dataset_root),
        "year_root": str(year_root),
        "files_processed": len(rows),
        "files_found": len(files),
        "missing_timestamps_count": len(missing_entries),
        "failed_files": failed_files,
        "target_lat": target_lat,
        "target_lon": target_lon,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "row_index": row_idx,
        "col_index": col_idx,
        "datasets": inventory,
    }
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(to_builtin(metadata), file_obj, indent=2)

    return csv_path, metadata_path, missing_report_path, len(rows), len(files), len(failed_files), len(missing_entries)


def main():
    args = parse_args()
    month = args.month.upper() if args.month else None
    (
        csv_path,
        metadata_path,
        missing_report_path,
        rows_written,
        files_found,
        failed_count,
        missing_count,
    ) = export_period(
        year=args.year,
        month=month,
        root=args.root,
        target_lat=args.target_lat,
        target_lon=args.target_lon,
        output_dir=args.output_dir,
        require_complete=args.require_complete,
    )

    print(f"Found files: {files_found}")
    print(f"Rows written: {rows_written}")
    print(f"Failed files: {failed_count}")
    print(f"Missing timestamps: {missing_count}")
    print(f"Merged CSV: {csv_path}")
    print(f"Metadata JSON: {metadata_path}")
    print(f"Missing report: {missing_report_path}")


if __name__ == "__main__":
    main()
