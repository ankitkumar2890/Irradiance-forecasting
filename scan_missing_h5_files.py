import argparse
from pathlib import Path

from hdf5_to_csv_merger import (
    build_output_paths,
    detect_missing_timestamps,
    filter_files_for_period,
    resolve_dataset_root,
    write_missing_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan a MOSDAC month/year folder and write expected missing HDF5 files to a txt report."
    )
    parser.add_argument("--year", type=int, required=True, help="Year folder to scan, e.g. 2017")
    parser.add_argument(
        "--month",
        type=str,
        help="Optional 3-letter month code, e.g. DEC. If omitted, the full year is scanned.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Optional dataset root like data_2017/3RIMG_L2C_INS. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Folder where the missing-timestamps txt file will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    month = args.month.upper() if args.month else None
    dataset_root = resolve_dataset_root(args.year, args.root)
    year_root = dataset_root / str(args.year)
    files = filter_files_for_period(year_root, month)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing_entries = detect_missing_timestamps(files)
    _, _, report_path = build_output_paths(args.output_dir, args.year, month)
    write_missing_report(report_path, year_root, missing_entries)

    print(f"Found files: {len(files)}")
    print(f"Missing timestamps: {len(missing_entries)}")
    print(f"Missing report: {report_path}")


if __name__ == "__main__":
    main()
