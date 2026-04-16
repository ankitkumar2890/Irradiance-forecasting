import argparse
from pathlib import Path

from check_missing_h5_from_report import load_report, write_unresolved_report
from download_missing_h5_from_report import download_entries
from hdf5_to_csv_merger import export_period


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use an existing missing-files report to retry downloads, re-check what is still missing, "
            "and merge to CSV only when the period is complete."
        )
    )
    parser.add_argument("--year", type=int, required=True, help="Year folder to process, e.g. 2017")
    parser.add_argument(
        "--month",
        type=str,
        help="Optional 3-letter month code, e.g. DEC. If omitted, the full year is processed.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Existing missing-timestamps txt report created by scan_missing_h5_files.py",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Optional dataset root like data_2017/3RIMG_L2C_INS. Auto-detected if omitted.",
    )
    parser.add_argument("--target-lat", type=float, default=9.14)
    parser.add_argument("--target-lon", type=float, default=77.92)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Folder where merged outputs will be written.",
    )
    parser.add_argument(
        "--unresolved-output",
        type=Path,
        help="Optional txt file for unresolved missing files after retry download.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    month = args.month.upper() if args.month else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_entries = load_report(args.report)
    print(f"Report entries loaded: {len(report_entries)}")

    download_entries(report_entries)

    refreshed_entries = load_report(args.report)
    unresolved_entries = [entry for entry in refreshed_entries if not entry["expected_path"].exists()]

    print(f"Still missing after retry download: {len(unresolved_entries)}")

    if args.unresolved_output:
        write_unresolved_report(args.unresolved_output, unresolved_entries)
        print(f"Unresolved report: {args.unresolved_output}")

    if unresolved_entries:
        print("Merge skipped because some expected HDF5 files are still missing.")
        return

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
        require_complete=True,
    )

    print(f"Found files: {files_found}")
    print(f"Rows written: {rows_written}")
    print(f"Failed files: {failed_count}")
    print(f"Missing timestamps after final check: {missing_count}")
    print(f"Merged CSV: {csv_path}")
    print(f"Metadata JSON: {metadata_path}")
    print(f"Missing report: {missing_report_path}")


if __name__ == "__main__":
    main()
