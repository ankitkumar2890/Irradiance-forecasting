import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read a missing-timestamps report and check which expected HDF5 files are still missing."
    )
    parser.add_argument(
        "report",
        type=Path,
        help="Path to the .txt report created by hdf5_to_csv_merger.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a filtered txt containing only the files still missing.",
    )
    return parser.parse_args()


def load_report(report_path):
    entries = []
    with report_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            day_folder, time_utc, expected_file, expected_path = line.split(",", 3)
            entries.append(
                {
                    "day_folder": day_folder,
                    "time_utc": time_utc,
                    "expected_file": expected_file,
                    "expected_path": Path(expected_path),
                }
            )
    return entries


def write_unresolved_report(output_path, unresolved_entries):
    with output_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("# Unresolved missing MOSDAC HDF5 files\n")
        file_obj.write("# format=day_folder,time_utc,expected_file,expected_path\n")
        for entry in unresolved_entries:
            line = ",".join(
                [
                    entry["day_folder"],
                    entry["time_utc"],
                    entry["expected_file"],
                    str(entry["expected_path"]),
                ]
            )
            file_obj.write(f"{line}\n")


def main():
    args = parse_args()
    entries = load_report(args.report)
    unresolved_entries = [entry for entry in entries if not entry["expected_path"].exists()]

    print(f"Report entries: {len(entries)}")
    print(f"Still missing: {len(unresolved_entries)}")

    for entry in unresolved_entries:
        print(
            f"{entry['time_utc']} | {entry['expected_file']} | {entry['expected_path']}"
        )

    if args.output:
        write_unresolved_report(args.output, unresolved_entries)
        print(f"Saved unresolved report: {args.output}")


if __name__ == "__main__":
    main()
