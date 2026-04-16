import argparse
from pathlib import Path

import requests

import test as mosdac_client
from check_missing_h5_from_report import load_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retry download only the HDF5 files listed in a missing-timestamps report."
    )
    parser.add_argument(
        "report",
        type=Path,
        help="Path to the missing-timestamps txt report.",
    )
    return parser.parse_args()

def fetch_records_for_day(day_text):
    params = {
        "datasetId": mosdac_client.datasetId,
        "startTime": day_text,
        "endTime": day_text,
    }
    optional_parameters = {
        "boundingBox": mosdac_client.boundingBox,
        "gId": mosdac_client.gId,
    }
    params.update({key: value for key, value in optional_parameters.items() if value})
    records_by_identifier = {}
    start_index = 1

    while True:
        paged_params = dict(params)
        paged_params["startIndex"] = start_index

        response = requests.get(mosdac_client.search_url, params=paged_params, timeout=30)
        if response.status_code >= 400:
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
            raise requests.HTTPError(
                f"Search API failed for {day_text} with status {response.status_code}: {payload}",
                response=response,
            )

        payload = response.json()
        entries = payload.get("entries", [])
        for entry in entries:
            records_by_identifier[entry["identifier"]] = entry

        items_per_page = payload.get("itemsPerPage") or len(entries)
        total_results = payload.get("totalResults", len(records_by_identifier))

        if not entries or len(records_by_identifier) >= total_results:
            break

        start_index += items_per_page

    print(f"Fetched {len(records_by_identifier)} records from search API for {day_text}.")
    return records_by_identifier


def download_entries(entries):
    unresolved_entries = [entry for entry in entries if not entry["expected_path"].exists()]
    if not unresolved_entries:
        print("No unresolved files left in the report.")
        return

    result = mosdac_client.get_token()
    if result is None:
        raise RuntimeError("Could not log in to MOSDAC.")

    tokens, username = result
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    print(f"Login successful for {username}.")

    day_record_cache = {}
    try:
        total = len(unresolved_entries)
        for index, entry in enumerate(unresolved_entries, start=1):
            identifier = entry["expected_file"]
            day_text = entry["time_utc"][:10]

            if day_text not in day_record_cache:
                day_record_cache[day_text] = fetch_records_for_day(day_text)

            record = day_record_cache[day_text].get(identifier)
            if not record:
                print(f"[WARNING] Could not find metadata for {identifier} on {day_text}.")
                continue

            record_id = record["id"]
            prod_date = record.get("updated")
            result = mosdac_client.download_data(access_token, record_id, identifier, prod_date, index, total)
            status = result["status"]

            if status == "Invalid/Expired Token":
                new_tokens = mosdac_client.refresh_access_token(refresh_token)
                if not new_tokens:
                    print(f"[ERROR] Token refresh failed while downloading {identifier}.")
                    continue
                access_token = new_tokens["access_token"]
                refresh_token = new_tokens["refresh_token"]
                result = mosdac_client.download_data(access_token, record_id, identifier, prod_date, index, total)
                status = result["status"]

            if status not in ("downloaded", "exists"):
                print(f"[WARNING] Download did not complete for {identifier}. Status: {status}")
    finally:
        mosdac_client.logout()


def main():
    args = parse_args()
    entries = load_report(args.report)
    download_entries(entries)


if __name__ == "__main__":
    main()
