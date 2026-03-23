from pathlib import Path
import re
import numpy as np

BASE_DIR = Path("/Users/IRFAN/Desktop/Irradiance-forecasting/shards")
SHARD_DIR = BASE_DIR

DATASET_DIR = BASE_DIR / "dataset_new"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = DATASET_DIR / "train.npz"
VAL_OUT = DATASET_DIR / "validate.npz"
TEST_OUT = DATASET_DIR / "test.npz"

TRAIN_YEARS = set(range(2010, 2016))   # 2010..2015
VAL_YEARS = {2016, 2017}
TEST_YEARS = {2018, 2019}

def years_from_filename(name: str):
    # grabs all 4-digit years in filename
    ys = [int(y) for y in re.findall(r"(20\d{2})", name)]
    return ys

def split_for_year(y: int):
    if y in TRAIN_YEARS:
        return "train"
    if y in VAL_YEARS:
        return "validate"
    if y in TEST_YEARS:
        return "test"
    return None

def merge_by_year():
    files = sorted(SHARD_DIR.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {SHARD_DIR}")

    buckets = {
        "train": {"X": [], "Y": [], "tile_id": [], "timestamp": [], "year": []},
        "validate": {"X": [], "Y": [], "tile_id": [], "timestamp": [], "year": []},
        "test": {"X": [], "Y": [], "tile_id": [], "timestamp": [], "year": []},
    }

    print(f"Reading shards from: {SHARD_DIR}")
    for f in files:
        yrs = years_from_filename(f.name)
        if not yrs:
            print(f"Skipping (no year in name): {f.name}")
            continue

        data = np.load(f, allow_pickle=False)
        year_arr = data["year"] if "year" in data else None

        # If year metadata exists, use it (most reliable)
        if year_arr is not None:
            for split_name in buckets.keys():
                target_years = (
                    TRAIN_YEARS if split_name == "train"
                    else VAL_YEARS if split_name == "validate"
                    else TEST_YEARS
                )
                mask = np.isin(year_arr, list(target_years))
                n = int(mask.sum())
                if n == 0:
                    continue
                buckets[split_name]["X"].append(data["X"][mask])
                buckets[split_name]["Y"].append(data["Y"][mask])
                buckets[split_name]["tile_id"].append(data["tile_id"][mask])
                buckets[split_name]["timestamp"].append(data["timestamp"][mask])
                buckets[split_name]["year"].append(year_arr[mask])
                print(f"{f.name} -> {split_name}: {n} samples")
        else:
            # Fallback: route whole file by years in filename
            split_names = {split_for_year(y) for y in yrs}
            split_names.discard(None)
            if len(split_names) != 1:
                print(f"Skipping ambiguous file (no year key): {f.name}")
                continue
            split_name = split_names.pop()
            buckets[split_name]["X"].append(data["X"])
            buckets[split_name]["Y"].append(data["Y"])
            buckets[split_name]["tile_id"].append(data["tile_id"])
            buckets[split_name]["timestamp"].append(data["timestamp"])
            yfill = np.full(data["X"].shape[0], yrs[0], dtype=np.int16)
            buckets[split_name]["year"].append(yfill)
            print(f"{f.name} -> {split_name}: {data['X'].shape[0]} samples")

    def save_split(split_name, out_path):
        b = buckets[split_name]
        if not b["X"]:
            raise RuntimeError(f"No data collected for split: {split_name}")

        X = np.concatenate(b["X"], axis=0)
        Y = np.concatenate(b["Y"], axis=0)
        tile_id = np.concatenate(b["tile_id"], axis=0)
        timestamp = np.concatenate(b["timestamp"], axis=0)
        year = np.concatenate(b["year"], axis=0)

        np.savez_compressed(
            out_path,
            X=X,
            Y=Y,
            tile_id=tile_id,
            timestamp=timestamp,
            year=year,
        )
        print(f"\nSaved {split_name}: {out_path}")
        print(f"{split_name} X shape: {X.shape}")
        print(f"{split_name} years: {sorted(np.unique(year).tolist())}")

    save_split("train", TRAIN_OUT)
    save_split("validate", VAL_OUT)
    save_split("test", TEST_OUT)

    print("\nDone.")

if __name__ == "__main__":
    merge_by_year()
