from pathlib import Path
import numpy as np

BASE_DIR = Path("/content/drive/MyDrive/Irradiance-forecasting/fno_dataset_10")
SHARD_DIR = BASE_DIR / "shards"

DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = DATASET_DIR / "train.npz"
VAL_OUT = DATASET_DIR / "validate.npz"


def merge_shards(pattern, out_file, label):
    files = sorted(SHARD_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shard files found in {SHARD_DIR} for pattern: {pattern}")

    X_list, Y_list, tile_list, ts_list, year_list = [], [], [], [], []

    print(f"\nMerging {label} shards from: {SHARD_DIR}")
    for f in files:
        data = np.load(f)
        X_list.append(data["X"])
        Y_list.append(data["Y"])
        tile_list.append(data["tile_id"])
        ts_list.append(data["timestamp"])
        year_list.append(data["year"] if "year" in data else np.full(data["X"].shape[0], -1, dtype=np.int16))
        print(f"- {f.name}: {data['X'].shape[0]} samples")

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    tile_id = np.concatenate(tile_list, axis=0)
    timestamp = np.concatenate(ts_list, axis=0)
    year = np.concatenate(year_list, axis=0)

    np.savez_compressed(out_file, X=X, Y=Y, tile_id=tile_id, timestamp=timestamp, year=year)

    print(f"\n{label} merged file saved: {out_file}")
    print(f"{label} X shape: {X.shape}")
    print(f"{label} Y shape: {Y.shape}")
    print(f"{label} samples: {X.shape[0]}")

def main():
    print(f"Shard dir: {SHARD_DIR}")
    print(f"Output dir: {BASE_DIR}")

    merge_shards("train_*.npz", TRAIN_OUT, "Train")
    merge_shards("val_*.npz", VAL_OUT, "Validation")
    print("\nDone merging all shards.")

if __name__ == "__main__":
    main()
