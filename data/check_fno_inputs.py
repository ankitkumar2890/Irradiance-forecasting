from pathlib import Path
import numpy as np


BASE = Path("/Users/IRFAN/Desktop/Irradiance-forecasting/dataset_new")
FILES = ["train_clean.npz", "validate_clean.npz", "test_clean.npz"]
MIN_EXPECTED_CHANNELS_WITH_PRITHVI = 11  # 10 ERA5 + at least 1 Prithvi PCA channel


def load_split(path: Path):
    d = np.load(path, allow_pickle=False)
    return d["X"], d["Y"], d["timestamp"], d["tile_id"], d["year"]


def main():
    missing = [f for f in FILES if not (BASE / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files under {BASE}: {missing}")

    keys_by_split = {}
    print(f"Checking dataset inputs in: {BASE}")
    for fname in FILES:
        path = BASE / fname
        X, Y, ts, tile_id, year = load_split(path)
        split = fname.replace(".npz", "")
        keys = np.array([f"{t}|{int(i)}" for t, i in zip(ts, tile_id)], dtype=object)
        keys_by_split[split] = set(keys.tolist())

        print(f"\n{fname}")
        print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
        print(f"  years: {sorted(set(map(int, year.tolist())))}")
        print(f"  channels: {X.shape[1]}")
        if X.shape[1] < MIN_EXPECTED_CHANNELS_WITH_PRITHVI:
            print(
                "  WARNING: Prithvi channels appear to be missing "
                f"(expected >= {MIN_EXPECTED_CHANNELS_WITH_PRITHVI}, got {X.shape[1]})."
            )

    print("\nCross-split overlap by (timestamp,tile_id):")
    split_names = list(keys_by_split.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a = split_names[i]
            b = split_names[j]
            overlap = len(keys_by_split[a] & keys_by_split[b])
            print(f"  {a} vs {b}: {overlap}")


if __name__ == "__main__":
    main()
