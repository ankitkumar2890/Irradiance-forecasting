# clean after sanity check

from pathlib import Path
import numpy as np
import hashlib

BASE = Path("/content/drive/MyDrive/Irradiance-forecasting/fno_dataset_10/dataset")
FILES = [
    ("train.npz", "train_clean.npz"),
    ("validate.npz", "validate_clean.npz"),
]

# Tuning
CONST_STD_TOL = 1e-8     # constant tile threshold
ROUND_DECIMALS = 6       # for robust duplicate hashing (float noise safe)

def sample_hash(x_sample: np.ndarray) -> str:
    # x_sample shape: (C,H,W)
    xq = np.round(x_sample, ROUND_DECIMALS).astype(np.float32, copy=False)
    return hashlib.blake2b(xq.tobytes(), digest_size=16).hexdigest()

def clean_file(in_name, out_name):
    in_path = BASE / in_name
    out_path = BASE / out_name

    data = np.load(in_path, allow_pickle=False)
    X = data["X"]              # (N,C,H,W)
    Y = data["Y"]
    tile_id = data["tile_id"]
    timestamp = data["timestamp"]
    year = data["year"] if "year" in data else np.full(X.shape[0], -1, dtype=np.int16)

    N = X.shape[0]
    print(f"\nProcessing {in_name}: N={N}")

    keep_idx = []
    seen = set()
    n_const = 0
    n_dup = 0

    # per-sample std over C,H,W
    sample_std = np.std(X.reshape(N, -1), axis=1)

    for i in range(N):
        # 1) remove constant / near-constant X
        if sample_std[i] <= CONST_STD_TOL:
            n_const += 1
            continue

        # 2) remove duplicate X
        h = sample_hash(X[i])
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        keep_idx.append(i)

    keep_idx = np.array(keep_idx, dtype=np.int64)

    Xc = X[keep_idx]
    Yc = Y[keep_idx]
    tc = tile_id[keep_idx]
    tsc = timestamp[keep_idx]
    yc = year[keep_idx]

    np.savez_compressed(
        out_path,
        X=Xc,
        Y=Yc,
        tile_id=tc,
        timestamp=tsc,
        year=yc,
    )

    print(f"Saved: {out_path}")
    print(f"Removed constants: {n_const}")
    print(f"Removed duplicates: {n_dup}")
    print(f"Final N: {Xc.shape[0]}")

def main():
    for in_name, out_name in FILES:
        clean_file(in_name, out_name)

if __name__ == "__main__":
    main()
