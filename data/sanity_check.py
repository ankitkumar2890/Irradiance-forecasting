import numpy as np
from pathlib import Path

DATA_PATH = "/content/drive/MyDrive/Irradiance-forecasting/fno_dataset_10/dataset/train_clean.npz"
RNG_SEED = 42
N_RANDOM_SAMPLES = 8
MIN_VALID_PIXELS_FOR_CORR = 100

np.random.seed(RNG_SEED)

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def stats(name, arr):
    finite = np.isfinite(arr)
    n_total = arr.size
    n_finite = finite.sum()
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()

    print(f"{name}:")
    print(f"  shape={arr.shape}, dtype={arr.dtype}")
    print(f"  finite_ratio={n_finite / n_total:.6f}, nan_ratio={n_nan / n_total:.6f}, inf_ratio={n_inf / n_total:.6f}")

    if n_finite > 0:
        vals = arr[finite]
        print(f"  min={vals.min():.6f}, max={vals.max():.6f}, mean={vals.mean():.6f}, std={vals.std():.6f}")
        p = np.percentile(vals, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print("  percentiles[0,1,5,25,50,75,95,99,100]=" + np.array2string(p, precision=4, separator=", "))
    else:
        print("  WARNING: no finite values")

def ensure_nchw(x, y, problems):
    """
    Returns X_nchw, Y_nchw
    Supports:
      X in NCHW or NHWC (detected independently of Y)
      Y in NCHW or NHW (will be converted to N1HW)
      X and Y MAY have different spatial dims (super-resolution tasks)
    """
    if y.ndim == 3:
        y = y[:, None, :, :]
    elif y.ndim != 4:
        problems.append(f"Y must be 3D or 4D, got ndim={y.ndim}")
        return None, None

    if x.ndim != 4:
        problems.append(f"X must be 4D, got ndim={x.ndim}")
        return None, None

    y_h, y_w = y.shape[-2], y.shape[-1]

    # Case 1: X and Y share the same spatial dims (standard case)
    if x.shape[-2:] == (y_h, y_w):
        return x, y

    # Case 2: X is NHWC matching Y spatial dims
    if x.shape[1:3] == (y_h, y_w):
        x = np.moveaxis(x, -1, 1)  # NHWC -> NCHW
        return x, y

    # Case 3: X and Y have DIFFERENT spatial dims (super-resolution / downscaling)
    # Assume X is already NCHW (most common convention for smaller grids)
    x_h, x_w = x.shape[-2], x.shape[-1]
    print(f"  NOTE: X spatial ({x_h}x{x_w}) != Y spatial ({y_h}x{y_w})")
    print(f"  Resolution ratio: ~{y_h / x_h:.1f}x in height, ~{y_w / x_w:.1f}x in width")
    print(f"  Assuming NCHW for both (super-resolution / downscaling task)")
    return x, y

def check_duplicates(arr_nchw, name, problems, tol=1e-12):
    # arr_nchw: (N,C,H,W)
    n = arr_nchw.shape[0]
    flat = arr_nchw.reshape(n, -1)

    finite = np.isfinite(flat).all(axis=1)
    if not finite.any():
        problems.append(f"{name}: no fully-finite samples for duplicate check")
        return

    idx = np.where(finite)[0]
    f = flat[idx]

    # quick signature: rounded mean/std/sum to identify likely duplicates
    sig = np.stack(
        [
            np.round(f.mean(axis=1), 8),
            np.round(f.std(axis=1), 8),
            np.round(f.sum(axis=1), 8),
        ],
        axis=1,
    )

    _, counts = np.unique(sig, axis=0, return_counts=True)
    dup_groups = (counts > 1).sum()
    if dup_groups > 0:
        problems.append(f"{name}: possible duplicate groups found via signatures = {dup_groups}")

    # exact constant samples
    sample_std = np.nanstd(flat, axis=1)
    n_constant = np.sum(sample_std < tol)
    if n_constant > 0:
        problems.append(f"{name}: constant samples detected = {n_constant}/{n}")

def main():
    problems = []

    section("LOAD")
    p = Path(DATA_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = np.load(DATA_PATH, allow_pickle=False)
    keys = list(data.keys())
    print("Keys:", keys)

    if "X" not in data or "Y" not in data:
        raise KeyError("Dataset must contain keys 'X' and 'Y'.")

    X = data["X"]
    Y = data["Y"]

    section("RAW SHAPES")
    print(f"X raw shape: {X.shape}, dtype={X.dtype}")
    print(f"Y raw shape: {Y.shape}, dtype={Y.dtype}")

    X, Y = ensure_nchw(X, Y, problems)
    if X is None or Y is None:
        section("FAILED EARLY")
        for i, msg in enumerate(problems, 1):
            print(f"{i}. {msg}")
        return

    section("NORMALIZED SHAPES (NCHW)")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    NX, CX, HX, WX = X.shape
    NY, CY, HY, WY = Y.shape

    # Basic alignment checks
    section("ALIGNMENT CHECKS")
    print(f"Samples: X={NX}, Y={NY}")
    print(f"Channels: X={CX}, Y={CY}")
    print(f"Spatial: X=({HX},{WX}), Y=({HY},{WY})")

    if NX != NY:
        problems.append(f"Sample count mismatch: X has {NX}, Y has {NY}")
    if (HX, WX) != (HY, WY):
        print(f"  INFO: Different spatial resolutions (super-resolution task)")
        print(f"  Resolution ratio: ~{HY/HX:.1f}x height, ~{WY/WX:.1f}x width")
    if CY != 1:
        print(f"WARNING: Y has {CY} channels (expected usually 1 for single target map)")

    # Global stats
    section("GLOBAL STATS")
    stats("X", X)
    stats("Y", Y)

    # Per-channel stats
    section("PER-CHANNEL STATS (X)")
    for c in range(CX):
        stats(f"X[:, {c}, :, :]", X[:, c, :, :])

    section("PER-CHANNEL STATS (Y)")
    for c in range(CY):
        stats(f"Y[:, {c}, :, :]", Y[:, c, :, :])

    # Per-sample valid coverage
    section("PER-SAMPLE VALIDITY (Y)")
    y_valid_ratio = np.isfinite(Y).mean(axis=(1, 2, 3))
    print(f"valid_ratio min={y_valid_ratio.min():.6f}, max={y_valid_ratio.max():.6f}, mean={y_valid_ratio.mean():.6f}")
    low_valid = np.where(y_valid_ratio < 0.05)[0]
    if len(low_valid) > 0:
        problems.append(f"Y has {len(low_valid)} samples with <5% valid pixels")

    # Random sample correlation checks
    same_spatial = (HX, WX) == (HY, WY)
    if same_spatial:
        section("RANDOM SAMPLE CORRELATION CHECK (X channel 0 vs Y channel 0) — pixel-wise")
        n = min(NX, NY)
        k = min(N_RANDOM_SAMPLES, n)
        chosen = np.random.choice(n, size=k, replace=False)

        corr_values = []
        for idx in chosen:
            x0 = X[idx, 0]
            y0 = Y[idx, 0]
            mask = np.isfinite(x0) & np.isfinite(y0)
            n_valid = mask.sum()

            if n_valid < MIN_VALID_PIXELS_FOR_CORR:
                print(f"sample {idx}: skipped (valid_pixels={n_valid})")
                continue

            xv = x0[mask]
            yv = y0[mask]

            if np.std(xv) == 0 or np.std(yv) == 0:
                print(f"sample {idx}: skipped (zero variance)")
                continue

            corr = np.corrcoef(xv, yv)[0, 1]
            corr_values.append(corr)
            print(f"sample {idx}: corr={corr:.6f}, valid_pixels={n_valid}")

        if len(corr_values) == 0:
            problems.append("No usable samples for pixel-wise correlation check")
        else:
            corr_values = np.array(corr_values)
            print(f"corr mean={corr_values.mean():.6f}, min={corr_values.min():.6f}, max={corr_values.max():.6f}")
    else:
        section("CROSS-RESOLUTION CORRELATION CHECK (sample-level means)")
        print("  X and Y have different spatial dims; using per-sample mean correlation.")
        n = min(NX, NY)
        x_means = np.array([np.nanmean(X[i, 0]) for i in range(n)])
        y_means = np.array([np.nanmean(Y[i, 0]) for i in range(n)])
        both_finite = np.isfinite(x_means) & np.isfinite(y_means)
        if both_finite.sum() < 10:
            problems.append(f"Too few finite sample means for cross-resolution correlation ({both_finite.sum()})")
        else:
            xm = x_means[both_finite]
            ym = y_means[both_finite]
            if np.std(xm) == 0 or np.std(ym) == 0:
                print("  Skipped: zero variance in sample means")
            else:
                corr = np.corrcoef(xm, ym)[0, 1]
                print(f"  Sample-mean correlation (X ch0 vs Y ch0): {corr:.6f} (n={both_finite.sum()})")

    # Duplicates / constants
    section("DUPLICATE / CONSTANT CHECKS")
    check_duplicates(X, "X", problems)
    check_duplicates(Y, "Y", problems)
    print("Completed duplicate/constant heuristics")

    # Optional time consistency checks if keys exist
    section("OPTIONAL TIME KEY CHECKS")
    time_keys = [k for k in keys if "time" in k.lower() or "date" in k.lower() or "timestamp" in k.lower()]
    if len(time_keys) == 0:
        print("No time-like keys found.")
    else:
        print("Time-like keys:", time_keys)
        for k in time_keys:
            arr = data[k]
            print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
            if arr.shape[0] != NX and arr.shape[0] != NY:
                problems.append(f"Time key '{k}' length {arr.shape[0]} does not match samples")

    # Final verdict
    section("FINAL VERDICT")
    if len(problems) == 0:
        print("PASS: No structural problems detected by this sanity check.")
    else:
        print(f"FOUND {len(problems)} POTENTIAL PROBLEM(S):")
        for i, msg in enumerate(problems, 1):
            print(f"{i}. {msg}")

if __name__ == "__main__":
    main()
