import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data_store/fno_dataset_2005.npz"

print("Loading dataset...")
data = np.load(DATA_PATH)

X = data["X"]
Y = data["Y"]

print("\n==============================")
print("BASIC INFO")
print("==============================")

print("X shape:", X.shape)
print("Y shape:", Y.shape)

N, C, H, W = X.shape

print(f"Samples (N): {N}")
print(f"Channels (C): {C}")
print(f"Spatial size: {H} x {W}")

# --------------------------------------------------
# NaN Statistics
# --------------------------------------------------

print("\n==============================")
print("NaN CHECK")
print("==============================")

print("X NaN ratio:", np.isnan(X).mean())
print("Y NaN ratio:", np.isnan(Y).mean())

# Per-sample NaN ratio in Y
for i in range(N):
    nan_ratio = np.isnan(Y[i]).mean()
    print(f"Sample {i} Y NaN ratio: {nan_ratio:.4f}")

# --------------------------------------------------
# Value Ranges
# --------------------------------------------------

print("\n==============================")
print("VALUE RANGES")
print("==============================")

for ch in range(C):
    ch_min = np.nanmin(X[:, ch])
    ch_max = np.nanmax(X[:, ch])
    ch_mean = np.nanmean(X[:, ch])
    print(f"X channel {ch}: min={ch_min:.4f}, max={ch_max:.4f}, mean={ch_mean:.4f}")

print("\nY (COT, log-transformed)")
print("Min:", np.nanmin(Y))
print("Max:", np.nanmax(Y))
print("Mean:", np.nanmean(Y))

# --------------------------------------------------
# Check Spatial Alignment
# --------------------------------------------------

print("\n==============================")
print("SPATIAL CHECK")
print("==============================")

print("X spatial dims:", X.shape[-2:])
print("Y spatial dims:", Y.shape[-2:])

if X.shape[-2:] == Y.shape[-2:]:
    print("✓ Spatial dimensions match")
else:
    print("❌ Spatial mismatch!")

# --------------------------------------------------
# Visual Inspection
# --------------------------------------------------

if N > 0:
    idx = 0

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("ERA5 Channel 0 (interp)")
    plt.imshow(X[idx, 0], cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("MODIS COT (log)")
    plt.imshow(Y[idx, 0], cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Valid Mask")
    plt.imshow(~np.isnan(Y[idx, 0]), cmap="gray")

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Basic Correlation Check
# --------------------------------------------------

if N > 0:
    print("\n==============================")
    print("QUICK CORRELATION CHECK")
    print("==============================")

    # Flatten valid pixels
    mask = ~np.isnan(Y[0, 0])
    x_flat = X[0, 0][mask]
    y_flat = Y[0, 0][mask]

    if len(x_flat) > 10:
        corr = np.corrcoef(x_flat, y_flat)[0, 1]
        print("Correlation between ERA5 channel 0 and COT:", corr)
    else:
        print("Not enough valid pixels for correlation check.")

print("\nSanity check complete.")
