import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# REAL DATASET (ERA5 → MODIS)
# =========================================================

class ERA5MODISDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)

        X = data["X"]      # [N, 9, H, W]
        Y = data["Y"]      # [N, 1, H, W]

        # Mask NaNs
        self.mask = ~np.isnan(Y)
        Y = np.nan_to_num(Y)
        X = np.nan_to_num(X)

        # Normalize ERA5 channels
        mean = X.mean(axis=(0,2,3), keepdims=True)
        std  = X.std(axis=(0,2,3), keepdims=True) + 1e-6
        X = (X - mean) / std

        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.mask = torch.tensor(self.mask, dtype=torch.float32)

        self.N, self.C, self.H, self.W = self.X.shape

        # Coordinate embedding (important)
        x = torch.linspace(0, 1, self.H)
        y = torch.linspace(0, 1, self.W)
        Xc, Yc = torch.meshgrid(x, y, indexing="ij")
        self.coords = torch.stack([Xc, Yc], dim=0)  # [2, H, W]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        m = self.mask[idx]

        # Add coordinates
        x = torch.cat([x, self.coords], dim=0)

        return x, y, m

# =========================================================
# SPECTRAL CONV
# =========================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels,
                modes1, modes2,
                dtype=torch.cfloat
            )
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            B, self.weights.shape[1],
            H, W//2 + 1,
            device=x.device,
            dtype=torch.cfloat
        )

        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights
        )

        out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return out

# =========================================================
# FNO MODEL
# =========================================================

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels):
        super().__init__()

        self.fc0 = nn.Linear(in_channels, width)

        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)

        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.fc0(x)
        x = x.permute(0,3,1,2)

        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))

        x = x.permute(0,2,3,1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0,3,1,2)

        return x

# =========================================================
# MASKED LOSS
# =========================================================

def masked_mse(pred, target, mask):
    return ((pred - target)**2 * mask).sum() / mask.sum()

# =========================================================
# TRAINING
# =========================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ERA5MODISDataset("data_store/fno_dataset_2005.npz")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = FNO2d(
        modes1=16,
        modes2=16,
        width=64,
        in_channels=9 + 2,  # ERA5 + coords
        out_channels=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        total_loss = 0
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = masked_mse(pred, y, m)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d} | Loss {total_loss/len(loader):.6f}")

    # Visual test
    x, y, m = dataset[0]
    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).cpu()[0]

    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    plt.title("Prediction")
    plt.imshow(pred[0], cmap="viridis")
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(y[0], cmap="viridis")
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.title("Mask")
    plt.imshow(m[0], cmap="gray")

    plt.show()

if __name__ == "__main__":
    main()
