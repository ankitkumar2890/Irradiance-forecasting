"""
TFT Dataset — PyTorch Dataset wrapping the existing .npy sliding-window files.

Reads X_past_{split}.npy, X_future_{split}.npy, y_future_{split}.npy produced
by phase2_finetuning/03_build_dataset.py and exposes them as (encoder_input,
decoder_input, target) tensors suitable for the TFT model.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATASET_DIR, PAST_STEPS, FUTURE_STEPS, BATCH_SIZE, NUM_WORKERS


class CAFTimeSeriesDataset(Dataset):
    """
    Wraps the pre-built .npy sliding windows for TFT consumption.

    Shapes loaded from disk:
        X_past   : (N, PAST_STEPS, 8)   — encoder observations
        X_future : (N, FUTURE_STEPS, 7)  — known future covariates
        y_future : (N, FUTURE_STEPS)     — CAF targets

    Returns per sample:
        encoder_input : (PAST_STEPS, 8)       float32
        decoder_input : (FUTURE_STEPS, 7)     float32
        target        : (FUTURE_STEPS,)       float32
    """

    def __init__(self, split: str):
        """
        Parameters
        ----------
        split : str
            One of "train", "val", "test".
        """
        self.split = split

        x_past_path = DATASET_DIR / f"X_past_{split}.npy"
        x_future_path = DATASET_DIR / f"X_future_{split}.npy"
        y_future_path = DATASET_DIR / f"y_future_{split}.npy"

        if not x_past_path.exists():
            raise FileNotFoundError(
                f"Missing {x_past_path}. Run phase2_finetuning/03_build_dataset.py first."
            )

        self.X_past = np.load(x_past_path).astype(np.float32)
        self.X_future = np.load(x_future_path).astype(np.float32)
        self.y_future = np.load(y_future_path).astype(np.float32)

        # Optional: load timestamps (for inference/evaluation only)
        times_path = DATASET_DIR / f"times_{split}.npy"
        self.times = (
            np.load(times_path, allow_pickle=True) if times_path.exists() else None
        )

        assert len(self.X_past) == len(self.X_future) == len(self.y_future), (
            f"Length mismatch: X_past={len(self.X_past)}, "
            f"X_future={len(self.X_future)}, y_future={len(self.y_future)}"
        )
        assert self.X_past.shape[1] == PAST_STEPS, (
            f"Expected PAST_STEPS={PAST_STEPS}, got {self.X_past.shape[1]}"
        )
        assert self.X_future.shape[1] == FUTURE_STEPS, (
            f"Expected FUTURE_STEPS={FUTURE_STEPS}, got {self.X_future.shape[1]}"
        )

    def __len__(self):
        return len(self.X_past)

    def __getitem__(self, idx):
        encoder_input = torch.from_numpy(self.X_past[idx])      # (72, 8)
        decoder_input = torch.from_numpy(self.X_future[idx])     # (24, 7)
        target = torch.from_numpy(self.y_future[idx])            # (24,)
        return encoder_input, decoder_input, target


def get_dataloader(split: str, batch_size: int = BATCH_SIZE, shuffle: bool = None):
    """
    Convenience factory for a DataLoader.

    Parameters
    ----------
    split : str
        "train", "val", or "test".
    batch_size : int
        Batch size.
    shuffle : bool or None
        If None, shuffles for "train" only.
    """
    ds = CAFTimeSeriesDataset(split)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# ── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    for split in ["train", "val"]:
        ds = CAFTimeSeriesDataset(split)
        print(f"{split}: {len(ds)} samples")
        enc, dec, tgt = ds[0]
        print(f"  encoder_input: {enc.shape}  decoder_input: {dec.shape}  target: {tgt.shape}")

    loader = get_dataloader("train", batch_size=4)
    batch = next(iter(loader))
    print(f"\nBatch shapes: enc={batch[0].shape}, dec={batch[1].shape}, tgt={batch[2].shape}")
