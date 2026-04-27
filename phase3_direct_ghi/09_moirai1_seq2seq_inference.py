"""Run validation inference for the custom Moirai 1.1 seq2seq forecaster."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    CHECKPOINT_DIR,
    CLUSTER_NAME,
    DATASET_DIR,
    DAYLIGHT_ZENITH_DEG,
    GHI_SCALE_FACTOR,
    NIGHT_ZENITH_DEG,
    RESULTS_DIR,
)
from moirai1_1_seq2seq import Moirai1Seq2Seq  # noqa: E402


def _build_daylight_mask(future_zenith_deg: torch.Tensor) -> torch.Tensor:
    daylight_span = max(NIGHT_ZENITH_DEG - DAYLIGHT_ZENITH_DEG, 1e-6)
    mask = (NIGHT_ZENITH_DEG - future_zenith_deg.float()) / daylight_span
    return mask.clamp(min=0.0, max=1.0)


class InferenceDataset(Dataset):
    def __init__(self, split: str) -> None:
        self.X_past = np.load(DATASET_DIR / f"X_past_{split}.npy")
        self.X_future = np.load(DATASET_DIR / f"X_future_{split}.npy")
        self.y_future = np.load(DATASET_DIR / f"y_future_{split}.npy")
        self.times = np.load(DATASET_DIR / f"times_{split}.npy", allow_pickle=True).astype("datetime64[ns]")
        self.station_ids = np.load(DATASET_DIR / f"station_ids_{split}.npy", allow_pickle=True)

    def __len__(self) -> int:
        return len(self.X_past)

    def __getitem__(self, idx: int) -> dict[str, object]:
        x_past = torch.from_numpy(self.X_past[idx]).float()
        x_future = torch.from_numpy(self.X_future[idx]).float()
        return {
            "past_target": x_past[:, :1] / GHI_SCALE_FACTOR,
            "past_dynamic_real": x_past[:, 1:],
            "future_dynamic_real": x_future,
            "target": torch.from_numpy(self.y_future[idx]).float(),
            "daylight_mask": _build_daylight_mask(x_future[:, 0]),
            "times": self.times[idx],
            "station_id": str(self.station_ids[idx]),
        }


def _collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "past_target": torch.stack([item["past_target"] for item in batch]),
        "past_dynamic_real": torch.stack([item["past_dynamic_real"] for item in batch]),
        "future_dynamic_real": torch.stack([item["future_dynamic_real"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "daylight_mask": torch.stack([item["daylight_mask"] for item in batch]),
        "times": [item["times"] for item in batch],
        "station_id": [item["station_id"] for item in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--checkpoint-name", default=f"moirai1_seq2seq_{CLUSTER_NAME}.pt")
    parser.add_argument(
        "--output-csv",
        default=str(RESULTS_DIR / f"moirai1_seq2seq_{CLUSTER_NAME}_val_predictions.csv"),
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device_arg == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    checkpoint = torch.load(CHECKPOINT_DIR / args.checkpoint_name, map_location=device)

    model = Moirai1Seq2Seq(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = InferenceDataset(args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )

    rows: list[dict[str, object]] = []
    all_preds: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    with torch.no_grad():
        row_offset = 0
        for batch in loader:
            past_target = batch["past_target"].to(device)
            past_dynamic_real = batch["past_dynamic_real"].to(device)
            future_dynamic_real = batch["future_dynamic_real"].to(device)
            daylight_mask = batch["daylight_mask"].to(device)
            target = batch["target"].to(device)

            preds = model(past_target, past_dynamic_real, future_dynamic_real)
            preds = torch.clamp(preds * daylight_mask, min=0.0)

            preds_np = (preds.cpu().numpy() * GHI_SCALE_FACTOR).astype(np.float32)
            target_np = (target.cpu().numpy() * GHI_SCALE_FACTOR).astype(np.float32)
            all_preds.append(preds_np)
            all_true.append(target_np)

            batch_size = preds_np.shape[0]
            for batch_idx in range(batch_size):
                sample_times = batch["times"][batch_idx]
                station_id = batch["station_id"][batch_idx]
                for horizon_idx, pred_value in enumerate(preds_np[batch_idx]):
                    ts = pd.Timestamp(sample_times[horizon_idx])
                    rows.append(
                        {
                            "station_id": station_id,
                            "datetime": ts,
                            "lead_time_h": horizon_idx + 1,
                            "GHI_true": float(target_np[batch_idx, horizon_idx]),
                            "GHI_pred": float(pred_value),
                        }
                    )
            row_offset += batch_size
            if row_offset % 256 == 0:
                print(f"  processed {row_offset}/{len(dataset)} windows")

    pred_all = np.concatenate(all_preds, axis=0)
    true_all = np.concatenate(all_true, axis=0)
    rmse = math.sqrt(float(np.mean(np.square(pred_all - true_all))))
    mae = float(np.mean(np.abs(pred_all - true_all)))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved predictions -> {out_path}")
    print(f"RMSE={rmse:.3f} W/m²  MAE={mae:.3f} W/m²")


if __name__ == "__main__":
    main()
