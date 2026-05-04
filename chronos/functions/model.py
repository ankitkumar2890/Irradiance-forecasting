"""Chronos-Bolt LoRA fine-tuning for standalone Method 3."""

from __future__ import annotations

import importlib.util
import json
import random
import site
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def import_installed_chronos_package():
    for root_str in list(site.getsitepackages()) + [site.getusersitepackages()]:
        if not root_str:
            continue
        root = Path(root_str)
        init_path = root / "chronos" / "__init__.py"
        if not init_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            "_installed_chronos_pkg",
            init_path,
            submodule_search_locations=[str(init_path.parent)],
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError("Could not locate pip-installed chronos package.")


class WindowDataset(Dataset):
    def __init__(self, x_past: np.ndarray, y_future: np.ndarray, station_ids: np.ndarray | None = None):
        self.x_past = torch.tensor(x_past, dtype=torch.float32)
        self.y_future = torch.tensor(y_future, dtype=torch.float32)
        self.station_ids = None if station_ids is None else np.asarray(station_ids, dtype=object)

    def __len__(self):
        return len(self.x_past)

    def __getitem__(self, idx):
        if self.station_ids is None:
            return self.x_past[idx], self.y_future[idx]
        return self.x_past[idx], self.y_future[idx], str(self.station_ids[idx])


def select_device(device_arg: str, smoke_test: bool = False):
    if smoke_test:
        return torch.device("cpu")
    if device_arg != "auto":
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if device_arg == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_chronos_bolt_model(model_id: str, device: torch.device):
    chronos_pkg = import_installed_chronos_package()
    pipeline = chronos_pkg.ChronosBoltPipeline.from_pretrained(model_id, device_map=str(device))
    return pipeline.model


def inject_lora(model, *, lora_rank, lora_alpha, lora_dropout, lora_target_modules):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        target_modules=list(lora_target_modules),
        lora_dropout=float(lora_dropout),
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _median_quantile_index(model) -> int:
    quantiles = model.quantiles.detach().cpu().numpy().tolist()
    if 0.5 in quantiles:
        return quantiles.index(0.5)
    return len(quantiles) // 2


def _evaluate_rmse(model, loader, device: torch.device):
    model.eval()
    median_idx = _median_quantile_index(model)
    losses = []
    rmses = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                context, target, _station_id = batch
            else:
                context, target = batch
            context = context.to(device)
            target = target.to(device)
            out = model(context=context, target=target)
            losses.append(float(out.loss.detach().cpu().item()))
            pred = out.quantile_preds[:, median_idx, : target.shape[-1]]
            rmse = torch.sqrt(torch.mean((pred - target) ** 2, dim=-1)).mean()
            rmses.append(float(rmse.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf"), float(np.mean(rmses)) if rmses else float("inf")


def _build_station_balanced_sampler(station_ids: np.ndarray):
    station_ids = np.asarray(station_ids, dtype=object)
    unique_ids, counts = np.unique(station_ids, return_counts=True)
    freq = {station_id: count for station_id, count in zip(unique_ids.tolist(), counts.tolist())}
    weights = np.asarray([1.0 / freq[station_id] for station_id in station_ids.tolist()], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def finetune_chronos_bolt(
    *,
    dataset_dir,
    checkpoint_dir,
    model_id,
    device,
    smoke_test,
    max_train_windows,
    max_val_windows,
    lora_rank,
    lora_alpha,
    lora_dropout,
    lora_target_modules,
    ft_batch_size,
    ft_lr,
    ft_weight_decay,
    ft_max_epochs,
    ft_patience,
    ft_gradient_clip,
    ft_num_workers,
    ft_log_every,
    seed,
    train_ghi_mask_wm2,
    station_balanced_sampling,
):
    dataset_dir = Path(dataset_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(dataset_dir / "X_past_train.npy")
    y_train = np.load(dataset_dir / "y_future_train.npy")
    x_val = np.load(dataset_dir / "X_past_val.npy")
    y_val = np.load(dataset_dir / "y_future_val.npy")
    station_ids_train = np.load(dataset_dir / "station_ids_train.npy", allow_pickle=True)
    station_ids_val = np.load(dataset_dir / "station_ids_val.npy", allow_pickle=True)

    if max_train_windows:
        x_train, y_train = x_train[:max_train_windows], y_train[:max_train_windows]
        station_ids_train = station_ids_train[:max_train_windows]
    if max_val_windows:
        x_val, y_val = x_val[:max_val_windows], y_val[:max_val_windows]
        station_ids_val = station_ids_val[:max_val_windows]
    if smoke_test:
        x_train, y_train = x_train[:128], y_train[:128]
        x_val, y_val = x_val[:32], y_val[:32]
        station_ids_train = station_ids_train[:128]
        station_ids_val = station_ids_val[:32]
        ft_max_epochs = min(int(ft_max_epochs), 2)

    print(f"Loading Chronos-Bolt base model: {model_id}")
    base_model = load_chronos_bolt_model(model_id, device)
    model = inject_lora(
        base_model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    model.to(device)

    train_sampler = (
        _build_station_balanced_sampler(station_ids_train)
        if station_balanced_sampling
        else None
    )

    train_loader = DataLoader(
        WindowDataset(x_train, y_train, station_ids_train),
        batch_size=int(ft_batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(ft_num_workers),
    )
    val_loader = DataLoader(
        WindowDataset(x_val, y_val, station_ids_val),
        batch_size=int(ft_batch_size),
        shuffle=False,
        num_workers=int(ft_num_workers),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(ft_lr),
        weight_decay=float(ft_weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(ft_max_epochs), eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))

    for epoch in range(1, int(ft_max_epochs) + 1):
        model.train()
        epoch_losses = []
        epoch_start = time.time()

        for batch in train_loader:
            if len(batch) == 3:
                context, target, _station_id = batch
            else:
                context, target = batch
            context = context.to(device)
            target = target.to(device)
            target_mask = (target > float(train_ghi_mask_wm2)).to(device)
            if not target_mask.any():
                continue
            optimizer.zero_grad()
            out = model(context=context, target=target, target_mask=target_mask)
            loss = out.loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), float(ft_gradient_clip))
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        val_loss, val_rmse = _evaluate_rmse(model, val_loader, device)
        elapsed = time.time() - epoch_start

        if int(ft_log_every) <= 1 or epoch % int(ft_log_every) == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{int(ft_max_epochs)} | train_loss={train_loss:.5f}  "
                f"val_loss={val_loss:.5f}  val_rmse={val_rmse:.2f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed/60:.1f}m"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(ft_patience):
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    adapter_name = "chronos_bolt_lora_adapter_smoke" if smoke_test else "chronos_bolt_lora_adapter"
    adapter_path = checkpoint_dir / adapter_name
    model.save_pretrained(str(adapter_path))

    config_path = checkpoint_dir / ("lora_config_smoke.json" if smoke_test else "lora_config.json")
    payload = {
        "base_model": model_id,
        "model_type": "chronos-bolt",
        "best_val_loss": float(best_val_loss),
        "best_val_rmse": float(best_val_rmse),
        "lora_rank": int(lora_rank),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "target_modules": list(lora_target_modules),
        "smoke_test": bool(smoke_test),
        "train_ghi_mask_wm2": float(train_ghi_mask_wm2),
        "station_balanced_sampling": bool(station_balanced_sampling),
    }
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n  LoRA adapter saved -> {adapter_path}")
    print(f"  Config saved -> {config_path}")
    print(f"  Best val RMSE: {best_val_rmse:.2f} W/m²")
    return adapter_path, payload
