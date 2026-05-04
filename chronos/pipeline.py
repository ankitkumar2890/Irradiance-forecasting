"""Standalone Chronos Method 3 pipeline: dataset -> finetune -> infer -> evaluate."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from functions.model import finetune_chronos_bolt, select_device
from functions.preprocess import build_dataset_method3
from functions.results import build_report_method3, run_inference_method3


FINAL_CSV_DIR = HERE / "final_csv"
DATASET_ROOT = HERE / "dataset"
CHECKPOINT_ROOT = HERE / "checkpoints"
RESULTS_ROOT = HERE / "results"


def _load_config_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def get_method_config():
    return _load_config_module("cfg_method3", HERE / "configs" / "method3.py")


def method_paths(model_slug: str):
    paths = {
        "final_csv_dir": FINAL_CSV_DIR / "method3",
        "dataset_dir": DATASET_ROOT / "method3",
        "checkpoint_dir": CHECKPOINT_ROOT / "method3" / model_slug,
        "results_dir": RESULTS_ROOT / "method3" / model_slug,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def model_slug_from_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def cmd_dataset(cfg, paths):
    print("\n=== CHRONOS METHOD 3: dataset ===\n")
    return build_dataset_method3(
        final_csv_dir=paths["final_csv_dir"],
        dataset_dir=paths["dataset_dir"],
        train_end=cfg.TRAIN_END,
        val_start=cfg.VAL_START,
        val_end=cfg.VAL_END,
        test_start=cfg.TEST_START,
        target_col=cfg.TARGET_COL,
        station_col=cfg.STATION_COL,
        past_hours=cfg.PAST_HOURS,
        future_hours=cfg.FUTURE_HOURS,
        anchor_hours=getattr(cfg, "ANCHOR_HOURS", None),
        min_past_dates=int(getattr(cfg, "MIN_PAST_DATES", 1)),
    )


def cmd_finetune(args, cfg, paths):
    print("\n=== CHRONOS METHOD 3: finetune ===\n")
    device = select_device(args.device, args.smoke_test)
    return finetune_chronos_bolt(
        dataset_dir=paths["dataset_dir"],
        checkpoint_dir=paths["checkpoint_dir"],
        model_id=args.model_id,
        device=device,
        smoke_test=args.smoke_test,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        lora_rank=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        lora_target_modules=cfg.LORA_TARGET_MODULES,
        ft_batch_size=cfg.FT_BATCH_SIZE,
        ft_lr=cfg.FT_LR,
        ft_weight_decay=cfg.FT_WEIGHT_DECAY,
        ft_max_epochs=cfg.FT_MAX_EPOCHS,
        ft_patience=cfg.FT_PATIENCE,
        ft_gradient_clip=cfg.FT_GRADIENT_CLIP,
        ft_num_workers=cfg.FT_NUM_WORKERS,
        ft_log_every=cfg.FT_LOG_EVERY,
        seed=cfg.FT_SEED,
        train_ghi_mask_wm2=cfg.TRAIN_GHI_MASK_WM2,
        station_balanced_sampling=bool(getattr(cfg, "STATION_BALANCED_SAMPLING", False)),
    )


def cmd_infer(args, cfg, paths):
    print("\n=== CHRONOS METHOD 3: inference ===\n")
    device = select_device(args.device, args.smoke_test)
    return run_inference_method3(
        cfg=cfg,
        dataset_dir=paths["dataset_dir"],
        checkpoint_dir=paths["checkpoint_dir"],
        results_dir=paths["results_dir"],
        final_csv_dir=paths["final_csv_dir"],
        model_id=args.model_id,
        device=device,
        smoke_test=args.smoke_test,
    )


def cmd_evaluate(args, cfg, paths):
    print("\n=== CHRONOS METHOD 3: evaluation ===\n")
    return build_report_method3(
        cfg=cfg,
        final_csv_dir=paths["final_csv_dir"],
        results_dir=paths["results_dir"],
        validation_ghi_filter_wm2=args.ghi_filter,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["dataset", "finetune", "infer", "evaluate", "all"])
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--ghi-filter", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_method_config()
    if not args.model_id:
        args.model_id = cfg.MODEL_ID
    if args.ghi_filter is None:
        args.ghi_filter = float(getattr(cfg, "DEFAULT_GHI_FILTER_WM2", 20.0))

    paths = method_paths(model_slug_from_id(args.model_id))

    if args.command == "dataset":
        cmd_dataset(cfg, paths)
    elif args.command == "finetune":
        cmd_finetune(args, cfg, paths)
    elif args.command == "infer":
        cmd_infer(args, cfg, paths)
    elif args.command == "evaluate":
        cmd_evaluate(args, cfg, paths)
    elif args.command == "all":
        cmd_dataset(cfg, paths)
        cmd_finetune(args, cfg, paths)
        cmd_infer(args, cfg, paths)
        cmd_evaluate(args, cfg, paths)


if __name__ == "__main__":
    main()
