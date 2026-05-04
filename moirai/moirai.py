"""
moirai.py - main entry point for the Moirai fine-tuning pipeline.

Three methods live side-by-side. Pick the method with ``--method`` and
optionally the model backbone with ``--variant``::

    # End-to-end (dataset -> finetune -> infer -> evaluate -> PDF):
    python moirai/moirai.py --method 1 all

    # Or run individual steps:
    python moirai/moirai.py --method 1 dataset
    python moirai/moirai.py --method 1 finetune [--smoke-test] [--device cuda]
    python moirai/moirai.py --method 1 infer    [--smoke-test]
    python moirai/moirai.py --method 1 evaluate [--ghi-filter 20]

    # Force the Moirai 2.0 backbone for any step (default is auto from cfg.MODEL_ID):
    python moirai/moirai.py --method 1 --variant moirai2 finetune

Layout (everything lives under ``moirai/`` -- the pipeline is fully
self-contained and does not depend on the sibling ``phase*/`` folders
for any runtime constant)::

    moirai/
    +-- configs/method{1,2,3}.py   <-- self-contained per-method configs
    +-- final_csv/method{1,2,3}/   <-- drop the prepared CSV for each method
    +-- dataset/method{1,2,3}/     <-- windowed .npy outputs
    +-- checkpoints/method{1,2,3}/ <-- LoRA adapters land here
    +-- results/method{1,2,3}/     <-- inference + evaluation artefacts
    +-- master_files/              <-- shared metrics/plots/PDF helpers
    +-- functions/preprocess.py    <-- CSV -> windowed dataset
    +-- functions/model.py         <-- Moirai 1.x + 2.0 + LoRA fine-tune
    +-- functions/results.py       <-- inference + metrics + plots + PDF
    +-- moirai.py

Methods::

    1 = CAF target, PVLib clear-sky, single station, ERA5 covariates
        -> configs/method1.py       (default backbone: Moirai 1.1-R-base)
    2 = CAF target, NSRDB clear-sky, multi-station; CAF -> GHI at eval
        -> configs/method2.py       (default backbone: Moirai 2.0-R-small)
    3 = Direct GHI target, multi-station, neighbor context, GHI scaling
        -> configs/method3.py       (default backbone: Moirai 2.0-R-small)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# ---- Path setup ---------------------------------------------------------
# moirai/ is fully self-contained: configs live in moirai/configs/, helpers
# in moirai/master_files/, and code in moirai/functions/. Only ``HERE``
# needs to be on sys.path so that ``from functions...`` and
# ``from configs...`` resolve.
HERE = Path(__file__).resolve().parent           # moirai/

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from functions.preprocess import (
    build_dataset_method1,
    build_dataset_method2,
    build_dataset_method3,
)
from functions.model import (
    finetune_method1,
    finetune_method2,
    finetune_method3,
)
from functions.results import (
    build_report_method1,
    build_report_method2,
    build_report_method3,
    run_inference_method1,
    run_inference_method2,
    run_inference_method3,
)

# ---- Project-level constants -------------------------------------------
FINAL_CSV_DIR  = HERE / "final_csv"
DATASET_ROOT   = HERE / "dataset"
CHECKPOINT_ROOT = HERE / "checkpoints"
RESULTS_ROOT   = HERE / "results"


# =====================================================================
# Per-method paths
# =====================================================================

def method_paths(method: int, variant: str | None = None) -> dict:
    """Resolve per-method input/output paths under moirai/.

    The dataset and the checkpoint directory are **shared** across variants
    (the LoRA adapter folders inside checkpoints/method{N}/ are already
    variant-aware, e.g. ``moirai1_lora_adapter/`` vs
    ``moirai2_lora_adapter/``), but the results directory is **variant-aware**
    so ``--variant moirai1`` and ``--variant moirai2`` write into separate
    folders and never overwrite each other::

        moirai/results/method1/moirai1/...
        moirai/results/method1/moirai2/...

    ``cmd_dataset`` does not need a variant — pass ``variant=None`` and the
    function still builds the dataset/checkpoint folders correctly.
    """
    sub = f"method{method}"
    results_dir = (
        RESULTS_ROOT / sub / variant if variant else RESULTS_ROOT / sub
    )
    paths = {
        "final_csv_dir":  FINAL_CSV_DIR / sub,
        "dataset_dir":    DATASET_ROOT / sub,
        "checkpoint_dir": CHECKPOINT_ROOT / sub,
        "results_dir":    results_dir,
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# =====================================================================
# Per-method configuration loader
# =====================================================================

def _load_config_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def resolve_variant(cfg, override: str = "auto") -> str:
    """Resolve which Moirai variant to use for this run."""
    if override and override != "auto":
        return override
    explicit = getattr(cfg, "MOIRAI_VARIANT", None)
    if explicit:
        return str(explicit)
    model_id = str(getattr(cfg, "MODEL_ID", "")).lower()
    if "moirai-2" in model_id or "moirai2" in model_id:
        return "moirai2"
    return "moirai1"


def resolve_model_id(cfg, variant: str) -> str:
    """Pick the right HF checkpoint id for ``variant``.

    Priority::

        1. cfg.MODEL_ID_MOIRAI1 / cfg.MODEL_ID_MOIRAI2 (per-variant overrides)
        2. cfg.MODEL_ID if its variant matches the resolved one
        3. cfg.MODEL_ID as a fallback (with a warning)

    This is what makes ``--variant moirai2`` actually work for Method 2:
    ``cfg.MODEL_ID = "Salesforce/moirai-1.1-R-large"`` would otherwise be
    sent to the Moirai 2 loader and crash. With ``MODEL_ID_MOIRAI2``
    defined in the cfg, the right checkpoint is loaded instead.
    """
    explicit = getattr(cfg, f"MODEL_ID_{variant.upper()}", None)
    if explicit:
        return str(explicit)
    cfg_model_id = str(getattr(cfg, "MODEL_ID", ""))
    cfg_model_id_lc = cfg_model_id.lower()
    cfg_is_moirai2 = "moirai-2" in cfg_model_id_lc or "moirai2" in cfg_model_id_lc
    if (variant == "moirai2" and cfg_is_moirai2) or (
        variant == "moirai1" and not cfg_is_moirai2
    ):
        return cfg_model_id
    print(
        f"  Warning: variant={variant!r} but cfg.MODEL_ID={cfg_model_id!r} "
        f"does not match. Define MODEL_ID_{variant.upper()} in the method "
        "config to silence this. Falling back to cfg.MODEL_ID."
    )
    return cfg_model_id


# Self-contained configs live next to this file under moirai/configs/.
# Each method module owns only the constants the pipeline actually reads
# (model id, window/feature lists, LoRA + FT hyperparameters, eval-time
# column hints). No API keys, no station-grid math, no path helpers - all
# of those still live in the matching ``phase*/config.py`` for the data
# preparation step that produces ``final_csv/method{N}/`` CSVs.
_METHOD_CONFIG_PATHS = {
    1: HERE / "configs" / "method1.py",
    2: HERE / "configs" / "method2.py",
    3: HERE / "configs" / "method3.py",
}


def get_method_config(method: int):
    """Return the config module that defines hyperparameters for ``method``."""
    if method not in _METHOD_CONFIG_PATHS:
        raise ValueError(f"Unknown method: {method}")
    cfg_path = _METHOD_CONFIG_PATHS[method]
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Method {method} config file not found at {cfg_path}. "
            "moirai/configs/ should ship with this repo - if it is missing, "
            "restore the file or copy values from the matching phase folder."
        )
    return _load_config_module(f"cfg_method{method}", cfg_path)


# =====================================================================
# Method dispatch tables
# =====================================================================

DATASET_BUILDERS = {
    1: build_dataset_method1,
    2: build_dataset_method2,
    3: build_dataset_method3,
}
FINETUNERS = {
    1: finetune_method1,
    2: finetune_method2,
    3: finetune_method3,
}
INFERERS = {
    1: run_inference_method1,
    2: run_inference_method2,
    3: run_inference_method3,
}
EVALUATORS = {
    1: build_report_method1,
    2: build_report_method2,
    3: build_report_method3,
}


# =====================================================================
# Subcommand handlers
# =====================================================================

def _common_dataset_kwargs(cfg, paths) -> dict:
    """Fields shared by all three method dataset builders."""
    return dict(
        final_csv_dir=paths["final_csv_dir"],
        dataset_dir=paths["dataset_dir"],
        train_end=cfg.TRAIN_END,
        val_start=cfg.VAL_START,
        val_end=cfg.VAL_END,
        test_start=cfg.TEST_START,
        past_hours=cfg.PAST_HOURS,
        future_hours=cfg.FUTURE_HOURS,
        past_features=cfg.PAST_FEATURES,
        future_features=cfg.FUTURE_FEATURES,
        anchor_hours=getattr(cfg, "ANCHOR_HOURS", None),
        min_past_dates=int(getattr(cfg, "MIN_PAST_DATES", 1)),
    )


def cmd_dataset(args):
    method = args.method
    paths = method_paths(method)
    cfg = get_method_config(method)
    kwargs = _common_dataset_kwargs(cfg, paths)

    if method == 1:
        DATASET_BUILDERS[method](
            **kwargs,
            item_id=cfg.FINETUNE_STATION,
            target_col="CAF",
        )
    else:
        # Methods 2 and 3 also need the clear-sky / measured-GHI column names so
        # the schema check in the builder matches the user's CSV. We source these
        # from the method's config.py rather than relying on the kwarg defaults
        # (which assume the NSRDB ``clearsky_ghi`` convention) so users can match
        # whatever convention their CSV uses (e.g. PVLib's ``clear_sky_ghi``).
        DATASET_BUILDERS[method](
            **kwargs,
            item_id=getattr(cfg, "FINETUNE_STATION", "method"),
            target_col=getattr(cfg, "TARGET_COL", "GHI" if method == 3 else "CAF"),
            clearsky_col=getattr(cfg, "CLEARSKY_GHI_COL", "clearsky_ghi"),
            measured_col=getattr(cfg, "MEASURED_GHI_COL", "w_ghr"),
        )


def cmd_finetune(args):
    method = args.method
    cfg = get_method_config(method)
    variant = resolve_variant(cfg, getattr(args, "variant", "auto"))
    paths = method_paths(method, variant=variant)
    model_id = resolve_model_id(cfg, variant)
    print(f"  Method {method} | Moirai variant: {variant} | model_id: {model_id}")
    print(f"  Results directory (variant-scoped): {paths['results_dir']}")

    finetune_kwargs = dict(
        dataset_dir=paths["dataset_dir"],
        checkpoint_dir=paths["checkpoint_dir"],
        model_id=model_id,
        context_length=cfg.CONTEXT_LENGTH,
        prediction_length=cfg.PREDICTION_LENGTH,
        target_dim=cfg.TARGET_DIM,
        feat_dim=cfg.FEAT_DIM,
        lora_rank=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_target_modules=cfg.LORA_TARGET_MODULES,
        lora_dropout=cfg.LORA_DROPOUT,
        ft_lr=cfg.FT_LR,
        ft_weight_decay=cfg.FT_WEIGHT_DECAY,
        ft_max_epochs=cfg.FT_MAX_EPOCHS,
        ft_patience=cfg.FT_PATIENCE,
        ft_batch_size=cfg.FT_BATCH_SIZE,
        ft_gradient_clip=cfg.FT_GRADIENT_CLIP,
        smoke_test=getattr(args, "smoke_test", False),
        max_train_windows=getattr(args, "max_train_windows", None),
        max_val_windows=getattr(args, "max_val_windows", None),
        device_arg=getattr(args, "device", "auto"),
        model_variant=variant,
        # Pass column-name lists so that npy_to_gluonts can align
        # feat_dynamic_real correctly if a method ever carries past-only
        # extras. For Methods 1/2 and Method 3's current direct-GHI
        # layout, the alignment is a no-op.
        past_features=list(cfg.PAST_FEATURES),
        future_features=list(cfg.FUTURE_FEATURES),
    )
    if method == 3:
        finetune_kwargs["target_scale"] = float(getattr(cfg, "GHI_SCALE_FACTOR", 1000.0))

    FINETUNERS[method](**finetune_kwargs)


def cmd_infer(args):
    method = args.method
    cfg = get_method_config(method)
    variant = resolve_variant(cfg, getattr(args, "variant", "auto"))
    paths = method_paths(method, variant=variant)
    model_id = resolve_model_id(cfg, variant)
    print(f"  Method {method} | Moirai variant: {variant} | model_id: {model_id}")
    print(f"  Results directory (variant-scoped): {paths['results_dir']}")

    INFERERS[method](
        cfg=cfg,
        dataset_dir=paths["dataset_dir"],
        checkpoint_dir=paths["checkpoint_dir"],
        results_dir=paths["results_dir"],
        model_id=model_id,
        model_variant=variant,
        smoke_test=getattr(args, "smoke_test", False),
    )


def cmd_evaluate(args):
    method = args.method
    cfg = get_method_config(method)
    variant = resolve_variant(cfg, getattr(args, "variant", "auto"))
    paths = method_paths(method, variant=variant)
    print(f"  Method {method} | Moirai variant: {variant}")
    print(f"  Results directory (variant-scoped): {paths['results_dir']}")

    EVALUATORS[method](
        cfg=cfg,
        final_csv_dir=paths["final_csv_dir"],
        results_dir=paths["results_dir"],
        validation_ghi_filter_wm2=getattr(args, "ghi_filter", 20.0),
    )


def cmd_all(args):
    """End-to-end: dataset -> finetune -> infer -> evaluate."""
    cmd_dataset(args)
    cmd_finetune(args)
    cmd_infer(args)
    cmd_evaluate(args)


# =====================================================================
# CLI
# =====================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Moirai fine-tuning pipeline orchestrator (3 methods).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--method", type=int, choices=[1, 2, 3], default=1,
        help=(
            "Which method to run. "
            "1 = CAF + PVLib + ERA5 (single station, phase2_era5_direct, default). "
            "2 = CAF + NSRDB + multi-station (phase2_finetuning). "
            "3 = Direct GHI + multi-station (phase3_direct_ghi)."
        ),
    )
    parser.add_argument(
        "--variant", choices=["auto", "moirai1", "moirai2"], default="auto",
        help=(
            "Which Moirai backbone to use. 'auto' picks based on cfg.MOIRAI_VARIANT "
            "or by inspecting cfg.MODEL_ID. With --variant moirai2 the loader will "
            "use cfg.MODEL_ID_MOIRAI2 if defined (otherwise cfg.MODEL_ID), and vice "
            "versa for moirai1."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "dataset",
        help="Read final_csv/method{N}/*.csv -> windowed dataset (.npy).",
    )

    p_ft = sub.add_parser(
        "finetune",
        help="LoRA fine-tune Moirai; saves to checkpoints/method{N}/.",
    )
    p_ft.add_argument(
        "--smoke-test", action="store_true",
        help="Run one tiny epoch on a few windows to verify the pipeline.",
    )
    p_ft.add_argument("--max-train-windows", type=int, default=None)
    p_ft.add_argument("--max-val-windows", type=int, default=None)
    p_ft.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
        help="Compute device. 'auto' prefers cuda, then cpu.",
    )

    p_all = sub.add_parser(
        "all",
        help=(
            "End-to-end: dataset -> finetune -> infer -> evaluate. "
            "Produces checkpoint, predictions, metrics, plots and PDF in one go."
        ),
    )
    p_all.add_argument(
        "--smoke-test", action="store_true",
        help="Run one tiny epoch + use the *_smoke adapter for inference.",
    )
    p_all.add_argument("--max-train-windows", type=int, default=None)
    p_all.add_argument("--max-val-windows", type=int, default=None)
    p_all.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
        help="Compute device for fine-tune. 'auto' prefers cuda, then cpu.",
    )
    p_all.add_argument(
        "--ghi-filter", type=float, default=20.0,
        help="Drop rows with measured GHI <= this many W/m^2 during evaluate (default: 20).",
    )

    p_infer = sub.add_parser(
        "infer", help="Run fine-tuned inference on the validation split."
    )
    p_infer.add_argument(
        "--smoke-test", action="store_true",
        help="Use the *_smoke LoRA adapter saved during a smoke-test fine-tune.",
    )

    p_eval = sub.add_parser(
        "evaluate",
        help="Build metrics + plots + PDF from finetuned_predictions.csv.",
    )
    p_eval.add_argument(
        "--ghi-filter", type=float, default=20.0,
        help="Drop rows with measured GHI <= this many W/m^2 from metrics/plots (default: 20).",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "dataset": cmd_dataset,
        "finetune": cmd_finetune,
        "infer": cmd_infer,
        "evaluate": cmd_evaluate,
        "all": cmd_all,
    }
    handlers[args.cmd](args)


if __name__ == "__main__":
    main()
