# Moirai Fine-Tuning Pipeline

Self-contained LoRA fine-tuning pipeline for Salesforce's **Moirai** time-series
foundation model, applied to GHI (Global Horizontal Irradiance) forecasting.

Three methods live side by side. Pick which one with `--method` and which
Moirai backbone with `--variant`. One command (`all`) does dataset → fine-tune
→ infer → evaluate → PDF in a single shot. Every method emits **p10 / p50 / p90**
prediction intervals, and Moirai 1.x and Moirai 2.0 outputs are written to
**separate result folders** so they never overwrite each other.

---

## Quick start

```bash
# End-to-end (everything: dataset, checkpoint, predictions, metrics, plots, PDF):
python moirai/moirai.py --method 1 all

# Same on CUDA:
python moirai/moirai.py --method 1 all --device cuda

# Smoke test (1 epoch, 8 train / 4 val windows; uses *_smoke adapter):
python moirai/moirai.py --method 1 all --smoke-test

# Force the Moirai 2.0 backbone instead of the cfg default:
# (writes into results/method1/moirai2/, not results/method1/moirai1/)
python moirai/moirai.py --method 1 --variant moirai2 all
```

---

## Command reference

### Top-level flags (apply to every sub-command)

| Flag | Default | What it does |
|---|---|---|
| `--method {1,2,3}` | `1` | Which method to run. See **Methods** below. |
| `--variant {auto,moirai1,moirai2}` | `auto` | Moirai backbone. `auto` reads `cfg.MOIRAI_VARIANT` if set, otherwise infers from `cfg.MODEL_ID` (anything containing `moirai-2` → `moirai2`, else `moirai1`). The CLI uses `cfg.MODEL_ID_MOIRAI1` / `cfg.MODEL_ID_MOIRAI2` (see **Moirai 1.x vs 2.0**) so the requested variant always loads the matching checkpoint. |

### Sub-commands

| Command | What it does | Flags |
|---|---|---|
| `dataset` | Reads the prepared CSV from `final_csv/method{N}/` and writes windowed `.npy` files to `dataset/method{N}/`. Prints a full audit of how many candidate windows were considered, kept, and dropped (with reasons) — nothing is silently discarded. | — |
| `finetune` | LoRA fine-tunes Moirai (1.x or 2.0). Writes the adapter to `checkpoints/method{N}/moirai{1,2}_lora_adapter/` and a `lora_config_moirai{1,2}.json` with `target_scale` recorded for the matching inference path. | `--smoke-test` `--device {auto,cpu,cuda,mps}` `--max-train-windows N` `--max-val-windows N` |
| `infer` | Loads the saved LoRA adapter and writes `finetuned_predictions.csv` (with `*_pred`, `*_pred_p10`, `*_pred_p90`) + `finetuned_metrics.json` to `results/method{N}/{variant}/`. | `--smoke-test` (uses the `*_smoke` adapter) |
| `evaluate` | Reads the predictions, joins them back to the prepared CSV (clear-sky, measured GHI, zenith), computes both custom and master metrics, renders all plots via `master_files/master_plots.py`, and assembles `finetuned_validation_report.pdf` via `master_files/export_pdf.build_finetuned_pdf` — the PDF includes p10/p90 prediction-interval pages for every method. | `--ghi-filter VAL` (drop rows with measured GHI ≤ VAL W/m²; default 20) |
| `all` | Runs `dataset` → `finetune` → `infer` → `evaluate` end-to-end. The single command that turns a prepared CSV into a checkpoint **and** a finished PDF report. | `--smoke-test` `--device {auto,cpu,cuda,mps}` `--max-train-windows N` `--max-val-windows N` `--ghi-filter VAL` |

---

## Methods

| Method | Config | Data-prep folder (CSV source) | Target | Clear-sky source | Stations | Default backbone |
|---|---|---|---|---|---|---|
| **1** | `configs/method1.py` | `phase2_era5_direct/` | CAF (= GHI / clear-sky GHI) — converted back to GHI at evaluation | PVLib (Ineichen model) | Single station (`cfg.FINETUNE_STATION`) | Moirai **1.1-R-base** |
| **2** | `configs/method2.py` | `phase2_finetuning/` | CAF — converted back to GHI at evaluation | NSRDB (`clearsky_ghi` column) | Multi-station (3×3 grid or 5-station explicit cluster) | Moirai **2.0-R-small** (use `--variant moirai1` for the Moirai 1.1-R-large path) |
| **3** | `configs/method3.py` | `phase3_direct_ghi/` | GHI directly (W/m²); no CAF intermediate | NSRDB | Multi-station + neighbor-context features | Moirai **2.0-R-small** |

Method-specific behaviour handled automatically:

* **Method 3** scales GHI by `cfg.GHI_SCALE_FACTOR` (default 1000) before training
  and un-scales at inference. The scale used at training time is stored in
  `lora_config_moirai{1,2}.json` (`target_scale`) and read back by inference,
  so the train/infer scaling can never drift apart.
* **Method 3** applies a daylight-zenith mask
  (`cfg.DAYLIGHT_ZENITH_DEG` / `cfg.NIGHT_ZENITH_DEG`) so night-time
  predictions are tapered to zero.
* **Every method** emits p10 / p50 / p90 prediction intervals:
  * Moirai 1.x derives them from the predicted distribution
    (`distr.icdf(q)` if available, MC sampling otherwise).
  * Moirai 2.0 reads them off the model's quantile head.
  The PDF report renders prediction-interval pages for the first week and a
  random later week of the validation period.
* **Methods 2 & 3** emit per-station metrics (the multi-station preprocess
  saves `station_ids_{train,val,test}.npy` alongside the X/y windows; the
  evaluator surfaces a per-station table in the PDF).

---

## Status

| Stage | Method 1 | Method 2 | Method 3 |
|---|---|---|---|
| `preprocess.build_dataset_method{N}` | ✅ done | ✅ done (multi-station, target = CAF) | ✅ done (multi-station, target = GHI; contract-checked at build time) |
| `model.finetune_method{N}` | ✅ done | ✅ done (reuses `_finetune_generic`) | ✅ done (`_finetune_generic` with `target_scale=cfg.GHI_SCALE_FACTOR`) |
| `results.run_inference_method{N}` | ✅ done (p10/p50/p90 from Moirai 1.x distr) | ✅ done (per-station, p10/p50/p90) | ✅ done (scaling, daylight mask, p10/p50/p90 from Moirai 2.0 quantile head) |
| `results.build_report_method{N}` | ✅ done | ✅ done (per-station + aggregate metrics, p10/p90 PDF pages) | ✅ done (per-station, p10/p90 PDF pages) |

Methods 2 and 3 will run end-to-end as soon as you drop a prepared CSV with
the columns documented in `functions/preprocess.py::build_dataset_method{2,3}`.

---

## Layout

```
moirai/
├── moirai.py                   # main CLI entrypoint (run everything from here)
├── configs/
│   ├── __init__.py
│   ├── method1.py              # CAF + PVLib + ERA5 (single station)
│   ├── method2.py              # CAF + NSRDB + multi-station (3x3 grid)
│   └── method3.py              # Direct GHI + multi-station (5-station cluster)
├── functions/
│   ├── __init__.py
│   ├── preprocess.py           # CSV -> windowed .npy dataset (per-method sections)
│   ├── model.py                # Moirai 1.x + 2.0 loading + LoRA fine-tune loop
│   └── results.py              # inference + metrics + plots + PDF report
├── master_files/
│   ├── master_metrics.py       # calculate_metrics, print_metrics
│   ├── master_plots.py         # plot_time_series, plot_two_week_comparison, plot_4panel_evaluation
│   └── export_pdf.py           # build_finetuned_pdf (assembles the per-method PDF)
├── final_csv/
│   ├── method1/                # drop method-1 starting CSV here
│   ├── method2/                # drop method-2 starting CSV here
│   └── method3/                # drop method-3 starting CSV here
├── dataset/method{1,2,3}/      # auto-generated .npy windows (variant-shared)
├── checkpoints/method{1,2,3}/  # LoRA adapters land here (variant-shared folder
│                               # with moirai1_lora_adapter/ + moirai2_lora_adapter/)
└── results/method{N}/{variant}/  # variant-scoped: results/method1/moirai1/...
                                  #                results/method1/moirai2/...
```

---

## Pre-flight checklist

1. Activate the project venv:
   ```bash
   source /Users/IRFAN/Desktop/moirai_finetuning/venv311/bin/activate
   ```
2. Drop the prepared CSV for the method you want to run into the matching
   `final_csv/method{N}/` folder.
   * **Method 1** (single-station): needs `datetime`, `CAF` (or `w_ghr` + `clear_sky_ghi`),
     ERA5 covariates, `zenith_angle`. The dataset builder also requires
     `w_ghr` to be present (used at evaluation time to reconstruct GHI from
     CAF) — this is checked up-front so you don't lose a 30-min training run
     to a missing column.
   * **Method 2** (multi-station, NSRDB clear-sky): needs `datetime`,
     `station_id`, `w_ghr`, `clearsky_ghi`, ERA5 cloud cover (`tcc/lcc/mcc/hcc`),
     `u10/v10` (or `wind_speed` + `wind_direction`), `zenith_angle`,
     `azimuth_angle`, `elevation_m`.
   * **Method 3** (direct GHI, multi-station): needs `datetime`, `station_id`,
     `w_ghr` (raw GHI), `clearsky_ghi`, `zenith_angle`, `azimuth_angle`, all the
     `*_delta_1h` cloud/weather signals, `temperature_2m`, RH, dew point,
     `surface_pressure` (+ deltas), `wind_speed`, `wind_direction`,
     `wind_gusts_10m`, `wind_speed_delta_1h`, `neighbor_*_mean` columns,
     `elevation_m`, sin/cos calendar columns. The builder validates the cfg
     contract (`PAST_FEATURES[0] == w_ghr`, every `FUTURE_FEATURE` also in
     `PAST_FEATURES`, `zenith_angle` in `FUTURE_FEATURES`) up-front.
3. The right `config.py` is already wired up — `--method 1` loads
   `moirai/configs/method1.py`, `--method 2` loads
   `moirai/configs/method2.py`, `--method 3` loads
   `moirai/configs/method3.py`. These three files are **self-contained**:
   they hold only the constants the pipeline actually reads (model ids,
   features, window/LoRA/FT hyperparameters, eval-time column hints).
   The original `phase*/config.py` files still exist for the data
   preparation step that produces the prepared CSV in `final_csv/method{N}/`,
   but `moirai/` no longer imports anything from them.

---

## Window selection (`ANCHOR_HOURS`)

`build_windows` keeps a window only if its **first forecast hour** matches one
of the entries in `cfg.ANCHOR_HOURS`. Defaults to `[6]` (one forecast issued at
06:00 local time per day) for backward compatibility, but you can crank that
up:

```python
# phase2_era5_direct/config.py
ANCHOR_HOURS = [6]            # 1 forecast/day at 06:00 (legacy)
ANCHOR_HOURS = [0, 6, 12, 18] # 4 forecasts/day
ANCHOR_HOURS = None           # 1 window per timestamp (no anchor filter)
```

The dataset step always prints a full audit so you know exactly what the
window selection produced:

```
train: kept 726/17449 candidate windows  (hours [6])
    dropped 16723 window(s) because the first forecast hour was not in the allowed anchor set.
```

---

## Training-loop semantics

* **Mini-batches via gradient accumulation.** `train_lora` accumulates
  `cfg.FT_BATCH_SIZE` per-window losses (each scaled by `1/FT_BATCH_SIZE`)
  before stepping the optimizer. This is what makes the configured
  `FT_BATCH_SIZE = 32` actually behave like a 32-window batch instead of
  per-window SGD.
* **Per-epoch shuffle.** Train pairs `(item, y)` are stored as a list of
  tuples (no separate `y_train` array indexed by `batch_idx`) and shuffled
  with a deterministic seed every epoch — the y always travels with its
  matching item, so alignment cannot drift.
* **Per-epoch logging.** The fine-tune loop prints train/val MSE every
  epoch (e.g. `Epoch  17/50 | train_mse=0.00298  val_mse=0.01238  lr=7.96e-05`),
  not every 5.
* **Smoke test.** `--smoke-test` runs 1 epoch with 8 train / 4 val windows
  on CPU and saves to a `*_smoke` adapter folder so it cannot overwrite a
  full-quality run.

---

## Outputs

After `python moirai/moirai.py --method 1 all` finishes:

```
moirai/
├── dataset/method1/
│   ├── X_past_train.npy / X_future_train.npy / y_future_train.npy / times_train.npy
│   ├── X_past_val.npy   / X_future_val.npy   / y_future_val.npy   / times_val.npy
│   └── X_past_test.npy  / X_future_test.npy  / y_future_test.npy  / times_test.npy
│
├── checkpoints/method1/
│   ├── moirai1_lora_adapter/         <- LoRA weights (PEFT format)
│   │   ├── adapter_model.bin
│   │   ├── adapter_config.json
│   │   └── ...
│   └── lora_config_moirai1.json      <- summary: lr, rank, best val MSE/RMSE, target_scale
│
└── results/method1/moirai1/
    ├── finetuned_predictions.csv     <- datetime, lead_time_h, forecast_start, CAF_true,
    │                                   CAF_pred, CAF_pred_p10, CAF_pred_p90 (+ station_id
    │                                   for Method 2/3)
    ├── finetuned_metrics.json        <- top-line CAF RMSE / MAE + target_scale + variant
    ├── evaluation_metrics.json       <- detailed CAF + GHI metrics + horizon RMSE +
    │                                   per-station metrics (Methods 2/3)
    ├── evaluation_report.txt         <- text summary
    ├── validation_report_data.csv    <- merged predictions + truth + clear-sky +
    │                                   GHI_pred_p10 / GHI_pred_p90
    ├── finetuned_timeseries.png
    ├── finetuned_two_week_comparison.png
    ├── finetuned_4panel.png
    ├── finetuned_horizon_rmse.png
    └── finetuned_validation_report.pdf  <- includes p10/p90 prediction-interval pages
```

`*_smoke` variants of the adapter and config files appear when you use
`--smoke-test`.

A separate `--variant moirai2` run lands in `results/method1/moirai2/` and
the moirai1 outputs are untouched — you can compare both runs side by side.

---

## Predictions CSV columns

| Column | Methods | Notes |
|---|---|---|
| `datetime` | 1, 2, 3 | Forecast valid time (i.e. the hour being forecast). |
| `hour` | 1, 2, 3 | `datetime.dt.hour`, kept for the 4-panel plot's hour-of-day axis. |
| `lead_time_h` | 1, 2, 3 | 1 .. `PREDICTION_LENGTH`. |
| `forecast_start` | 1, 2, 3 | **Issue time** = last context hour (= first forecasted hour − 1h). Same convention for every method. |
| `CAF_true`, `CAF_pred`, `CAF_pred_p10`, `CAF_pred_p90` | 1, 2 | CAF target column. |
| `GHI_true`, `GHI_pred`, `GHI_pred_p10`, `GHI_pred_p90` | 3 | GHI target column (W/m²). For Methods 1/2 these are added later in `_build_report_generic` via `GHI_pred = CAF_pred * clear_sky_ghi`. |
| `station_id` | 2, 3 | Multi-station only. |

---

## Moirai 1.x vs 2.0

`moirai.py::resolve_variant(cfg, override)` picks the variant in this order:

1. **CLI override** (`--variant moirai1` / `--variant moirai2`).
2. **`cfg.MOIRAI_VARIANT`** if defined in the method's config module.
3. **Inferred from `cfg.MODEL_ID`** — anything containing `moirai-2` or
   `moirai2` resolves to `"moirai2"`, otherwise `"moirai1"`.

`moirai.py::resolve_model_id(cfg, variant)` then picks the actual checkpoint.
This is what makes `--variant moirai2` work for Method 2 (whose default
`cfg.MODEL_ID` is a Moirai 1.x checkpoint):

1. `cfg.MODEL_ID_MOIRAI1` / `cfg.MODEL_ID_MOIRAI2` if defined (preferred).
2. `cfg.MODEL_ID` if its variant matches the resolved one.
3. `cfg.MODEL_ID` as a last-resort fallback (with a warning printed).

Out-of-the-box that means:

| Method | `cfg.MODEL_ID_MOIRAI1` | `cfg.MODEL_ID_MOIRAI2` |
|---|---|---|
| 1 | `Salesforce/moirai-1.1-R-base` | `Salesforce/moirai-2.0-R-small` |
| 2 | `Salesforce/moirai-1.1-R-large` | `Salesforce/moirai-2.0-R-small` |
| 3 | `Salesforce/moirai-1.1-R-base` | `Salesforce/moirai-2.0-R-small` |

The variant flows through every layer:

* **Module loading**: `load_moirai_module` (1.x) vs `load_moirai2_module`
  (2.0). Moirai 2.0 skips `hydra.utils.instantiate(distr_output)` and
  converts `quantile_levels` to a tuple.
* **Forecast wrapper**: `MoiraiForecast` vs `Moirai2Forecast` (no
  `patch_size` kwarg for 2.0).
* **Per-window quantile forecast**: `forecast_quantiles_v1` (1.x — analytical
  `distr.icdf` with MC sampling fallback) vs `_forecast_quantiles_v2` (2.0 —
  reads the appropriate index from the model's quantile head).
* **Saved adapter folder**: `moirai1_lora_adapter/` vs `moirai2_lora_adapter/`
  inside the same `checkpoints/method{N}/`.
* **Results folder**: `results/method{N}/moirai1/` vs
  `results/method{N}/moirai2/` — independent metrics, plots, and PDFs per
  variant, no overwrites.

---

## Command Cheat Sheet (All Methods + Model Switching)

Use this table when you want one place that answers:
- which command to run,
- what it does,
- how to switch method (`--method 1|2|3`),
- how to switch model backbone (`--variant moirai1|moirai2`).

> Replace `N` with `1`, `2`, or `3`.

| Goal | Command template | What it does | Typical output location |
|---|---|---|---|
| Build dataset only | `python moirai/moirai.py --method N dataset` | Reads `final_csv/methodN/*.csv` and writes windowed `.npy` arrays. | `dataset/methodN/` |
| Fine-tune only | `python moirai/moirai.py --method N finetune` | Trains LoRA adapter using the method config defaults. | `checkpoints/methodN/moirai{1,2}_lora_adapter/` |
| Inference only | `python moirai/moirai.py --method N infer` | Loads fine-tuned adapter and writes predictions + quick metrics JSON. | `results/methodN/<variant>/` |
| Evaluate only | `python moirai/moirai.py --method N evaluate` | Builds plots, detailed metrics, report text, and PDF from predictions. | `results/methodN/<variant>/` |
| Full pipeline (recommended) | `python moirai/moirai.py --method N all` | Runs `dataset -> finetune -> infer -> evaluate` end-to-end. | Dataset + checkpoint + `results/methodN/<variant>/` |
| Full pipeline on CUDA | `python moirai/moirai.py --method N all --device cuda` | Same as above, but forces CUDA for fine-tuning. | Same as above |
| Full pipeline on CPU | `python moirai/moirai.py --method N all --device cpu` | Same as above, but forces CPU for fine-tuning. | Same as above |
| Smoke-test pipeline | `python moirai/moirai.py --method N all --smoke-test` | Tiny verification run (1 epoch, tiny window subset, smoke adapter). | `checkpoints/methodN/*_smoke*`, `results/methodN/<variant>/` |
| Limit train/val windows | `python moirai/moirai.py --method N finetune --max-train-windows 512 --max-val-windows 256` | Uses only a subset of windows for faster experiments. | `checkpoints/methodN/` |
| Change evaluation daylight filter | `python moirai/moirai.py --method N evaluate --ghi-filter 50` | Drops rows with measured GHI `<= 50 W/m²` for evaluation metrics/plots. | `results/methodN/<variant>/` |

### Method Selection

| Method | Command flag | Config file | Problem setup | Default model in config |
|---|---|---|---|---|
| Method 1 | `--method 1` | `moirai/configs/method1.py` | CAF target, PVLib clear-sky, single station | `moirai1` (`Salesforce/moirai-1.1-R-base`) |
| Method 2 | `--method 2` | `moirai/configs/method2.py` | CAF target, NSRDB clear-sky, multi-station | `moirai2` (`Salesforce/moirai-2.0-R-small`) |
| Method 3 | `--method 3` | `moirai/configs/method3.py` | Direct GHI target, multi-station + neighbor context | `moirai2` (`Salesforce/moirai-2.0-R-small`) |

### Model (Backbone) Switching

| Want to use | Add this flag | Effect |
|---|---|---|
| Method default model | *(no `--variant` flag)* or `--variant auto` | Uses `MODEL_ID` from `moirai/configs/methodN.py`. |
| Force Moirai 1.x | `--variant moirai1` | Uses `MODEL_ID_MOIRAI1` from method config. |
| Force Moirai 2.0 | `--variant moirai2` | Uses `MODEL_ID_MOIRAI2` from method config. |

### High-Value Ready-to-Run Examples

| Scenario | Command |
|---|---|
| Method 1 full run (default model) | `python moirai/moirai.py --method 1 all` |
| Method 1 but force Moirai 2.0 | `python moirai/moirai.py --method 1 --variant moirai2 all` |
| Method 2 full run (now defaults to small Moirai 2.0) | `python moirai/moirai.py --method 2 all` |
| Method 2 but force large Moirai 1.1 | `python moirai/moirai.py --method 2 --variant moirai1 all` |
| Method 3 full run | `python moirai/moirai.py --method 3 all` |
| Method 3 evaluate only with stricter filter | `python moirai/moirai.py --method 3 evaluate --ghi-filter 50` |
