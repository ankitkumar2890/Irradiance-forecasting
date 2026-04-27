# TFT Model Project Guide

## What This Repository Is Doing

This repository is a solar forecasting research workspace centered on predicting **CAF (Cloud Attenuation Factor)** and then recovering solar irradiance from it:

```text
CAF = measured_GHI / clear_sky_GHI
predicted_GHI = predicted_CAF * clear_sky_GHI
```

The project is organized into multiple experiments that all work on related weather and radiation forecasting tasks for a set of Tamil Nadu stations:

- `phase1_cloud_mapper/`: learns a cloud-mapping model from ERA5 features.
- `phase2_finetuning/`: fine-tunes Moirai using Phase 1 style inputs.
- `phase2_era5_direct/`: fine-tunes Moirai directly from ERA5 features.
- `tft_model/`: a self-contained **Temporal Fusion Transformer (TFT)** baseline built in pure PyTorch.

This `tft_model/` folder is important because it provides a strong, interpretable baseline that does not depend on Moirai or GluonTS fine-tuning machinery. It uses the same forecasting idea, the same 72h-to-24h window setup, and the same five stations, but trains a local TFT model end to end.

## What `tft_model/` Predicts

The TFT pipeline predicts the next **24 hours of CAF** using:

- The previous **72 hours** of history
- Known future covariates for the next 24 hours
- Multi-station data from:
  - `tirunelveli`
  - `madurai`
  - `coimbatore`
  - `trichy`
  - `chennai`

The target is always `CAF`, while the model uses weather and time features such as:

- `clear_sky_ghi`
- ERA5 cloud features: `tcc`, `lcc`, `mcc`, `hcc`
- ERA5 wind features: `u10`, `v10`
- `zenith_angle`
- cyclic time features: `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`

## High-Level Pipeline

```text
01_fetch_data.py
    Download or prepare station-level GHI + ERA5 + clear-sky inputs
        ↓
02_build_features.py
    Merge sources and compute CAF + temporal features
        ↓
03_build_dataset.py
    Build sliding windows:
    72 hours past  →  24 hours future
        ↓
train_tft.py
    Train TFT, save checkpoint, predictions, metrics, and plots
```

## How This Fits With the Rest of the Repo

The full repository is not just one model; it is a set of related forecasting experiments:

### 1. Phase 1 Cloud Mapper

`phase1_cloud_mapper/` builds an earlier model that maps ERA5 meteorology to cloud-related signals. It acts as an upstream experiment and data source for some of the Moirai pipelines.

### 2. Moirai Fine-Tuning Pipelines

There are two major Moirai branches:

- `phase2_finetuning/`: uses Phase 1 style or synthetic cloud inputs
- `phase2_era5_direct/`: uses direct ERA5 features

These are transformer/foundation-model experiments with LoRA fine-tuning.

### 3. TFT Baseline

`tft_model/` is the classical deep-learning baseline:

- fully local PyTorch implementation
- easier to inspect and debug
- exposes variable importance through TFT variable selection networks
- produces comparable prediction CSVs for downstream comparison

So in practice:

- Moirai branches test foundation-model fine-tuning
- TFT gives you a transparent baseline on the same forecasting task

## Folder Structure

```text
tft_model/
├── 01_fetch_data.py
├── 02_build_features.py
├── 03_build_dataset.py
├── config.py
├── dataset.py
├── model.py
├── train_tft.py
├── downloads/
├── dataset/
├── checkpoints/
└── results/
```

## Main Files Explained

### `config.py`

This is the central configuration file for the TFT experiment. It defines:

- all important paths
- API keys from environment variables
- the five station definitions
- year range: `2017, 2018, 2019`
- split boundaries
- window sizes:
  - `PAST_STEPS = 72`
  - `FUTURE_STEPS = 24`
- feature lists for encoder and decoder
- model hyperparameters
- training hyperparameters

Notable split setup:

- Train: up to `2018-12-31 23:00`
- Validation: `2019-01-01 00:00` to `2019-12-31 23:00`
- Test: from `2020-01-01 00:00`

Because the configured data years stop at 2019, the current test split is expected to be empty unless newer data is added.

### `01_fetch_data.py`

This script prepares the raw inputs needed by the TFT pipeline.

It does three key jobs:

1. Downloads hourly **GHI** from the NREL NSRDB API for each station-year.
2. Loads or builds a multi-station **ERA5** source with the exact cloud and wind variables needed by TFT.
3. Produces station-level yearly CSVs inside `tft_model/downloads/multi_station_era5/`.

Important details:

- It can reuse an existing raw ERA5 CSV from `phase1_cloud_mapper/downloads/era5_2017_2019.csv`.
- If required TFT ERA5 columns are missing, it can build a TFT-specific enriched ERA5 file.
- It uses `CDSAPI_KEY` when it needs to download missing ERA5 variables from Copernicus.
- It handles NREL data alignment issues like half-hour timestamps.

The result is a local raw-data layer for the TFT experiment.

### `02_build_features.py`

This script merges:

- measured GHI
- ERA5 weather features
- clear-sky irradiance

Then it computes:

- `CAF`
- time-based cyclic features
- a cleaned, merged multi-station dataframe

Final output:

- `tft_model/dataset/processed_data_2017_2019.csv`

This is the core tabular feature file used by the next stage.

### `03_build_dataset.py`

This script converts the processed continuous time series into sliding windows.

For each station:

- take `72` past hours as encoder input
- take `24` future hours as decoder-known input
- predict the `24` future CAF values

It saves NumPy arrays for each split:

- `X_past_train.npy`
- `X_future_train.npy`
- `y_future_train.npy`
- and the same for `val` and `test`

It also saves:

- timestamps for each horizon
- station ids for each sample

A window is only kept when:

- the forecast starts at hour `06:00`
- the past context spans at least 3 calendar days

That gives the model a standardized daily forecast anchor.

### `dataset.py`

This wraps the `.npy` arrays into a PyTorch `Dataset` and `DataLoader`.

Returned tensors per sample:

- encoder input: `(72, encoder_features)`
- decoder input: `(24, decoder_features)`
- target: `(24,)`

This file is the bridge between saved NumPy windows and model training.

### `model.py`

This contains a pure PyTorch implementation of the **Temporal Fusion Transformer**.

Main architectural blocks:

- `GatedLinearUnit`
- `GatedResidualNetwork`
- `VariableSelectionNetwork`
- `InterpretableMultiHeadAttention`
- `TemporalFusionTransformer`

What the model does:

- selects important variables separately for past and future inputs
- encodes past history with an LSTM
- decodes future-known covariates with another LSTM
- applies causal self-attention over the combined temporal sequence
- outputs 24-step forecasts of CAF

The model also returns:

- encoder variable weights
- decoder variable weights
- attention weights

That makes it more interpretable than a plain black-box forecaster.

### `train_tft.py`

This is the end-to-end training script.

It:

- loads train and validation datasets
- builds the TFT model
- trains with `AdamW`
- uses cosine annealing learning-rate scheduling
- applies gradient clipping
- performs early stopping
- saves the best checkpoint
- runs final validation inference
- exports metrics, predictions, and interpretability plots

Outputs include:

- `tft_model/checkpoints/tft_best.pt` or `tft_best_smoke.pt`
- `tft_model/results/tft_predictions.csv`
- `tft_model/results/tft_metrics.json`
- `tft_model/results/tft_training_curves.png`
- `tft_model/results/tft_variable_importance.png`

It also writes a comparison-friendly prediction file to:

- `phase2_finetuning/results/finetuned_predictions_tft.csv`

That helps compare TFT against Moirai outputs using similar downstream tooling.

## Data Sources

This TFT experiment uses a mix of external and local sources:

- **NREL NSRDB** for measured GHI
- **ERA5** for cloud and wind predictors
- **PVLib clear-sky calculations** for clear-sky irradiance and solar geometry
- local repository structure for shared raw inputs from other phases

## Feature Design

The TFT setup separates past-observed features and future-known features.

### Past features

Used in the encoder:

- `CAF`
- `clear_sky_ghi`
- `tcc`
- `lcc`
- `mcc`
- `hcc`
- `u10`
- `v10`
- `zenith_angle`
- `hour_sin`
- `hour_cos`
- `doy_sin`
- `doy_cos`

### Future features

Used in the decoder:

- `clear_sky_ghi`
- `tcc`
- `lcc`
- `mcc`
- `hcc`
- `u10`
- `v10`
- `zenith_angle`
- `hour_sin`
- `hour_cos`
- `doy_sin`
- `doy_cos`

The only past-only feature is `CAF`, because future target values are unknown at inference time.

## Training Setup

Current default configuration in `config.py`:

- hidden size: `64`
- attention heads: `4`
- LSTM layers: `1`
- dropout: `0.1`
- batch size: `32`
- learning rate: `1e-3`
- weight decay: `1e-5`
- max epochs: `100`
- patience: `10`
- loss: `mse`

There is also a smoke-test mode:

```bash
python tft_model/train_tft.py --smoke-test
```

This runs a much smaller sanity-check training pass and saves `tft_best_smoke.pt`.

## End-to-End Run Order

From the repo root:

```bash
export NREL_API_KEY="your-key"
export NREL_EMAIL="your-email"
export CDSAPI_KEY="your-cds-key"   # needed only if ERA5 enrichment is required

python tft_model/01_fetch_data.py
python tft_model/02_build_features.py
python tft_model/03_build_dataset.py
python tft_model/train_tft.py
```

## Important Outputs

After a normal successful run, the most important artifacts are:

- processed features: `tft_model/dataset/processed_data_2017_2019.csv`
- training windows: `tft_model/dataset/*.npy`
- best model checkpoint: `tft_model/checkpoints/tft_best.pt`
- validation predictions: `tft_model/results/tft_predictions.csv`
- metrics summary: `tft_model/results/tft_metrics.json`
- training curve plot: `tft_model/results/tft_training_curves.png`
- variable-importance plot: `tft_model/results/tft_variable_importance.png`

## Practical Notes

- `tft_model/` is designed to be self-contained, even though it can reuse shared raw data from `phase1_cloud_mapper/`.
- The current checked-in `results/` and `checkpoints/` already show that a smoke test has been run.
- The `dataset/` directory may be empty until you run the preprocessing scripts.
- The configured test range begins in 2020, so with only 2017-2019 data you should expect no test windows unless you extend the dataset.

## In One Sentence

`tft_model/` is the repository's interpretable PyTorch baseline for multi-station, multi-horizon CAF forecasting, built to mirror the broader solar forecasting project while remaining simpler to inspect, train, and compare against the Moirai experiments.
