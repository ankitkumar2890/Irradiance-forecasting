# Mapping Model Process

This folder is a dedicated home for the Phase 1 mapping model documentation.
It explains how the ERA5-to-ICON mapping pipeline works and why the model is
designed the way it is.

## What the mapping model does

The mapping model learns a relationship between ERA5 cloud fractions and ICON
cloud cover. In practice, it:

1. downloads or loads ERA5 and ICON data for the training years,
2. aligns both datasets on the same timestamps and station IDs,
3. builds tabular features from ERA5 values, time features, and station data,
4. trains a residual MLP with a gated skip connection from ERA5,
5. calibrates the raw outputs with isotonic regression,
6. evaluates the predictions against the ICON target,
7. saves model checkpoints, scalers, and validation artifacts,
8. checks the correlation ceiling to understand the hard limit of this task.

## Detailed flow

### 1. Data loading

The mapping pipeline starts by loading ERA5 and ICON data from the Phase 1
downloads folder. The relevant code path is:

- [`phase1_cloud_mapper/train_mapper.py`](/Users/IRFAN/Desktop/moirai_finetuning/phase1_cloud_mapper/train_mapper.py)
- [`phase1_cloud_mapper/fetch_data.py`](/Users/IRFAN/Desktop/moirai_finetuning/phase1_cloud_mapper/fetch_data.py)

### 2. Timestamp alignment

Both datasets are converted to IST and merged on shared time keys, plus
`station_id` when it exists. This keeps the training pairs clean and prevents
using misaligned samples.

### 3. Feature engineering

The model uses:

- ERA5 cloud fractions at time `t`
- ERA5 cloud fractions at time `t-1`
- time features
- station features

That gives the model trend context without turning the problem into a sequence
forecasting task.

### 4. Model training

The core network is a residual MLP. It learns corrections on top of ERA5 rather
than trying to completely replace it. A gated residual skip connection keeps the
raw ERA5 signal in the loop when that signal is already strong.

### 5. Regularization

Because the mapping task is noisy, the training uses strong regularization:

- dropout
- weight decay
- SmoothL1 loss
- early stopping

These choices help the model avoid memorizing noise in a low-signal setting.

### 6. Calibration

After training, isotonic regression is applied to the output. This step
improves calibration by correcting systematic bias in the raw predictions.

### 7. Evaluation

The evaluation step writes:

- validation CSVs
- metrics JSON
- time-series plots
- 4-panel diagnostic plots

This is handled by:

- [`phase1_cloud_mapper/mapping_evaluation.py`](/Users/IRFAN/Desktop/moirai_finetuning/phase1_cloud_mapper/mapping_evaluation.py)

### 8. Check correlation ceiling

Your note is especially important here:

> correlation ceiling ~0.5–0.6

If that is true, then:

- mapping cannot outperform ERA5 reliably,
- the model’s job is mostly to reduce error where possible,
- the best achievable gain may be limited by the underlying data relationship,
- the ceiling may be the fundamental limitation of the task.

This is why the code emphasizes residual learning, calibration, and careful
evaluation instead of expecting a dramatic leap over ERA5.

## Files involved

- [`phase1_cloud_mapper/config.py`](/Users/IRFAN/Desktop/moirai_finetuning/phase1_cloud_mapper/config.py)
- [`phase1_cloud_mapper/features.py`](/Users/IRFAN/Desktop/moirai_finetuning/phase1_cloud_mapper/features.py)
- [`phase1_cloud_mapper/model_architecture.py`](/Users/IRFAN/Desktop/moirai_finetuning/phase1_cloud_mapper/model_architecture.py)
- [`phase1_cloud_mapper/train_mapper.py`](/Users/IRFAN/Desktop/moirai_finetuning/phase1_cloud_mapper/train_mapper.py)
- [`phase1_cloud_mapper/mapping_evaluation.py`](/Users/IRFAN/Desktop/moirai_finetuning/phase1_cloud_mapper/mapping_evaluation.py)

## Short version

The pipeline takes ERA5, aligns it with ICON, builds tabular features, trains a
regularized residual model, calibrates the output, and evaluates whether it can
beat the ERA5 baseline. The correlation ceiling check tells us whether the task
is inherently capped, which is why it is treated as a key design constraint.
