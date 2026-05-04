# Standalone Direct GHI with Amazon Chronos

This folder is self-contained. It includes:

- its own `run_chronos_zero_shot.py`
- its own training pipeline entrypoint: `pipeline.py`
- local helper files: `config.py`, `master_metrics.py`, `master_plots.py`, `export_pdf.py`
- local training pipeline modules under `configs/` and `functions/`
- local source CSV under `final_csv/method3/`
- a local copy of the dataset in `dataset/`
- a local copy of the station downloads in `downloads/multi_station/`
- local outputs in `results/`

Default model:

- `amazon/chronos-bolt-mini`

## Zero-shot baseline

Run:

```bash
./venv311/bin/python chronos/run_chronos_zero_shot.py
```

Common options:

```bash
./venv311/bin/python chronos/run_chronos_zero_shot.py \
  --model-id amazon/chronos-bolt-small \
  --batch-size 64 \
  --num-samples 64
```

Expected outputs land in `chronos/results/`:

- `chronos_zero_shot_predictions.csv`
- `chronos_zero_shot_validation_report_data.csv`
- `chronos_zero_shot_evaluation_metrics.json`
- `chronos_zero_shot_evaluation_report.txt`
- `chronos_zero_shot_validation_report.pdf` when PDF export succeeds

## Fine-tuning pipeline

The new standalone training path mirrors the Moirai Method 3 workflow:

- `dataset` builds windows from `final_csv/method3/processed_data_2017_2019.csv`
- `finetune` trains a LoRA adapter on Chronos-Bolt
- `infer` writes validation forecasts
- `evaluate` writes the text report, metrics JSON, plots, and PDF
- `all` runs the full chain

Commands:

```bash
./venv311/bin/python chronos/pipeline.py dataset
./venv311/bin/python chronos/pipeline.py finetune
./venv311/bin/python chronos/pipeline.py infer
./venv311/bin/python chronos/pipeline.py evaluate
./venv311/bin/python chronos/pipeline.py all
```

Useful flags:

```bash
./venv311/bin/python chronos/pipeline.py --model-id amazon/chronos-bolt-small all
./venv311/bin/python chronos/pipeline.py --device cuda finetune
./venv311/bin/python chronos/pipeline.py --smoke-test all
./venv311/bin/python chronos/pipeline.py --max-train-windows 512 --max-val-windows 128 finetune
```

Fine-tuned outputs land under a model-specific folder, for example:

- `chronos/checkpoints/method3/amazon__chronos-bolt-mini/`
- `chronos/results/method3/amazon__chronos-bolt-mini/`

The fine-tune pipeline uses Chronos-Bolt's native quantile training loss on:

- context: past 72 hours of `w_ghr`
- target: next 24 hours of `w_ghr`
- split: train `2017-2018`, validation `2019`
- anchor hour: `06:00`

Dependency note:

- Install `chronos-forecasting` in the Python 3.11 environment before running.
