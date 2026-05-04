# Phase 3 Direct GHI with Amazon Chronos

This folder runs a zero-shot Amazon Chronos baseline on the existing
`phase3_direct_ghi` validation dataset and writes the validation report in the
same overall format as the existing direct-GHI pipeline.

Default model:

- `amazon/chronos-bolt-mini`

Run:

```bash
./venv311/bin/python phase3_direct_ghi_chronos/run_chronos_zero_shot.py
```

Common options:

```bash
./venv311/bin/python phase3_direct_ghi_chronos/run_chronos_zero_shot.py \
  --model-id amazon/chronos-bolt-small \
  --batch-size 64 \
  --num-samples 64
```

Expected outputs land in `phase3_direct_ghi_chronos/results/`:

- `chronos_zero_shot_predictions.csv`
- `chronos_zero_shot_validation_report_data.csv`
- `chronos_zero_shot_evaluation_metrics.json`
- `chronos_zero_shot_evaluation_report.txt`
- `chronos_zero_shot_validation_report.pdf` when PDF export succeeds

Dependency note:

- Install `chronos-forecasting` in the Python 3.11 environment before running.
