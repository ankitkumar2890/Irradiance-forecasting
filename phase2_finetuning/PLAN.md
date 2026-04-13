# Phase 2: Moirai CAF Fine-Tuning Pipeline — Complete Plan

## Objective

Fine-tune the Salesforce Moirai time-series foundation model to predict **CAF (Cloud Attenuation Factor)** for solar GHI forecasting at Tirunelveli, India.

```
CAF = measured_GHI / clear_sky_GHI      (clipped [0, 1])
predicted_GHI = predicted_CAF * clear_sky_GHI   (at inference)
```

No zero-shot analysis is performed. We go straight to fine-tuning on the multi-year dataset.

---

## Data Layout

**Years:** 2017, 2018, 2019
**Location:** Tirunelveli, Tamil Nadu (9.14N, 77.92E, 45m ASL) (Same primary station as Phase 1)

### Temporal Split

We split the 3 years (Jan 2017 to Dec 2019) chronologically to ensure no data leakage from the future into the past.

```
  Train               | Val               | Test
  Jan 2017 - Jun 2019 | Jul 2019 - Sep 2019| Oct 2019 - Dec 2019
  ~30 months          | ~3 months         | ~3 months
  ~21,900 hours       | ~2,208 hours      | ~2,208 hours
```

- **Train:** Used to fine-tune Moirai weights via LoRA
- **Val:** Used for early stopping & LR scheduling during fine-tuning
- **Test:** Held-out test set; NEVER seen during training; final metrics reported here

---

## Master Architecture Diagram

```
 ═══════════════════════════════════════════════════════════════════════
 LAYER 1: DATA ACQUISITION  (01_fetch_data.py)
 ═══════════════════════════════════════════════════════════════════════

   ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐
   │ NREL NSRDB   │    │ Phase 1 Synth │    │ PVLib Ineichen  │
   │ msg-iodc API │    │ CloudMapper   │    │ (local compute) │
   │              │    │               │    │                 │
   │ GHI (W/m²)   │    │ cloud_cover   │    │ clear_sky_ghi   │
   │ hourly, IST  │    │ hourly, UTC   │    │ zenith_angle    │
   │              │    │               │    │ hourly, IST     │
   └──────┬───────┘    └──────┬────────┘    └────────┬────────┘
          │                   │                      │
          ▼                   ▼                      ▼
   ghi_{17-19}.csv     icon_{17-19}.csv      clearsky_{17-19}.csv


 ═══════════════════════════════════════════════════════════════════════
 LAYER 2: FEATURE ENGINEERING  (02_build_features.py)
 ═══════════════════════════════════════════════════════════════════════

   ┌────────────────────────────────────────────────────────────────┐
   │  1. Synthetic ICON: UTC → IST naive (.dt.tz_convert)          │
   │  2. Inner join all 3 sources on datetime                      │
   │  3. CAF = GHI / clear_sky_GHI  (clip [0,1], night = 0)       │
   │  4. Temporal: hour_sin, hour_cos, doy_sin, doy_cos            │
   │  5. Output: processed_data_2017_2019.csv                      │
   │                                                                │
   │  Columns (9):                                                  │
   │  datetime | CAF | clear_sky_ghi | cloud_cover | zenith_angle  |
   │  hour_sin | hour_cos | doy_sin | doy_cos                      │
   └────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼

 ═══════════════════════════════════════════════════════════════════════
 LAYER 3: DATASET CONSTRUCTION  (03_build_dataset.py)
 ═══════════════════════════════════════════════════════════════════════

   ┌────────────────────────────────────────────────────────────────┐
   │  TEMPORAL SPLIT (3 Years):                                     │
   │    Train = Jan 2017 – Jun 2019                                 │
   │    Val   = Jul 2019 – Sep 2019                                 │
   │    Test  = Oct 2019 – Dec 2019                                 │
   │                                                                │
   │  FORMAT A: Sliding Window .npy  (for local test inference)    │
   │   ◄── 72h past ──►│◄── 24h future ──►                        │
   │                                                                │
   │  FORMAT B: GluonTS Arrow  (for uni2ts fine-tuning)             │
   │   Saved as HuggingFace Arrow dataset (Train, Val, Test)        │
   └────────────────────────────────────────────────────────────────┘
                   │
                   ▼

 ═══════════════════════════════════════════════════════════════════════
 LAYER 4: LoRA FINE-TUNING AND INFERENCE (04_finetune.py & 05_...)
 ═══════════════════════════════════════════════════════════════════════

   ┌────────────────────────────────────────────────────────────────┐
   │  Model: Salesforce/moirai-1.1-R-base                           │
   │  Method: LoRA (rank=16, alpha=32, target: q_proj, v_proj)     │
   │  1. 04_finetune.py: Trains on Train/Val splits                 │
   │  2. 05_finetuned_inference.py: Runs predictions on Test split  │
   └────────────────────────────────────────────────────────────────┘
                   │
                   ▼

 ═══════════════════════════════════════════════════════════════════════
 LAYER 5: EVALUATION  (06_evaluate.py)
 ═══════════════════════════════════════════════════════════════════════

   ┌────────────────────────────────────────────────────────────────┐
   │  1. GHI Recovery (CAF * clear_sky)                            │
   │  2. Daytime Metrics (RMSE, MAE, nRMSE, MAPE)                  │
   │  3. Persistence Baseline Comparison                           │
   │  4. Stratified & Multi-horizon Plotting                       │
   └────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
phase2_finetuning/
├── PLAN.md                        ← this complete plan file
├── config.py                      ← multi-year paths, LoRA hyperparams
├── requirements.txt               ← python dependencies
│
├── 01_fetch_data.py               ← fetch GHI, load Phase 1 ICON, PVLib clearsky
├── 02_build_features.py           ← merge 3 sources → CAF + temporal CSV
├── 03_build_dataset.py            ← 3-year split → sliding windows (.npy) + Arrow
│
├── 04_finetune.py                 ← the main fine-tuning script (un2its + peft)
├── 05_finetuned_inference.py      ← run test data through the LoRA adapter
├── 06_evaluate.py                 ← compare vs persistence, plot timeseries & scatter
│
├── downloads/                     ← raw CSVs
├── dataset/                       ← processed CSV + .npy + Arrow files
├── checkpoints/                   ← LoRA adapter weights target directory
└── results/                       ← metrics JSONs + plots target directory
```

---

## Detailed Step Descriptions

### Step 1: `01_fetch_data.py`
Sources GHI from NREL directly (2017 to 2019). Uses the exact `icon_synthetic_2017_2019.csv` output produced by the Phase 1 CloudMapper model to get the `cloud_cover` covariate. Uses PVLib to generate `clear_sky_ghi`.

### Step 2: `02_build_features.py`
Standardizes all timestamps to IST Naive. Calculates CAF (`GHI / clear_sky_ghi`) explicitly clipped between `0.0` and `1.0`. Any hours where clear sky GHI is `< 1.0` (night) output a CAF of 0. Generates `hour_sin/cos` and `doy_sin/cos`. **Only 7 covariates are used** because CloudMapper strictly predicts `cloud_cover`, not altitude specific bands.

### Step 3: `03_build_dataset.py`
Parses the multi-year history into Train (Jan '17 - Jun '19), Val (Jul '19 - Sep '19), and Test (Oct '19 - Dec '19). Combines into NumPy windows mapping a 72-hour `PAST` history targeting a 24-hour `FUTURE` prediction. Translates this structure natively into GluonTS-compatible streaming `Arrow` datasets for the official fine-tuning pipeline.

### Step 4: `04_finetune.py`
Initalizes `moirai-1.1-R-base` and freezes the base weights. Appends trainable LoRA matrices (`r=16, alpha=32`) to the Attention `q_proj` and `v_proj` modules targeting roughly 400K parameters. Operates via PyTorch optimized with AdamW on `bf16-mixed` precision. Stops early if Val Loss degrades 5 epochs consecutively.

### Step 5: `05_finetuned_inference.py`
Loads the frozen Moirai base model and merges the LoRA adapter computed in the previous step. Feeds the entire Test `.npy` split exactly and aggregates `CAF_pred`. Outputs `.csv` and `.json` prediction structures.

### Step 6: `06_evaluate.py`
Performs GHI recovery via multiplication over clear-sky logic for the test set. Removes nighttime operations (`zenith > 85`). Outputs overall Daytime Error (RMSE/MAE) and a direct forecast skill improvement relative to simple Persistence tracking. Distributes plots over prediction horizons (degradation curve) and provides sample timeseries outputs for Nov 2019.

---

## Run Order

```bash
cd /Users/IRFAN/Desktop/moirai_finetuning/phase2_finetuning

export NREL_API_KEY="your-key"
export NREL_EMAIL="your@email.com"

# ── Data Pipeline ──
python 01_fetch_data.py            # download/verify data
python 02_build_features.py        # merge → CAF → temporal
python 03_build_dataset.py         # format for training

# ── Stage B: Fine-Tuning ──
python 04_finetune.py              # LoRA training on train/val
python 05_finetuned_inference.py   # Run test window predictions

# ── Stage C: Evaluation ──
python 06_evaluate.py              # Produce metrics and graphs
```
