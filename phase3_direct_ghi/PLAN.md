# Phase 3: Direct GHI Time-Series Forecasting — Complete Plan

## Core Strategy

Train a dedicated Moirai 2.0 time-series model for specific **local clusters** to
directly predict the final **Global Horizontal Irradiance (GHI)** in W/m².

Unlike Phase 2 (which predicts the Cloud Attenuation Factor and recovers GHI via
`CAF × clear_sky_ghi`), this pipeline targets GHI as the native forecasting
variable, eliminating the intermediate CAF recovery step and its compounding error.

---

## Spatial Grouping (The Cluster Approach)

To increase training instances without losing the specific weather signature:

- Group approximately **five nearby geographic coordinates** (e.g., adjacent 4 km ×
  4 km grid points) into a single training **cluster**.
- Train **one distinct model per cluster**.
- Each cluster member contributes its own time-series, but they share the same
  LoRA adapter — leveraging shared mesoscale weather patterns while elevation
  differentiates micro-climate effects.

### Default Cluster: South Tamil Nadu

| Station     | Lat   | Lon   | Alt (m) |
|-------------|-------|-------|---------|
| Tirunelveli | 9.14  | 77.92 | 45      |
| Madurai     | 9.93  | 78.12 | 101     |
| Coimbatore  | 11.02 | 76.96 | 411     |
| Trichy      | 10.79 | 78.70 | 88      |
| Chennai     | 13.08 | 80.27 | 7       |

---

## Target Variable

**Final Global Horizontal Irradiance (GHI)** in W/m² — predicted directly by the
model decoder.

---

## Input Feature Pipeline

### Autoregressive (Encoder only — past context)
- Historical measured **GHI** (W/m²)

### Solar Geometry
- **Solar zenith angle** (degrees)
- **Solar azimuth angle** (degrees)

### Temporal (Cyclically Encoded)
- **hour_sin**, **hour_cos** — Time of day
- **doy_sin**, **doy_cos** — Day of year

### Meteorological
- **Total cloud cover** (tcc, fraction 0–1)
- **Low cloud cover** (lcc)
- **Medium cloud cover** (mcc)
- **High cloud cover** (hcc)

### Advection Vectors
- **Wind speed** (m/s, derived from u10/v10)
- **Wind direction** (radians, derived from u10/v10)

### Spatial (Static per station)
- **Elevation** (metres, crucial for differentiating the specific geographical
  nuances of the five sites within the hyperlocal cluster)

---

## Data Layout

**Years:** 2017, 2018, 2019
**Locations:** 5-station South Tamil Nadu cluster

### Temporal Split

```
  Train               | Val               | Test
  Jan 2017 - Dec 2018 | Jan 2019 - Dec 2019| (reserved)
  ~24 months          | ~12 months        |
  ~17,520 hrs/station | ~8,760 hrs/station|
```

- **Train:** Used to fine-tune Moirai 2.0 weights via LoRA
- **Val:** Used for early stopping & LR scheduling during fine-tuning
- **Test:** Held-out test set; reserved for future data

---

## Master Architecture Diagram

```
 ═══════════════════════════════════════════════════════════════════════
 LAYER 1: DATA ACQUISITION  (01_fetch_data.py)
 ═══════════════════════════════════════════════════════════════════════

   ┌──────────────┐    ┌───────────────-┐    ┌─────────────────┐
   │ NREL NSRDB   │    │ ERA5 Reanalysis│    │ PVLib Ineichen  │
   │ msg-iodc API │    │ via CDS API    │    │ (local compute) │
   │              │    │                │    │                 │
   │ GHI (W/m²)   │    │ tcc,lcc,mcc,   │    │ zenith_angle    │
   │ hourly, IST  │    │ hcc,u10,v10    │    │ azimuth_angle   │
   │              │    │ hourly, UTC    │    │                 │
   └──────┬───────┘    └──────┬────────-┘    └────────┬────────┘
          │                   │                      │
          ▼                   ▼                      ▼
   ghi_{17-19}.csv     era5_{17-19}.csv      clearsky_{17-19}.csv


 ═══════════════════════════════════════════════════════════════════════
 LAYER 2: FEATURE ENGINEERING  (02_build_features.py)
 ═══════════════════════════════════════════════════════════════════════

   ┌────────────────────────────────────────────────────────────────┐
   │  1. ERA5: UTC → IST naive (.dt.tz_convert)                     │
   │  2. Inner join all 3 sources on datetime                       │
   │  3. Derive wind_speed & wind_direction from u10, v10           │
   │  4. Temporal: hour_sin, hour_cos, doy_sin, doy_cos             │
   │  5. Static: elevation_m injected per station                   │
   │  6. Target: GHI (w_ghr) in W/m² directly                       │
   │  7. Output: processed_data_2017_2019.csv                       │
   │                                                                │
   │  Columns (14):                                                 │
   │  station_id | datetime | w_ghr (GHI target) |                  │
   │  zenith_angle | azimuth_angle |                                │
   │  tcc | lcc | mcc | hcc | wind_speed | wind_direction |         │
   │  elevation_m | hour_sin | hour_cos | doy_sin | doy_cos         │
   └────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼

 ═══════════════════════════════════════════════════════════════════════
 LAYER 3: DATASET CONSTRUCTION  (03_build_dataset.py)
 ═══════════════════════════════════════════════════════════════════════

   ┌────────────────────────────────────────────────────────────────┐
   │  TEMPORAL SPLIT (3 Years):                                     │
   │    Train = Jan 2017 – Dec 2018                                 │
   │    Val   = Jan 2019 – Dec 2019                                 │
   │                                                                │
   │  FORMAT A: Sliding Window .npy  (for local test inference)     │
   │   ◄── 72h past ──►│◄── 24h future ──►                          │
   │                                                                │
   │  FORMAT B: GluonTS Arrow  (for uni2ts fine-tuning)             │
   │   Saved as HuggingFace Arrow dataset (Train, Val)              │
   │                                                                │
   │  KEY: target = GHI (W/m²), NOT CAF                             │
   └────────────────────────────────────────────────────────────────┘
                   │
                   ▼

 ═══════════════════════════════════════════════════════════════════════
 LAYER 4: LoRA FINE-TUNING AND INFERENCE (04_finetune.py & 05_...)
 ═══════════════════════════════════════════════════════════════════════

   ┌────────────────────────────────────────────────────────────────┐
   │  Model: Salesforce/moirai-2.0-R-small                          │
   │  Method: LoRA (rank=16, alpha=32, target: q_proj, v_proj)      │
   │  1. 04_finetune.py: Trains on Train/Val splits                 │
   │     - All 5 cluster stations train the SAME adapter            │
   │  2. 05_finetuned_inference.py: Runs predictions on Val split   │
   │     - Outputs GHI directly (no CAF recovery step)              │
   └────────────────────────────────────────────────────────────────┘
                   │
                   ▼

 ═══════════════════════════════════════════════════════════════════════
 LAYER 5: EVALUATION  (06_evaluate.py)
 ═══════════════════════════════════════════════════════════════════════

   ┌────────────────────────────────────────────────────────────────┐
   │  1. Direct GHI comparison (no recovery step needed)            │
   │  2. Daytime Metrics (RMSE, MAE, nRMSE, MAPE)                   │
   │  3. Persistence Baseline Comparison                            │
   │  4. Per-station & cluster-level metrics                        │
   │  5. Stratified & Multi-horizon Plotting                        │
   └────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
phase3_direct_ghi/
├── PLAN.md                        ← this complete plan file
├── config.py                      ← cluster definitions, feature lists, LoRA params
├── requirements.txt               ← python dependencies
│
├── 01_fetch_data.py               ← fetch GHI, ERA5, PVLib clearsky + azimuth
├── 02_build_features.py           ← merge → wind speed/dir → temporal → elevation
├── 03_build_dataset.py            ← temporal split → sliding windows + Arrow
│
├── 04_finetune.py                 ← LoRA fine-tuning (Moirai 2.0, target=GHI)
├── 05_finetuned_inference.py      ← run predictions → direct GHI output
├── 06_evaluate.py                 ← compare vs persistence, plot timeseries
│
├── downloads/                     ← raw CSVs
├── dataset/                       ← processed CSV + .npy + Arrow files
├── checkpoints/                   ← LoRA adapter weights
└── results/                       ← metrics JSONs + plots
```

---

## Key Differences from Phase 2

| Aspect                | Phase 2 (CAF)                         | Phase 3 (Direct GHI)                   |
|-----------------------|---------------------------------------|----------------------------------------|
| Target variable       | CAF ∈ [0, 1]                          | GHI (W/m²)                             |
| GHI recovery          | `CAF × clear_sky_ghi`                 | Direct output — no recovery step       |
| Solar geometry        | Zenith only                           | Zenith + **Azimuth**                   |
| Wind features         | Raw u10, v10 components               | Derived **speed** + **direction**      |
| Spatial feature       | Implicit (separate station training)  | Explicit **elevation** feature         |
| Training scope        | Single station fine-tune              | **Cluster** (5 stations, 1 adapter)    |
| Model                 | Moirai 2.0-R-small                    | Moirai 2.0-R-small                     |

---

## Run Order

```bash
cd /Users/IRFAN/Desktop/moirai_finetuning/phase3_direct_ghi

export NREL_API_KEY="your-key"
export NREL_EMAIL="your@email.com"
export CDSAPI_KEY="your-cds-key"

# ── Data Pipeline ──
python 01_fetch_data.py            # download/verify data
python 02_build_features.py        # merge → derive wind → temporal → elevation
python 03_build_dataset.py         # format for training

# ── Fine-Tuning ──
python 04_finetune.py              # LoRA training on train/val (cluster)
python 05_finetuned_inference.py   # Run val window predictions

# ── Evaluation ──
python 06_evaluate.py              # Produce metrics and graphs
```
