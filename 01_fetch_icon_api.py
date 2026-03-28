#!/usr/bin/env python3
"""
01 — Fetch ICON Historical Predictors (v3 — Null-Safe)
========================================================

Key fix over v2:
    The Open-Meteo Archive API (/v1/archive) does NOT guarantee all
    variables for all models × date ranges.  When 'models=icon_global'
    is requested, some variables (e.g. wind_direction_10m, cape,
    direct_normal_irradiance) may come back as arrays of null / None.

    Calling np.radians() on a column of None → TypeError:
        "loop of ufunc does not support argument 0 of type NoneType
         which has no callable radians method"

    This version:
        1. Diagnoses which variables the API actually returned vs null
        2. Skips derived features whose inputs are missing
        3. Falls back to ERA5 reanalysis (no model filter) if ICON
           returns too many nulls
        4. Prints a clear variable availability report
"""

import sys
import numpy as np
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TARGET_LAT, TARGET_LON, ICON_RAW_CSV, DOWN_DIR, START_DATE, END_DATE


# ═══════════════════════════════════════════════════════════════════
#  Solar Geometry — computed locally, never depends on the API
# ═══════════════════════════════════════════════════════════════════

def solar_zenith_angle(times: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """
    Spencer (1971) solar position.
    Returns cos(θ_z) clipped to [0, 1].
    """
    doy = times.dayofyear.values.astype(float)
    hour_utc = times.hour.values + times.minute.values / 60.0

    B = (2 * np.pi / 365) * (doy - 1)
    decl = (0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
            - 0.006758 * np.cos(2*B) + 0.000907 * np.sin(2*B)
            - 0.002697 * np.cos(3*B) + 0.00148 * np.sin(3*B))

    eot = 229.18 * (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B)
                     - 0.014615 * np.cos(2*B) - 0.04089 * np.sin(2*B))

    tst = hour_utc * 60 + eot + 4 * lon
    ha = np.radians((tst / 4.0) - 180.0)

    lat_r = np.radians(lat)
    cos_sza = (np.sin(lat_r) * np.sin(decl)
               + np.cos(lat_r) * np.cos(decl) * np.cos(ha))
    return np.clip(cos_sza, 0.0, 1.0)


def clear_sky_ghi_ineichen(cos_sza: np.ndarray,
                            altitude_m: float = 0.0) -> np.ndarray:
    """Simplified Ineichen–Perez clear-sky GHI (W/m²)."""
    I0 = 1361.0
    sza_deg = np.degrees(np.arccos(np.clip(cos_sza, 1e-6, 1.0)))
    am = 1.0 / (cos_sza + 0.50572 * (96.07995 - sza_deg) ** -1.6364)
    am = np.clip(am, 1, 40)

    TL = 2.5
    fh1 = np.exp(-altitude_m / 8000.0)
    fh2 = np.exp(-altitude_m / 1250.0)
    cg1 = 5.09e-5 * altitude_m + 0.868
    cg2 = 3.92e-5 * altitude_m + 0.0387

    ghi_clear = (cg1 * I0 * cos_sza
                 * np.exp(-cg2 * am * (fh1 + fh2 * (TL - 1)))
                 * np.exp(0.01 * am**1.8))

    ghi_clear = np.where(cos_sza > 0.01, ghi_clear, 0.0)
    return np.clip(ghi_clear, 0, I0)


# ═══════════════════════════════════════════════════════════════════
#  Null-diagnosis helper
# ═══════════════════════════════════════════════════════════════════

def _check_var(hourly: dict, api_name: str) -> bool:
    """Return True if the variable exists and has at least one non-null value."""
    vals = hourly.get(api_name)
    if vals is None:
        return False
    # The API returns a list; check if it's all None
    if all(v is None for v in vals):
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
#  Main Fetch  (with fallback & null guards)
# ═══════════════════════════════════════════════════════════════════

def fetch_historical(lat: float, lon: float,
                     start_date: str, end_date: str,
                     model: str | None = None) -> pd.DataFrame | None:
    """
    Fetch from Open-Meteo Archive API.

    If model is None  → default reanalysis (ERA5-based, most complete).
    If model is set   → request that specific model (may have gaps).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    # ── All variables we want (superset) ─────────────────────────
    hourly_vars = [
        "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
        "shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
        "temperature_2m", "surface_pressure", "relative_humidity_2m",
        "cape",
        "wind_speed_10m", "wind_direction_10m",
    ]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
        "timezone": "UTC",
    }
    if model:
        params["models"] = model

    label = model or "best_match (ERA5 reanalysis)"
    try:
        print(f"\n  Requesting {label}: {start_date} → {end_date} ...")
        resp = requests.get(url, params=params, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        if "hourly" not in data:
            print(f"  [ERROR] Response has no 'hourly' key. Keys: {list(data.keys())}")
            return None
        hourly = data["hourly"]
    except Exception as e:
        print(f"  [ERROR] API request failed: {e}")
        return None

    # ── Diagnose variable availability ───────────────────────────
    print(f"\n  Variable availability report ({label}):")
    avail = {}
    for var in hourly_vars:
        ok = _check_var(hourly, var)
        avail[var] = ok
        status = "✅" if ok else "❌ NULL"
        print(f"    {var:35s} {status}")

    # ── Core requirement: at least cloud_cover + temperature must exist ──
    if not avail["cloud_cover"] or not avail["temperature_2m"]:
        print("\n  [FATAL] Core variables (cloud_cover, temperature_2m) are null.")
        return None

    # ── Build DataFrame with only available columns ──────────────
    time_idx = pd.to_datetime(hourly["time"], utc=True)
    df = pd.DataFrame({"time": time_idx})

    # Mapping: our_column_name → api_variable_name
    col_map = {
        "total_cloud":      "cloud_cover",
        "low_cloud":        "cloud_cover_low",
        "mid_cloud":        "cloud_cover_mid",
        "high_cloud":       "cloud_cover_high",
        "ghi":              "shortwave_radiation",
        "dni":              "direct_normal_irradiance",
        "dhi":              "diffuse_radiation",
        "temperature_2m":   "temperature_2m",
        "surface_pressure": "surface_pressure",
        "humidity_2m":      "relative_humidity_2m",
        "cape":             "cape",
        "wind_speed":       "wind_speed_10m",
        "wind_dir":         "wind_direction_10m",
    }

    for col_name, api_name in col_map.items():
        if avail.get(api_name, False):
            df[col_name] = pd.to_numeric(hourly[api_name], errors="coerce")
        else:
            df[col_name] = np.nan

    # ── Wind U/V decomposition (GUARDED) ─────────────────────────
    if avail.get("wind_speed_10m") and avail.get("wind_direction_10m"):
        wd_rad = np.radians(df["wind_dir"].values.astype(float))
        df["U_wind"] = -df["wind_speed"].values * np.sin(wd_rad)
        df["V_wind"] = -df["wind_speed"].values * np.cos(wd_rad)
        print("  ✅ Wind U/V decomposed from speed + direction")
    else:
        df["U_wind"] = np.nan
        df["V_wind"] = np.nan
        print("  ⚠️  Wind speed/direction NULL → U_wind, V_wind set to NaN")
    df.drop(columns=["wind_speed", "wind_dir"], inplace=True)

    # ── Solar geometry (computed, never null) ─────────────────────
    cos_sza = solar_zenith_angle(time_idx, lat, lon)
    df["cos_sza"] = cos_sza

    # ── Diffuse fraction k_d (GUARDED) ───────────────────────────
    if avail.get("shortwave_radiation") and avail.get("diffuse_radiation"):
        ghi_safe = df["ghi"].replace(0, np.nan)
        df["k_d"] = (df["dhi"] / ghi_safe).clip(0, 1)
        print("  ✅ Diffuse fraction k_d computed")
    else:
        df["k_d"] = np.nan
        print("  ⚠️  GHI or DHI NULL → k_d set to NaN")

    # ── Clear-sky GHI & empirical alpha (GUARDED) ────────────────
    ghi_clear = clear_sky_ghi_ineichen(cos_sza, altitude_m=0.0)
    df["ghi_clear"] = ghi_clear

    if avail.get("shortwave_radiation"):
        ghi_clear_safe = np.where(ghi_clear > 20.0, ghi_clear, np.nan)
        df["alpha"] = 1.0 - (df["ghi"].values / ghi_clear_safe)
        df["alpha"] = df["alpha"].clip(0, 1)
        print("  ✅ Attenuation factor α computed")
    else:
        df["alpha"] = np.nan
        print("  ⚠️  GHI NULL → α set to NaN (will need satellite-only approach)")

    # ── Temporal lag features ────────────────────────────────────
    lag_cols = ["temperature_2m"]
    if avail.get("cloud_cover_low"):
        lag_cols.append("low_cloud")
    if avail.get("shortwave_radiation"):
        lag_cols.extend(["alpha", "k_d"])

    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_t-1"] = df[col].shift(1)

    # Cloud tendency
    if avail.get("cloud_cover"):
        df["d_total_cloud"] = df["total_cloud"].diff()
    if avail.get("cloud_cover_high"):
        df["d_high_cloud"] = df["high_cloud"].diff()

    # ── Drop first row (lag NaN) ─────────────────────────────────
    df = df.iloc[1:].reset_index(drop=True)

    return df


# ═══════════════════════════════════════════════════════════════════
#  Entrypoint with automatic fallback
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  01 — Fetch ICON Historical Predictors (v3 Null-Safe)")
    print("=" * 64)

    # ── Attempt 1: ICON-Global ───────────────────────────────────
    print("\n[1/2] Trying model=icon_global ...")
    df = fetch_historical(TARGET_LAT, TARGET_LON, START_DATE, END_DATE,
                          model="icon_global")

    # Check if we got enough usable columns
    if df is not None:
        null_frac = df.drop(columns=["time"]).isna().all()
        all_null_cols = null_frac[null_frac].index.tolist()
        if len(all_null_cols) > 5:
            print(f"\n  ⚠️  ICON-Global returned {len(all_null_cols)} fully-null columns.")
            print(f"      Columns: {all_null_cols}")
            print("      Falling back to default reanalysis (ERA5) ...")
            df = None

    # ── Attempt 2: Fallback to default reanalysis (ERA5) ────────
    if df is None:
        print("\n[2/2] Falling back to default reanalysis (no model filter) ...")
        df = fetch_historical(TARGET_LAT, TARGET_LON, START_DATE, END_DATE,
                              model=None)

    if df is None:
        print("\n[FATAL] Both ICON-Global and ERA5 fallback failed.")
        return

    # ── Save ─────────────────────────────────────────────────────
    DOWN_DIR.mkdir(parents=True, exist_ok=True)
    ICON_RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ICON_RAW_CSV, index=False)

    non_null_cols = [c for c in df.columns if c != "time" and not df[c].isna().all()]
    print(f"\n{'='*64}")
    print(f"  ✅ Saved {len(df)} rows → {ICON_RAW_CSV}")
    print(f"     Usable columns ({len(non_null_cols)}): {non_null_cols}")

    # Daytime statistics
    day = df[df["cos_sza"] > 0.1]
    if len(day) > 0 and "alpha" in non_null_cols:
        print(f"\n     Daytime α:  mean={day['alpha'].mean():.3f}  "
              f"std={day['alpha'].std():.3f}  "
              f"median={day['alpha'].median():.3f}")
    if len(day) > 0 and "k_d" in non_null_cols:
        print(f"     Daytime k_d: mean={day['k_d'].mean():.3f}  "
              f"std={day['k_d'].std():.3f}")

    print(f"\n     First 3 rows:")
    print(df.head(3).to_string())
    print()


if __name__ == "__main__":
    main()