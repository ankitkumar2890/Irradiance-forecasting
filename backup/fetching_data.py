# =============================================================================
# ERA5-Land SSRD Downloader
# 45 km × 45 km box centered on site
# 9 km × 9 km internal grid (5 × 5)
# 5 AM – 7 PM IST, 2024
# =============================================================================

import cdsapi
import os
import glob
import zipfile
import xarray as xr
import numpy as np
import pandas as pd
import calendar
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

sites = [
    {"name": "Solar_Site_01", "lat": 9.14,  "lon": 77.92},
    {"name": "Solar_Site_02", "lat": 27.99, "lon": 79.94},
    {"name": "Solar_Site_03", "lat": 22.08, "lon": 70.39}
]

VARIABLE = ["surface_solar_radiation_downwards"]

# IST 5 AM – 7 PM  → UTC = 23, 00, 01, ... 13
utc_hours = list(range(23, 24)) + list(range(0, 14))
TIMES = [f"{h:02d}:00" for h in utc_hours]

# Physical grid definition
BOX_KM = 45.0            # total box size
HALF_BOX_KM = BOX_KM / 2 # 22.5 km
GRID_SPACING_KM = 9.0    # ERA5-Land resolution
GRID_STEPS = np.array([-2, -1, 0, 1, 2])  # 5 × 5 grid

DOWNLOAD_DIR = "era5_ssrd_5am_7pm_IST_2024"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# CDS CLIENT
# -----------------------------------------------------------------------------

def get_cds_client():
    home = os.path.expanduser("~")
    rcfile = os.path.join(home, ".cdsapirc")

    if os.path.exists(rcfile):
        os.remove(rcfile)

    url = os.environ.get("CDSAPI_URL", "").strip()
    key = os.environ.get("CDSAPI_KEY", "").strip()

    if not url or not key:
        raise RuntimeError("Set CDSAPI_URL and CDSAPI_KEY in .env")

    with open(rcfile, "w") as f:
        f.write(f"url: {url}\nkey: {key}\n")

    return cdsapi.Client()

client = get_cds_client()

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def get_days(year, month):
    ndays = calendar.monthrange(int(year), int(month))[1]
    return [f"{d:02d}" for d in range(1, ndays + 1)]

def open_era5_dataset(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            nc_files = [f for f in z.namelist() if f.endswith(".nc")]
            if not nc_files:
                raise RuntimeError("ZIP contains no NetCDF")
            z.extract(nc_files[0], os.path.dirname(path))
            path = os.path.join(os.path.dirname(path), nc_files[0])

    try:
        return xr.open_dataset(path, engine="netcdf4")
    except Exception:
        return xr.open_dataset(path, engine="h5netcdf")

# -----------------------------------------------------------------------------
# DOWNLOAD
# -----------------------------------------------------------------------------

def download_era5_land(site, year, month):
    lat = site["lat"]
    lon = site["lon"]

    lat_offset = HALF_BOX_KM / 111.0
    lon_offset = HALF_BOX_KM / (111.0 * np.cos(np.deg2rad(lat)))

    area = [
        lat + lat_offset,  # North
        lon - lon_offset,  # West
        lat - lat_offset,  # South
        lon + lon_offset   # East
    ]

    outfile = f"{DOWNLOAD_DIR}/{site['name']}_{year}_{month}_ssrd.nc"

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Downloading {site['name']} {year}-{month}")

    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": VARIABLE,
            "year": year,
            "month": month,
            "day": get_days(year, month),
            "time": TIMES,
            "area": area,
            "format": "netcdf"
        },
        outfile
    )

    return outfile

# -----------------------------------------------------------------------------
# PROCESSING
# -----------------------------------------------------------------------------

def process_to_45km_9km_grid(nc_path, site):
    ds = open_era5_dataset(nc_path)

    if "time" not in ds.coords and "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    lat = site["lat"]
    lon = site["lon"]

    lat_km_per_deg = 111.0
    lon_km_per_deg = 111.0 * np.cos(np.deg2rad(lat))

    new_lats = lat + GRID_STEPS * GRID_SPACING_KM / lat_km_per_deg
    new_lons = lon + GRID_STEPS * GRID_SPACING_KM / lon_km_per_deg

    ds = ds.interp(latitude=new_lats, longitude=new_lons, method="linear")

    # UTC → IST
    t = pd.to_datetime(ds.time.values)
    t = t.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
    ds = ds.assign_coords(time=t)

    # Daytime filter
    hour = ds.time.dt.hour
    ds = ds.where((hour >= 5) & (hour <= 19), drop=True)

    return ds

# -----------------------------------------------------------------------------
# RUN PIPELINE
# -----------------------------------------------------------------------------

for site in sites:
    for year in ["2024"]:
        for month in ["04"]:
            nc = download_era5_land(site, year, month)
            try:
                ds_final = process_to_45km_9km_grid(nc, site)
                out_nc = nc.replace(".nc", "_45km_9km_IST_5am-7pm.nc")
                ds_final.to_netcdf(out_nc)
                print(f"Saved {out_nc}")
            except Exception as e:
                print(f"ERROR processing {nc}: {e}")

print("NetCDF processing done.")

# -----------------------------------------------------------------------------
# CSV EXPORT
# -----------------------------------------------------------------------------

for site in sites:
    files = sorted(glob.glob(
        f"{DOWNLOAD_DIR}/{site['name']}_*_45km_9km_IST_5am-7pm.nc"
    ))

    if not files:
        continue

    dfs = []
    for f in files:
        ds = xr.open_dataset(f)
        df = ds["ssrd"].stack(point=("latitude", "longitude")).to_dataframe().reset_index()
        df["ssrd_Wm2"] = df["ssrd"] / 3600.0
        dfs.append(df)

    final_df = pd.concat(dfs).sort_values("time")
    csv_path = f"{DOWNLOAD_DIR}/{site['name']}_ssrd_45km_9km_IST_all_months.csv"
    final_df.to_csv(csv_path, index=False)

    print(f"Saved CSV: {csv_path} ({len(final_df):,} rows)")

print("All done.")
