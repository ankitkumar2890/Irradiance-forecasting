"""
MODIS COT Downloader — Fully Robust Version
Handles curl 56, connection reset, partial files, extraction retry.
Never crashes mid-run.
"""

import os
import subprocess
import time
import datetime
import math
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pyhdf.SD import SD, SDC
import requests

# =========================
# CONFIG
# =========================

LAADS_TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImlyZmFuNyIsImV4cCI6MTc3NjE0NjQ1NywiaWF0IjoxNzcwOTYyNDU3LCJpc3MiOiJodHRwczovL3Vycy5lYXJ0aGRhdGEubmFzYS5nb3YiLCJpZGVudGl0eV9wcm92aWRlciI6ImVkbF9vcHMiLCJhY3IiOiJlZGwiLCJhc3N1cmFuY2VfbGV2ZWwiOjN9.0pYUSRC458lq9mJXgmurU_aqgkPi0h4FAb2ATls6dRXxEHb4u8MLS4RyZLSGYfX1JLrysOsAwPDtV02CCaD29mY8NVXeecxP1ts4eB1RV1C6zxvwrKbnLU5XllCzhXrTZ4-2LyNVa5-MnsS8d1g5d3RxiuEXRIeFz__Brja5VMXollV-fscz8f_L7_xcurWmfSbFn_woQfIaWbnBRkiL3arsV0ZY1egngbCnT0BhOPlFw_ixzYbSXZhIsvBYNjURWRWplk_-oOdnD7W_vxvuxidLV0Lu11nV9A3jgu3HoE0GbUb2AWdBx-mjJR_979M92cnFyyG90Bo6L7TqZ6lq9Q"

START_DATE = "2005-01-01"
END_DATE   = "2005-01-01"

SITE_LAT = 9.14
SITE_LON = 77.92
BUFFER_DEG = 1.0   # Match ERA5

MIN_LON = SITE_LON - BUFFER_DEG
MAX_LON = SITE_LON + BUFFER_DEG
MIN_LAT = SITE_LAT - BUFFER_DEG
MAX_LAT = SITE_LAT + BUFFER_DEG

OUTPUT_DIR = "data_store/MOD06L2_COT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAILED_DOWNLOADS_LOG = os.path.join(OUTPUT_DIR, "failed_downloads.txt")

SKIP_DOWNLOAD = True

# =========================
# DATE RANGE
# =========================

def daterange(start, end):
    d0 = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    while d0 <= d1:
        yield d0
        d0 += datetime.timedelta(days=1)

# =========================
# LIST FILES
# =========================

def list_files_for_date(date):
    year = date.year
    doy = date.timetuple().tm_yday

    url = f"https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details?products=MOD06_L2&temporalRanges={year}-{doy:03d}"
    headers = {"Authorization": f"Bearer {LAADS_TOKEN}"}

    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()

        files = []
        for item in data.get("content", []):
            name = item["name"]
            parts = name.split('.')
            date_part = parts[1]
            year_str = date_part[1:5]
            doy_str  = date_part[5:8]

            dl_url = (
                f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/"
                f"MOD06_L2/{year_str}/{doy_str}/{name}"
            )

            files.append({"name": name, "url": dl_url})
        return files

    except Exception as e:
        print("API error:", e)
        return []

# =========================
# ROBUST DOWNLOAD
# =========================

def download_file(url, output_path, max_retries=6):

    cookie_file = "/tmp/.laads_cookies"

    cmd = [
        "curl",
        "-L",
        "-c", cookie_file,
        "-b", cookie_file,
        "-H", f"Authorization: Bearer {LAADS_TOKEN}",
        "--retry", "5",
        "--retry-delay", "5",
        "--retry-connrefused",
        "--connect-timeout", "60",
        "--max-time", "600",
        "-o", output_path,
        url
    ]

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}...")
            subprocess.run(cmd, check=True)

            # Validate file
            if not os.path.exists(output_path):
                raise RuntimeError("No file created")

            size = os.path.getsize(output_path)
            if size < 500_000:
                raise RuntimeError(f"File too small ({size} bytes)")

            return True

        except Exception as e:
            print(f"  Download error: {e}")

            if os.path.exists(output_path):
                os.remove(output_path)

            wait = 5 * attempt
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    return False

# =========================
# COT EXTRACTION
# =========================

def extract_cot_dataset(hdf):
    for name in ["Cloud_Optical_Thickness", "Cloud_Optical_Thickness_16"]:
        try:
            return hdf.select(name).get()
        except:
            pass
    raise RuntimeError("COT not found")

def extract_cot_to_geotiff(hdf_path, out_tif, max_retries=3):

    from pyresample import geometry, kd_tree
    from scipy.ndimage import zoom

    for attempt in range(1, max_retries + 1):
        try:
            hdf = SD(hdf_path, SDC.READ)

            cot = extract_cot_dataset(hdf)
            lat = hdf.select("Latitude").get()
            lon = hdf.select("Longitude").get()

            if lat.shape != cot.shape:
                zoom_y = cot.shape[0] / lat.shape[0]
                zoom_x = cot.shape[1] / lat.shape[1]
                lat = zoom(lat, (zoom_y, zoom_x), order=1)
                lon = zoom(lon, (zoom_y, zoom_x), order=1)

            cot = cot.astype(np.float32)
            cot[cot < 0] = np.nan
            cot[cot > 150] = np.nan

            mask = (
                (lat >= MIN_LAT) & (lat <= MAX_LAT) &
                (lon >= MIN_LON) & (lon <= MAX_LON)
            )

            if not np.any(mask):
                return False

            swath_def = geometry.SwathDefinition(lons=lon, lats=lat)

            KM_PER_DEG_LAT = 111.0
            KM_PER_DEG_LON = 111.0 * math.cos(math.radians(SITE_LAT))

            pixel_lat = 1.0 / KM_PER_DEG_LAT
            pixel_lon = 1.0 / KM_PER_DEG_LON

            grid_h = int((MAX_LAT - MIN_LAT) / pixel_lat)
            grid_w = int((MAX_LON - MIN_LON) / pixel_lon)

            area_def = geometry.AreaDefinition(
                'target_area',
                '2deg domain 1km grid',
                'latlon',
                {'proj': 'latlong'},
                grid_w,
                grid_h,
                (MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)
            )

            cot_grid = kd_tree.resample_nearest(
                swath_def,
                cot,
                area_def,
                radius_of_influence=5000,
                fill_value=np.nan
            )

            transform = from_bounds(
                MIN_LON, MIN_LAT, MAX_LON, MAX_LAT,
                grid_w, grid_h
            )

            with rasterio.open(
                out_tif, "w",
                driver="GTiff",
                height=grid_h,
                width=grid_w,
                count=1,
                dtype="float32",
                crs=CRS.from_epsg(4326),
                transform=transform,
                nodata=np.nan
            ) as dst:
                dst.write(cot_grid, 1)

            return True

        except Exception as e:
            print(f"  Extraction failed (attempt {attempt}): {e}")
            time.sleep(3 * attempt)

    return False

# =========================
# MAIN
# =========================

def main():

    print("="*60)
    print("MODIS COT Downloader — Fully Robust")
    print("="*60)

    failed_files = []

    if not SKIP_DOWNLOAD:
        for date in daterange(START_DATE, END_DATE):
            files = list_files_for_date(date)

            for file_info in files:
                filename = file_info["name"]
                output_path = os.path.join(OUTPUT_DIR, filename)

                if os.path.exists(output_path):
                    continue

                print("Downloading:", filename)

                success = download_file(file_info["url"], output_path)

                if not success:
                    print("  ❌ Failed permanently:", filename)
                    failed_files.append(filename)

                time.sleep(2)

    print("\nProcessing HDF → GeoTIFF...\n")

    for hdf_file in sorted(os.listdir(OUTPUT_DIR)):
        if not hdf_file.endswith(".hdf"):
            continue

        hdf_path = os.path.join(OUTPUT_DIR, hdf_file)

        parts = hdf_file.split('.')
        date_part = parts[1]
        time_part = parts[2]

        year = int(date_part[1:5])
        doy  = int(date_part[5:8])
        hour = time_part[:2]
        minute = time_part[2:4]

        date = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
        timestamp = f"{date.strftime('%Y-%m-%d')}_{hour}-{minute}"

        tif_path = os.path.join(OUTPUT_DIR, f"{timestamp}_COT.tif")

        success = extract_cot_to_geotiff(hdf_path, tif_path)

        if success:
            print("✓", os.path.basename(tif_path))
        else:
            print("⚠ Extraction failed:", hdf_file)

    if failed_files:
        with open(FAILED_DOWNLOADS_LOG, "w") as f:
            for name in failed_files:
                f.write(name + "\n")

        print("\nSome files failed. See:", FAILED_DOWNLOADS_LOG)

    print("\nDone.")

if __name__ == "__main__":
    main()
