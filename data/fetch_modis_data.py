"""
MODIS COT Downloader — Optimized Production Version

Features:
✔ Robust curl download
✔ UTC pre-filter (reduces useless swaths)
✔ IST daylight filter (06–19 IST)
✔ Spatial overlap check
✔ File integrity validation
✔ Safe extraction retry
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
END_DATE   = "2005-12-31"

SITE_LAT = 9.14
SITE_LON = 77.92
BUFFER_DEG = 1.0  # ±1° → ~222 km × 222 km (matches ERA5)

MIN_LON = SITE_LON - BUFFER_DEG
MAX_LON = SITE_LON + BUFFER_DEG
MIN_LAT = SITE_LAT - BUFFER_DEG
MAX_LAT = SITE_LAT + BUFFER_DEG

OUTPUT_DIR = "data_store/MOD06L2_COT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SKIP_DOWNLOAD = False

# =========================
# TIME FILTERS
# =========================

# 1️⃣ UTC prefilter (reduce useless swaths)
UTC_MIN_HOUR = 3
UTC_MAX_HOUR = 11

# 2️⃣ IST daylight filter
IST_START_HOUR = 6
IST_END_HOUR   = 19

def is_valid_utc(hour):
    return UTC_MIN_HOUR <= hour <= UTC_MAX_HOUR

def is_within_ist_window(year, doy, hour, minute):
    date_utc = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
    dt_utc = date_utc.replace(hour=hour, minute=minute)
    dt_ist = dt_utc + datetime.timedelta(hours=5, minutes=30)
    return IST_START_HOUR <= dt_ist.hour <= IST_END_HOUR

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
# LIST FILES (TEMPORAL ONLY)
# =========================

def list_files_for_date(date):
    year = date.year
    doy = date.timetuple().tm_yday

    bbox = f"[BBOX]W{MIN_LON} N{MAX_LAT} E{MAX_LON} S{MIN_LAT}"

    url = (
        "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content/details?"
        f"products=MOD06_L2"
        f"&temporalRanges={year}-{doy:03d}"
        f"&regions={bbox}"
    )

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
            time_part = parts[2]

            year_i = int(date_part[1:5])
            doy_i  = int(date_part[5:8])
            hour   = int(time_part[:2])
            minute = int(time_part[2:4])

            # UTC prefilter
            if not is_valid_utc(hour):
                continue

            # IST daylight filter
            if not is_within_ist_window(year_i, doy_i, hour, minute):
                continue

            dl_url = (
                f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/"
                f"MOD06_L2/{year_i}/{doy_i:03d}/{name}"
            )

            files.append({"name": name, "url": dl_url})

        print(f"  {len(files)} candidate swaths after time filtering")
        return files

    except Exception as e:
        print("API error:", e)
        return []

# =========================
# ROBUST DOWNLOAD
# =========================

def download_file(url, output_path, max_retries=5):

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
        "--max-time", "900",
        "-o", output_path,
        url
    ]

    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(cmd, check=True)

            if not os.path.exists(output_path):
                raise RuntimeError("No file created")

            size = os.path.getsize(output_path)

            # Stronger validation (real MOD06 > 10MB)
            if size < 5_000_000:
                raise RuntimeError(f"File too small ({size} bytes)")

            # Detect HTML error
            with open(output_path, "rb") as f:
                head = f.read(200)
                if b"DOCTYPE" in head or b"html" in head:
                    raise RuntimeError("Downloaded HTML instead of HDF")

            return True

        except Exception as e:
            print(f"  Download error: {e}")

            if os.path.exists(output_path):
                os.remove(output_path)

            time.sleep(5 * attempt)

    return False

# =========================
# COT EXTRACTION
# =========================

def extract_cot_to_geotiff(hdf_path, out_tif):

    from pyresample import geometry, kd_tree
    from scipy.ndimage import zoom

    try:
        hdf = SD(hdf_path, SDC.READ)

        for name in ["Cloud_Optical_Thickness", "Cloud_Optical_Thickness_16"]:
            try:
                sds = hdf.select(name)
                cot = sds.get().astype(np.float32)

                # Apply scale factor (MOD06_L2 COT: scale_factor=0.01)
                attrs = sds.attributes()
                scale = attrs.get("scale_factor", 1.0)
                offset = attrs.get("add_offset", 0.0)
                fill = attrs.get("_FillValue", -9999)

                cot[cot == fill] = np.nan
                cot = (cot - offset) * scale
                break
            except:
                continue
        else:
            return False

        lat = hdf.select("Latitude").get()
        lon = hdf.select("Longitude").get()

        if lat.shape != cot.shape:
            zoom_y = cot.shape[0] / lat.shape[0]
            zoom_x = cot.shape[1] / lat.shape[1]
            lat = zoom(lat, (zoom_y, zoom_x), order=1)
            lon = zoom(lon, (zoom_y, zoom_x), order=1)

        # Filter physically invalid COT (after scaling: valid range 0–150)
        cot[(cot < 0) | (cot > 150)] = np.nan

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
            out_tif,
            "w",
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
        print(f"  ✗ Extraction failed for {hdf_path}: {e}")
        return False

# =========================
# MAIN
# =========================

def main():

    print("="*60)
    print("MODIS COT Downloader — Optimized Production Version")
    print("="*60)

    for date in daterange(START_DATE, END_DATE):

        files = list_files_for_date(date)

        for file_info in files:

            filename = file_info["name"]
            output_path = os.path.join(OUTPUT_DIR, filename)

            if not SKIP_DOWNLOAD and not os.path.exists(output_path):
                print("Downloading:", filename)
                if not download_file(file_info["url"], output_path):
                    continue

            parts = filename.split('.')
            date_part = parts[1]
            time_part = parts[2]

            year = int(date_part[1:5])
            doy  = int(date_part[5:8])
            hour = int(time_part[:2])
            minute = int(time_part[2:4])

            timestamp = (
                datetime.datetime(year, 1, 1) +
                datetime.timedelta(doy - 1)
            ).strftime("%Y-%m-%d")

            tif_path = os.path.join(
                OUTPUT_DIR,
                f"{timestamp}_{hour:02d}-{minute:02d}_COT.tif"
            )

            if extract_cot_to_geotiff(output_path, tif_path):
                print("✓", os.path.basename(tif_path))

    print("\nDone.")

if __name__ == "__main__":
    main()
