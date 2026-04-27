import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path("phase3_direct_ghi").resolve()))
from config import MULTI_STATION_DOWNLOADS_DIR, STATIONS, YEARS
from fetch_data import average_half_hour_to_top_of_hour, clearsky_time_grid

# Ensure we import the function correctly
# wait, fetch_data is 01_fetch_data.py it might have syntax issues with import
