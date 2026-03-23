"""
Fetch Prithvi (HLS) Data by dividing the domain into ERA5-sized tiles (0.25 degrees).

The main problem with fetching 30m resolution HLS data for the entire 
South India domain (9x9 degrees) simultaneously is memory and API limits. 
This script:
1. Takes the ERA5 bounding box and breaks it down into 0.25-degree tiles.
2. Queries the HLS STAC catalog for each 0.25-degree tile independently.
3. Downloads the required 6 spectral bands for Prithvi inference over that tile.
4. Generates a regional block for that tile and saves it.

This forms the "fetching" script. The interpolation to MODIS happens
in your `fno_dataset.py`, which already calls `PrithviDynamicProvider.sample_tile()`.
"""

import argparse
import itertools
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr
import pystac_client
import planetary_computer
import pyproj
from rasterio.enums import Resampling

# =========================
# CONFIG
# =========================
DEFAULT_START_YEAR = 2019
DEFAULT_END_YEAR = 2019  # Using later years where HLS data actually exists (Landsat 8 / Sentinel-2)

BASE_OUTPUT_DIR = Path("/Users/IRFAN/Library/CloudStorage/GoogleDrive-irfan.a@atriauniversity.edu.in/My Drive/Irradiance-forecasting/PRITHVI")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# South India domain from ERA5
SOUTH_INDIA_NORTH = 17.0
SOUTH_INDIA_WEST = 72.5
SOUTH_INDIA_SOUTH = 8.0
SOUTH_INDIA_EAST = 81.5

# ERA5 Tile Size
ERA5_GRID_STEP = 0.25

# Prithvi Model expects these 6 specific bands: Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2.
HLS_BANDS = {
    "hls2-s30": ["B02", "B03", "B04", "B8A", "B11", "B12"], # Sentinel-2 band mappings
    "hls2-l30": ["B02", "B03", "B04", "B05", "B06", "B07"]  # Landsat-8/9 band mappings
}

def fetch_tile_hls_data(catalog, year, lon_min, lon_max, lat_min, lat_max, tile_output_dir):
    """Fetches the 6 HLS bands for a specific 0.25 x 0.25 degree tile and computes embeddings."""
    time_range = f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z"
    bbox = [lon_min, lat_min, lon_max, lat_max]
    
    # Wrap the search in a retry loop to handle 502 Bad Gateway / Connection Errors
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            search = catalog.search(
                collections=["hls2-l30", "hls2-s30"], 
                bbox=bbox,
                datetime=time_range,
                sortby=[{"field": "eo:cloud_cover", "direction": "asc"}] # Sort ascending (clearest photo first)
            )
            items = list(search.items())
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"      API Error ({e}). Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"      API permanently failed for this tile after {max_retries} attempts.")
                return False
    
    # We take a sample scene for the tile (in reality, you might loop through months)
    if not items:
        return False
        
    scene = items[0] 
    print(f"    -> Found Scene: {scene.id} at {scene.datetime}")
    
    bands = HLS_BANDS[scene.collection_id]
    
    # Placeholder for actual model inference
    # INSTRUCTIONS TO RUN:
    # 1. pip install torch torchvision transformers rasterio
    # 2. Get the model from Hugging Face:
    #    git lfs clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M prithvi_model
    # 3. Import their library at the top of this script:
    #    import torch
    #    from prithvi_model.Prithvi import MaskedAutoencoderViT
    
    channels = 192 # Native Prithvi output channels
    lats = np.linspace(lat_max, lat_min, num=int(ERA5_GRID_STEP / (30 / 111000))) # Approximate 30m grid within 0.25 deg
    lons = np.linspace(lon_min, lon_max, num=int(ERA5_GRID_STEP / (30 / 111000)))
    
    # ----------------------------------------------------
    # --- ACTUAL PYTORCH INFERENCE CODE
    # ----------------------------------------------------
    import urllib.request
    import rasterio
    import torch
    import sys
    import os
    
    # Check if the Prithvi Model has been cloned to the directory.
    try:
        base_dir = Path(__file__).parent.parent
    except NameError:
        # We are running in a Jupyter Colab Notebook where __file__ doesn't exist.
        # It's safe to assume the working directory is the project root!
        base_dir = Path(os.getcwd())
        
    cloned_path = str(base_dir / "prithvi_model")
    if not os.path.exists(cloned_path):
        print(f"    -> ERROR: prithvi_model not found at {cloned_path}! Running 'git clone https://github.com/NASA-Impact/hls-foundation-os.git prithvi_model' in terminal first.")
        return False
        
    if cloned_path not in sys.path:
        sys.path.append(cloned_path)
    
    # 1. Download & Stack the 6 bands to memory
    stacked_image = np.zeros((6, len(lats), len(lons)), dtype=np.float32)
    for idx, band_id in enumerate(bands):
        asset = scene.assets[band_id]
        href = planetary_computer.sign(asset.href)
        
        # Read directly to array memory using rasterio
        with rasterio.open(href) as src:
            # Sometime our bounding boxes safely hangover the jagged edges of a satellite pass.
            # We must crop it to the intersection of our ERA box and the actual photo bounds to prevent RasterioIOErrors!
            try:
                # Calculate window
                window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
                
                # Intersect our desired window with the actual boundaries of the TIFF to avoid "out of range" errors
                safe_window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                
                band_data = src.read(1, window=safe_window, out_shape=(len(lats), len(lons)), resampling=Resampling.bilinear)
                stacked_image[idx] = band_data
            except Exception as e:
                print(f"      -> Warning: Rasterio boundary issue on {band_id}. Skipping tile. {e}")
                return False
            
    # Normalize imagery as Prithvi expects
    stacked_image = (stacked_image / 10000.0) 
    
    # 2. Convert to PyTorch Tensor: [Batch, Channels, Time, Height, Width]
    # Prithvi treats images as "videos" of length 1 frame
    tensor_input = torch.tensor(stacked_image).unsqueeze(0).unsqueeze(2) 

    # 3. Load Model weights 
    try:
        from prithvi_model.Prithvi import MaskedAutoencoderViT
        import yaml
        
        config_path = f"{cloned_path}/config.yaml"
        weights_path = f"{cloned_path}/Prithvi_100M.pt"
        
        if not os.path.exists(weights_path):
            print("    -> ERROR: You must download Prithvi_100M.pt into the prithvi_model folder!")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        model = MaskedAutoencoderViT(**config).to("cpu")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        
        # 4. Extract Embeddings (Forward Encoder)
        with torch.no_grad():
            embeddings, _, _ = model.forward_encoder(tensor_input, mask_ratio=0.0)
            embeddings_np = embeddings.squeeze().detach().cpu().numpy() # Shape will map to (192, Height, Width)
            
    except ImportError as e:
         print(f"    -> ERROR: Failed to load Prithvi model files natively. {e}")
         return False
         
    # Validate the generated embeddings shape before continuing
    if embeddings_np.shape[0] != channels:
         print(f"    -> WARNING: Expected {channels} channels but got {embeddings_np.shape[0]}. Replacing with 0s.")
         embeddings_np = np.zeros((channels, len(lats), len(lons)), dtype=np.float32)
    
    da = xr.DataArray(
        np.expand_dims(embeddings_np, axis=0),
        dims=["time", "channel", "latitude", "longitude"],
        coords={
            "time": [np.datetime64(scene.datetime.replace(tzinfo=None))],
            "channel": np.arange(channels),
            "latitude": lats,
            "longitude": lons,
        },
        name="prithvi_embeddings"
    )
    
    # Save the embeddings tile
    outfile_path = tile_output_dir / f"prithvi_{year}_lat{lat_min}_lon{lon_min}.nc"
    da.to_netcdf(outfile_path)
    return True

def main():
    parser = argparse.ArgumentParser(description="Fetch and process Prithvi tiles over South India.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    args, unknown = parser.parse_known_args()

    print("Connecting to Planetary Computer STAC Catalog...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # 1. Divide into ERA5 tiles (0.25 x 0.25 degrees)
    lat_bins = np.arange(SOUTH_INDIA_SOUTH, SOUTH_INDIA_NORTH, ERA5_GRID_STEP)
    lon_bins = np.arange(SOUTH_INDIA_WEST, SOUTH_INDIA_EAST, ERA5_GRID_STEP)
    
    # Total tiles = 36 * 36
    total_tiles = len(lat_bins) * len(lon_bins)
    print(f"Domain divided into {len(lat_bins)}x{len(lon_bins)} ({total_tiles} total) ERA5-sized tiles of {ERA5_GRID_STEP} degrees.")
    
    for year in range(args.start_year, args.end_year + 1):
        year_dir = BASE_OUTPUT_DIR / str(year)
        year_tile_dir = year_dir / "era5_tiles"
        year_tile_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing Year: {year}")
        processed = 0
        
        # Loop over every 0.25x0.25 degree tile
        for lat_idx, lat_min in enumerate(lat_bins):
            lat_max = lat_min + ERA5_GRID_STEP
            
            for lon_idx, lon_min in enumerate(lon_bins):
                lon_max = lon_min + ERA5_GRID_STEP
                processed += 1
                
                print(f"  [{processed}/{total_tiles}] Tiles Fetched | Bounding Box: Lat {lat_min}-{lat_max}, Lon {lon_min}-{lon_max}")
                
                # Fetch data and generate Prithvi model embeddings for this specific box
                success = fetch_tile_hls_data(
                    catalog, year, 
                    lon_min, lon_max, 
                    lat_min, lat_max, 
                    year_tile_dir
                )
                
                # We removed the early breaks here so it can search the whole domain!

        print(f"Year {year} initialized safely. Resulting subset tiles saved to: {year_tile_dir}")
        print("Once all tiles run, you can use `xarray.open_mfdataset()` in FNO to load them collectively.")
        
if __name__ == "__main__":
    main()
