# src/data/process_flux_data.py

import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm

# --- Configuration ---
RAW_DIR = "data/raw/goes_xrs/"
OUTPUT_PATH = "data/interim/goes_xrs_flux_log.parquet"
# Best Practice: Define a list of possible variable names to check for.
# The script will use the first one it finds.
POSSIBLE_VAR_NAMES = ["xrsb_flux", "xrsb", "flux"] 

def find_flux_variable(dataset):
    """Checks for a list of possible flux variable names in the dataset."""
    for name in POSSIBLE_VAR_NAMES:
        if name in dataset.variables:
            return name
    return None

def process_single_netcdf(file_path):
    """Safely processes a single NetCDF file by finding the correct flux variable."""
    try:
        with xr.open_dataset(file_path) as ds:
            var_name = find_flux_variable(ds)
            
            if var_name is None:
                print(f"[WARN] No valid flux variable found in {os.path.basename(file_path)}. Skipping.")
                return None
            
            df = ds[var_name].to_dataframe().rename(columns={var_name: "flux"})
            df = df[~df.index.duplicated(keep='first')]
            return df
            
    except Exception as e:
        print(f"[ERROR] Failed to process {os.path.basename(file_path)}: {e}")
        return None

def main():
    """Main function to process all NetCDF files in the raw directory."""
    all_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.nc")))
    
    if not all_files:
        print(f"❌ No NetCDF (.nc) files found in '{RAW_DIR}'.")
        return

    print(f"📦 Found {len(all_files)} NetCDF files. Starting processing...")
    
    successful_dfs = []
    for file_path in tqdm(all_files, desc="Processing files"):
        df = process_single_netcdf(file_path)
        if df is not None and not df.empty:
            successful_dfs.append(df)
            
    if not successful_dfs:
        print("❌ No data could be processed. No output file was generated.")
        return

    print("🧩 Merging all successfully processed data...")
    full_df = pd.concat(successful_dfs).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep='first')]

    print("📉 Resampling to 1-minute intervals and log-transforming...")
    flux_df = full_df.resample("1min").mean().interpolate(method='time', limit_direction='both', limit=5)
    flux_df["flux_log"] = np.log10(flux_df["flux"].clip(lower=1e-10))
    
    final_df = flux_df[["flux_log"]].dropna()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_parquet(OUTPUT_PATH)
    
    print(f"\n[✓] Final cleaned dataset saved to: {OUTPUT_PATH} ({len(final_df)} rows)")

if __name__ == "__main__":
    main()