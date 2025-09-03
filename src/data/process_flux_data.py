# src/data/process_flux_data.py
import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
import argparse

POSSIBLE_VAR_NAMES = ["xrsb_flux", "xrsb", "flux"]

def find_flux_variable(dataset):
    for name in POSSIBLE_VAR_NAMES:
        if name in dataset.variables: return name
    return None

def process_single_netcdf(file_path):
    try:
        with xr.open_dataset(file_path) as ds:
            var_name = find_flux_variable(ds)
            if var_name is None:
                print(f"[WARN] No valid flux variable in {os.path.basename(file_path)}. Skipping.")
                return None
            df = ds[var_name].to_dataframe().rename(columns={var_name: "flux"})
            return df[~df.index.duplicated(keep='first')]
    except Exception as e:
        print(f"[ERROR] Failed to process {os.path.basename(file_path)}: {e}")
        return None

def main(root_dir):
    RAW_DIR = os.path.join(root_dir, "data/raw/goes_xrs/")
    OUTPUT_PATH = os.path.join(root_dir, "data/interim/goes_xrs_flux_log.parquet")

    all_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.nc")))
    if not all_files:
        print(f"❌ No .nc files found in '{RAW_DIR}'.")
        return

    print(f"📦 Found {len(all_files)} files. Processing...")
    successful_dfs = [df for f in tqdm(all_files) if (df := process_single_netcdf(f)) is not None and not df.empty]

    if not successful_dfs:
        print("❌ No data could be processed.")
        return

    print("🧩 Merging data...")
    full_df = pd.concat(successful_dfs).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep='first')]

    print("📊 Resampling and interpolating...")
    # FIX: Add back interpolation to handle small gaps in data
    flux_df = full_df.resample("1min").mean().interpolate(method='time', limit_direction='both', limit=5)
    
    flux_df["flux_log"] = np.log10(flux_df["flux"].clip(lower=1e-10))
    final_df = flux_df[["flux_log"]].dropna()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_parquet(OUTPUT_PATH)
    print(f"[✓] Saved cleaned flux log to: {OUTPUT_PATH} ({len(final_df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw GOES NetCDF data.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the project.")
    args = parser.parse_args()
    main(args.root_dir)