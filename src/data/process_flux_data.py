import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_DIR = "data/raw/goes_xrs/"
OUTPUT_PATH = "data/interim/goes_xrs_flux_log.parquet"
TEMP_DIR = "data/interim/tmp_chunks/"
MAX_WORKERS = 3  # Custom for i5 dual-core (4 threads total)

BATCH_SIZE = 25  # Keep RAM use low

os.makedirs(TEMP_DIR, exist_ok=True)


def process_single_netcdf(file_path):
    """
    Processes a single NetCDF file, extracts 'xrsb' flux, and returns a DataFrame.
    Returns None if 'xrsb' is not found, or if an error occurs.
    """
    try:
        ds = xr.open_dataset(file_path)
        # --- FIX START ---
        # Change 'xrsb' to 'xrsb_flux' as found in the NetCDF inspection
        if "xrsb_flux" not in ds:
            # print(f"DEBUG: 'xrsb_flux' not found in {file_path}. Skipping.")
            return None
        flux = ds["xrsb_flux"].to_dataframe().dropna() # Access the correct variable
        # --- FIX END ---
        
        if flux.empty:
            # print(f"DEBUG: No valid flux data after dropping NaNs for {file_path}. Skipping.")
            return None
        
        # --- FIX START ---
        # Rename the column from 'xrsb_flux' to 'flux'
        flux = flux.rename(columns={"xrsb_flux": "flux"}) 
        # --- FIX END ---
        
        flux = flux[~flux.index.duplicated()] # Remove duplicate indices if any
        
        if flux.empty:
            # print(f"DEBUG: Flux dataframe became empty after deduplication for {file_path}. Skipping.")
            return None
            
        return flux
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_batch(batch_files, batch_idx):
    """
    Processes a list of NetCDF files, concatenates their DataFrames,
    and saves them as a parquet chunk.
    """
    results = []
    for file_path in batch_files:
        result = process_single_netcdf(file_path)
        if result is not None and not result.empty:
            results.append(result)

    if results:
        chunk_df = pd.concat(results)
        chunk_df = chunk_df[~chunk_df.index.duplicated()]
        chunk_path = os.path.join(TEMP_DIR, f"chunk_{batch_idx}.parquet")
        chunk_df.to_parquet(chunk_path)


def parallel_batches(all_files):
    """
    Submits batches of files for parallel processing and waits for completion.
    """
    total_batches = (len(all_files) + BATCH_SIZE - 1) // BATCH_SIZE 
    
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in range(total_batches):
            batch = all_files[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            if batch:
                futures.append(executor.submit(process_batch, batch, i))

        print("Processing batches in parallel...")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Batching and Processing"):
            exception = future.exception()
            if exception:
                print(f"Error in a batch processing task: {exception}")


def merge_and_transform_chunks():
    """
    Merges all saved parquet chunks, resamples, transforms, and saves the final dataset.
    """
    print("🧩 Merging all saved chunks...")

    all_chunks = sorted(glob.glob(os.path.join(TEMP_DIR, "*.parquet")))
    
    if not all_chunks:
        print("⚠️ Warning: No parquet chunks found to merge. This might indicate an issue with file processing.")
        # This print will now only happen if no data was valid across ALL 11052 files.
        return 

    all_df = [pd.read_parquet(chunk) for chunk in all_chunks]

    flux_df = pd.concat(all_df)
    flux_df = flux_df[~flux_df.index.duplicated()]
    flux_df = flux_df.sort_index()

    print("📉 Resampling to 1-minute intervals and log-transforming...")

    flux_df = flux_df.resample("1min").mean()
    flux_df["flux_log"] = np.log10(flux_df["flux"].clip(lower=1e-9)) 
    flux_df = flux_df[["flux_log"]].dropna()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    flux_df.to_parquet(OUTPUT_PATH)
    print(f"✅ Final cleaned dataset saved to: {OUTPUT_PATH} ({len(flux_df)} rows)")


def main():
    print("🔍 Scanning GOES NetCDF files...")
    all_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.nc")))

    if not all_files:
        print("❌ No NetCDF files found in the raw data directory. Please check if GOES data was unrarred correctly.")
        return

    print(f"📦 Found {len(all_files)} NetCDF files. Starting processing...")
    parallel_batches(all_files)
    
    merge_and_transform_chunks()


if __name__ == "__main__":
    main()