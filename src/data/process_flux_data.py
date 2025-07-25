#src\data\process_flux_data.py

import os
import pathlib as Path
import pandas as pd
import numpy as np
from sunpy.timeseries import TimeSeries

def process_flux_directory(raw_dir, out_path):
    ts_list = []
    # Ensure raw_dir is a Path object for cleaner joins
    raw_dir_path = Path(raw_dir)

    # Use Path.iterdir() for more Pythonic iteration
    for file_path in sorted(raw_dir_path.iterdir()):
        if file_path.suffix == ".nc": # Use .suffix for robustness
            try:
                ts = TimeSeries(file_path)
                df = ts.to_dataframe()
                
                # Dynamic column renaming: check if 'xrsb' exists, otherwise look for other flux columns
                if 'xrsb' in df.columns:
                    df = df.rename(columns={"xrsb": "flux"})
                elif 'long' in df.columns: # For older GOES data, it might be named 'long'
                    df = df.rename(columns={"long": "flux"})
                else:
                    # Log an error if expected flux column is not found
                    print(f"[ERROR] Expected flux column (xrsb or long) not found in {file_path}. Available columns: {df.columns}. Skipping.")
                    continue # Skip to next file

                df = df[["flux"]].dropna() # Ensure only flux column is processed here
                df["flux_log"] = np.log10(df["flux"] + 1e-9)
                ts_list.append(df)
            except Exception as e:
                print(f"[WARN] Skipped {file_path.name} due to error: {e}")

    if not ts_list:
        print(f"[ERROR] No valid GOES XRS dataframes processed from {raw_dir}. Check raw data files.")
        return # Exit if no data was processed

    full_df = pd.concat(ts_list)
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    # Use astype(datetime) to ensure index is datetime64[ns] before asfreq
    full_df.index = pd.to_datetime(full_df.index)
    full_df = full_df.asfreq("1min").interpolate("time")
    
    # Drop NaNs that might appear at the start/end due to interpolation or gaps
    full_df = full_df.dropna() 

    # Ensure output directory exists before saving
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(out_path)

    print(f"[✓] Saved cleaned flux data to: {out_path}")

if __name__ == "__main__":
    process_flux_directory("data/raw/goes_xrs/", "data/interim/goes_xrs_flux_log.parquet")