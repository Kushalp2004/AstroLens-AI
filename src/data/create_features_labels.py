#src\data\create_features_labels.py

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime, timedelta # Import timedelta

# --- SET_PATHS ---
INTERIM_DIR = Path("data/interim")
RAW_FLARE_CATALOGUE_DIR = Path("data/raw/flare_catalogue") # Renamed for clarity
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Load flux data ---
flux_df = pd.read_parquet(INTERIM_DIR / "goes_xrs_flux_log.parquet")
print(f"[INFO] Loaded flux data with {len(flux_df)} entries from {INTERIM_DIR / 'goes_xrs_flux_log.parquet'}")


# --- Load NOAA flare catalogue ---
def load_flare_catalogue(raw_dir: Path):
    all_flares = []
    # Explicitly look for common filenames
    flare_catalogue_files = [f for f in os.listdir(raw_dir) if f.lower() in ['goes-xray-flares.txt', 'goes_xray_flares_1975-present.txt']]

    if not flare_catalogue_files:
        print(f"[ERROR] No expected flare catalogue file found in {raw_dir}. Cannot proceed with labeling.")
        raise FileNotFoundError(f"Missing GOES flare catalogue in {raw_dir}")

    for file_name in flare_catalogue_files:
        file_path = raw_dir / file_name
        try:
            # Column 4 is "class". Column 3 is "magnitude". Let's get both.
            # Names should match the header-less file format
            df = pd.read_csv(file_path, delim_whitespace=True, header=None, comment="#",
                             names=["date", "start_time_str", "peak_time_str", "end_time_str", "class", "magnitude_raw"])
            
            # Combine date and peak time for precise timestamp
            df["datetime"] = pd.to_datetime(df["date"] + " " + df["peak_time_str"], format="%Y%m%d %H%M") # Explicit format for robustness
            df = df[["datetime", "class"]].copy() # Keep only necessary columns
            df["class"] = df["class"].astype(str).str[0] # Take only the letter (C, M, X, B, A)

            # Filter for C, M, X class flares only for efficiency in labeling
            df = df[df["class"].isin(["C", "M", "X"])].copy() # We care about these for classification

            all_flares.append(df)
        except Exception as e:
            print(f"[WARN] Failed parsing {file_path.name}: {e}")

    if not all_flares:
        print(f"[ERROR] No valid flare entries parsed from catalogue files. Cannot proceed with labeling.")
        return pd.DataFrame(columns=["datetime", "class"]) # Return empty DataFrame
    
    flares = pd.concat(all_flares, ignore_index=True)
    flares = flares.drop_duplicates(subset=["datetime", "class"]) # Remove exact duplicates if any
    flares = flares.set_index("datetime").sort_index()
    print(f"[INFO] Loaded {len(flares)} flare events from catalogue.")
    return flares

flare_df = load_flare_catalogue(RAW_FLARE_CATALOGUE_DIR)

# --- Label assignment function ---
def assign_labels_to_flux(flux_index, flare_events_df, prediction_window_hours=24): # Renamed for clarity
    labels = []
    window = timedelta(hours=prediction_window_hours) # Define timedelta once

    # Ensure flare_events_df has a datetime index and 'class' column
    if not isinstance(flare_events_df.index, pd.DatetimeIndex):
        raise ValueError("flare_events_df must have a DatetimeIndex.")
    if 'class' not in flare_events_df.columns:
        raise ValueError("flare_events_df must have a 'class' column.")

    # Use a faster, vectorized approach if possible, or optimize iteration
    # For large datasets, direct iteration can be slow.
    # A more optimized approach involves rolling window or merge operations.
    # However, for an intermediate project, the loop is conceptually clear and fine for now.

    # Optimization: Create a boolean series indicating flare occurrences
    # This might still be slow for 12 years of 1-minute data
    # Let's consider a windowed approach or re-evaluate performance later if this is too slow.

    # For demonstration, keeping your iterative approach with refinements:
    for timestamp in flux_index:
        window_end = timestamp + window
        # Use .loc[start:end] on a sorted DatetimeIndex for efficient slicing
        # Ensure 'class' column is available and string type
        # .str[0] is good if it's 'C1.0' -> 'C'
        
        # Use .intersects() with IntervalIndex for more robust windowing (advanced)
        # For now, your slicing approach is fine if index is datetime and sorted.

        # Filter flares within the prediction window
        future_flares = flare_events_df.loc[
            (flare_events_df.index > timestamp) & # strictly greater than current timestamp
            (flare_events_df.index <= window_end)
        ]
        
        # Get only the class letter
        # Ensure 'class' column holds values like 'C', 'M', 'X', 'B', 'A'
        future_flare_classes = future_flares["class"].values

        if "X" in future_flare_classes:
            labels.append("X")
        elif "M" in future_flare_classes:
            labels.append("M")
        elif "C" in future_flare_classes:
            labels.append("C")
        else:
            labels.append("No Flare") # Changed "B" to "No Flare" as per typical classification goals

    return labels

# Assign labels directly to the DataFrame (this will create a new column)
print(f"[INFO] Assigning labels to flux data. Prediction window: 24 hours.")
flux_df["label"] = assign_labels_to_flux(flux_df.index, flare_df, prediction_window_hours=24) # Set to 24 hours as discussed

# --- Feature engineering ---
def create_features(df, col="flux_log"):
    df = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    df["flux_delta"] = df[col].diff()

    # Define rolling windows in minutes
    windows = [5, 15, 30, 60, 180, 360, 720] # Added smaller windows for short-term dynamics
    for win in windows:
        # min_periods=1 allows calculation even if window is not full at start of series
        df[f"roll_mean_{win}"] = df[col].rolling(win, min_periods=1).mean()
        df[f"roll_std_{win}"] = df[col].rolling(win, min_periods=1).std()
        df[f"roll_min_{win}"] = df[col].rolling(win, min_periods=1).min()
        df[f"roll_max_{win}"] = df[col].rolling(win, min_periods=1).max()

    # Time-based features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear
    df["month"] = df.index.month # Add month
    df["year"] = df.index.year   # Add year

    # Dropping NaNs created by feature engineering (e.g., at the very beginning of the series)
    # This should be done after all feature creation and before saving.
    initial_rows = len(df)
    df = df.dropna()
    print(f"[INFO] Dropped {initial_rows - len(df)} rows due to NaNs after feature engineering.")
    return df

print("[INFO] Creating features for the final dataset.")
final_df = create_features(flux_df.copy()) # Pass a copy to create_features

# --- Final Save ---
final_df.to_parquet(PROCESSED_DIR / "final_dataset.parquet")
print(f"[✓] Final dataset saved: {PROCESSED_DIR / 'final_dataset.parquet'}")
print(final_df.head())
print(f"Final dataset shape: {final_df.shape}")
print(f"Label distribution:\n{final_df['label'].value_counts()}")