# src/data/create_features_labels.py

import os
import pandas as pd
import numpy as np

# --- Configuration ---
CONFIG = {
    "prediction_window": "60min",     # Look ahead 60 minutes for a flare
    "flux_path": "data/interim/goes_xrs_flux_log.parquet",
    "flare_path": "data/interim/flare_catalogue_combined.parquet",
    "output_dir": "data/processed",
    "rolling_windows": ["15min", "30min", "60min", "180min"]
}

def load_data(flux_path, flare_path):
    """Loads the interim flux and flare parquet files."""
    print("🔍 Loading interim flux and flare data...")
    if not os.path.exists(flux_path) or not os.path.exists(flare_path):
        raise FileNotFoundError("Interim data not found. Please run previous scripts first.")
        
    flux_df = pd.read_parquet(flux_path)
    flare_df = pd.read_parquet(flare_path)

    flux_df.index = pd.to_datetime(flux_df.index)
    flare_df["datetime"] = pd.to_datetime(flare_df["datetime"])
    
    return flux_df.sort_index(), flare_df

def create_features(df, windows):
    """Engineers time-series features efficiently using pandas rolling windows."""
    print("🛠️  Creating features...")
    for win in windows:
        df[f"roll_mean_{win}"] = df["flux_log"].rolling(win, min_periods=1).mean()
        df[f"roll_std_{win}"] = df["flux_log"].rolling(win, min_periods=1).std()
        df[f"roll_max_{win}"] = df["flux_log"].rolling(win, min_periods=1).max()
        df[f"delta_{win}"] = df["flux_log"].diff(int(pd.to_timedelta(win).total_seconds() // 60))

    df["hour"] = df.index.hour
    df["day_of_year"] = df.index.dayofyear
    
    return df.dropna()

def create_labels(flux_df, flare_df, window):
    """Creates labels by checking for the max flare class in a future window."""
    print("🏷️  Creating labels...")
    flare_map = {'X': 4, 'M': 3, 'C': 2, 'B': 1, 'A': 1}
    flare_df['label'] = flare_df['class'].str[0].map(flare_map).fillna(0)
    
    flare_events = pd.Series(flare_df['label'].values, index=flare_df['datetime'])
    # Align flare events to the flux timeline
    flare_events_aligned = flare_events.reindex(flux_df.index).fillna(0)

    # Use rolling().max() on a reversed series to look into the future efficiently
    labels = flare_events_aligned.iloc[::-1].rolling(window).max().iloc[::-1].fillna(0)
    
    return labels.astype(int)

def main():
    """Main pipeline function to generate the final dataset."""
    flux_df, flare_df = load_data(CONFIG["flux_path"], CONFIG["flare_path"])
    
    featured_df = create_features(flux_df, CONFIG["rolling_windows"])
    
    labels = create_labels(featured_df, flare_df, CONFIG["prediction_window"])
    
    # Combine features and labels into a single DataFrame
    featured_df['label'] = labels
    
    final_df = featured_df.dropna()
    
    # --- Save Final Processed DataFrame ---
    output_dir = CONFIG["output_dir"]
    output_path = os.path.join(output_dir, "features_and_labels.parquet")
    os.makedirs(output_dir, exist_ok=True)
    
    final_df.to_parquet(output_path)
    
    print(f"\n[✓] Final feature dataset created. Shape: {final_df.shape}")
    print(f"📁 Saved final dataset to '{output_path}'")
    print("\nClass distribution:")
    print(final_df["label"].value_counts(normalize=True).round(4))

if __name__ == "__main__":
    main()