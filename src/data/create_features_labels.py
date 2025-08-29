# src/data/create_features_labels.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# Use a dictionary for clear, centralized configuration
CONFIG = {
    "past_history": "60min",          # Look back 60 minutes for features
    "prediction_window": "60min",     # Look ahead 60 minutes for a flare
    "flux_path": "data/interim/goes_xrs_flux_log.parquet",
    "flare_path": "data/interim/flare_catalogue_combined.parquet", # <-- CORRECTED PATH
    "output_dir": "data/processed",
    "rolling_windows": ["15min", "30min", "60min", "180min"] # Feature windows
}

def load_data(flux_path, flare_path):
    """Loads and preprocesses the interim flux and flare data."""
    print("🔍 Loading interim flux and flare data...")
    if not os.path.exists(flux_path) or not os.path.exists(flare_path):
        raise FileNotFoundError("Interim data not found. Please run previous scripts first.")
        
    flux_df = pd.read_parquet(flux_path)
    flare_df = pd.read_parquet(flare_path)

    # Ensure correct dtypes and sorted index
    flux_df.index = pd.to_datetime(flux_df.index)
    flare_df["datetime"] = pd.to_datetime(flare_df["datetime"])
    
    return flux_df.sort_index(), flare_df

def create_features(df, windows):
    """
    Engineers time-series features efficiently using pandas rolling windows.
    This is a vectorized approach and avoids slow loops.
    """
    print("🛠️  Creating features...")
    for win in windows:
        # Rolling statistics
        df[f"roll_mean_{win}"] = df["flux_log"].rolling(win).mean()
        df[f"roll_std_{win}"] = df["flux_log"].rolling(win).std()
        df[f"roll_max_{win}"] = df["flux_log"].rolling(win).max()

        # Rate of change (delta) over the window
        df[f"delta_{win}"] = df["flux_log"].diff(pd.to_timedelta(win).total_seconds() // 60)

    # Time-based features
    df["hour"] = df.index.hour
    df["day_of_year"] = df.index.dayofyear
    
    # Drop initial rows with NaNs from rolling windows
    return df.dropna()

def create_labels(flux_df, flare_df, window):
    """
    Creates labels by checking for the max flare class in a future window.
    This is a highly optimized method that avoids iterating.
    """
    print("🏷️  Creating labels...")
    # Map flare classes to numerical values (X > M > C > B > A)
    flare_map = {'X': 4, 'M': 3, 'C': 2, 'B': 1, 'A': 1}
    # Get the first letter of the class and map it
    flare_df['label'] = flare_df['class'].str[0].map(flare_map).fillna(0)
    
    # Create a Series of flare events aligned with the flux index
    flare_events = pd.Series(flare_df['label'].values, index=flare_df['datetime'])
    flare_events = flare_events.reindex(flux_df.index, method='ffill', limit=1).fillna(0)

    # Use rolling().max() on a reversed series to look into the future efficiently
    reversed_labels = flare_events.iloc[::-1].rolling(window).max().iloc[::-1]
    
    return reversed_labels.astype(int)

def create_sequences(features, labels, past_history_steps):
    """Creates input sequences (X) and target labels (y) for the model."""
    print("🧠 Creating sequences for modeling...")
    X, y, timestamps = [], [], []
    
    feature_values = features.values
    label_values = labels.values
    index_values = features.index
    
    for i in tqdm(range(len(features) - past_history_steps)):
        X.append(feature_values[i : i + past_history_steps])
        y.append(label_values[i + past_history_steps - 1]) # Label corresponds to the end of the sequence
        timestamps.append(index_values[i + past_history_steps - 1])
        
    return np.array(X), np.array(y), pd.DataFrame({"timestamp": timestamps})

def main():
    """Main pipeline function to generate the final dataset."""
    flux_df, flare_df = load_data(CONFIG["flux_path"], CONFIG["flare_path"])
    
    featured_df = create_features(flux_df, CONFIG["rolling_windows"])
    
    labels = create_labels(featured_df, flare_df, CONFIG["prediction_window"])
    
    # Align features and labels
    aligned_features, aligned_labels = featured_df.align(labels, join='inner', axis=0)
    
    # Convert past history from time string (e.g., "60min") to number of steps
    past_steps = int(pd.to_timedelta(CONFIG["past_history"]).total_seconds() // 60)
    
    X, y, meta_df = create_sequences(aligned_features, aligned_labels, past_steps)
    
    # --- Save Outputs ---
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X_features.npy"), X)
    np.save(os.path.join(output_dir, "y_labels.npy"), y)
    meta_df.to_csv(os.path.join(output_dir, "meta_index.csv"), index=False)
    
    print(f"\n[✓] Final dataset created. Shape of X: {X.shape}, Shape of y: {y.shape}")
    print(f"📁 Saved outputs to '{output_dir}' directory.")
    print("\nClass distribution in y:")
    print(pd.Series(y).value_counts(normalize=True).round(4))

if __name__ == "__main__":
    main()