# src/data/create_features_labels.py
import os
import pandas as pd
import numpy as np
import argparse

def main(root_dir):
    CONFIG = {
        "prediction_window": "60min",
        "flux_path": os.path.join(root_dir, "data/interim/goes_xrs_flux_log.parquet"),
        "flare_path": os.path.join(root_dir, "data/interim/flare_catalogue_combined.parquet"),
        "output_dir": os.path.join(root_dir, "data/processed"),
        "rolling_windows": ["15min", "30min", "60min", "180min"],
    }

    # --- 1. Load Data ---
    print("🔍 Loading interim data...")
    if not os.path.exists(CONFIG["flux_path"]) or not os.path.exists(CONFIG["flare_path"]):
        print("❌ Missing required interim files. Please run the previous scripts first.")
        return

    flux_df = pd.read_parquet(CONFIG["flux_path"])
    flare_df = pd.read_parquet(CONFIG["flare_path"])
    flux_df.index = pd.to_datetime(flux_df.index)
    flare_df["datetime"] = pd.to_datetime(flare_df["datetime"])

    # --- 2. Create Rolling Features ---
    print("🛠️  Creating features...")
    for win in CONFIG["rolling_windows"]:
        flux_df[f"flux_log_mean_{win}"] = flux_df["flux_log"].rolling(win, min_periods=1).mean()
        flux_df[f"flux_log_std_{win}"] = flux_df["flux_log"].rolling(win, min_periods=1).std()
    
    # Drop rows with NaNs created during feature generation
    flux_df.dropna(inplace=True)

    # --- 3. Create Labels (CORRECT, ROBUST LOGIC) ---
    print("🏷️  Creating labels...")
    flare_map = {'X': 4, 'M': 3, 'C': 2, 'B': 1, 'A': 1}
    flare_df['label_val'] = flare_df['class'].str[0].map(flare_map).fillna(0)
    
    # Create a Series of flare events aligned with the full flux timeline
    flare_events = pd.Series(flare_df['label_val'].values, index=flare_df['datetime'])
    # Reindex and forward-fill to place flare events onto the flux timeline
    flare_events_aligned = flare_events.reindex(flux_df.index, method='ffill', limit=1).fillna(0)

    # Efficiently find the max flare class in the future prediction window
    # This is the key step for creating a forward-looking label
    labels = flare_events_aligned.iloc[::-1].rolling(CONFIG["prediction_window"]).max().iloc[::-1].fillna(0)
    
    flux_df["label"] = labels.astype(int)

    # --- 4. Save Final DataFrame ---
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    output_path = os.path.join(CONFIG["output_dir"], "features_labels.parquet")
    flux_df.to_parquet(output_path)
    
    print(f"\n[✓] Features and labels saved: {output_path} ({len(flux_df)} rows)")
    print("\nClass distribution:")
    print(flux_df["label"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create features and labels for modeling.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the project.")
    args = parser.parse_args()
    main(args.root_dir)