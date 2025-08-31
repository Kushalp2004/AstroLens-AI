# src/data/combine_flare_catalogues.py

import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import re
import argparse

NOAA_DIR = "data/raw/flare_catalogue/"
HEK_FILE = "data/raw/flare_catalogue/goes_flare_catalogue_fido_hek.csv"
OUTPUT_PATH = "data/interim/flare_catalogue_combined.parquet"

def get_year_from_filename(file_path):
    fname = os.path.basename(file_path)
    match = re.search(r'(\d{4})', fname)
    return int(match.group(1)) if match else None

def parse_standard_noaa_txt(file_path, verbose=False):
    # This function appears to be working well based on your logs, so we'll keep it.
    flare_events = []
    file_year = get_year_from_filename(file_path)
    if file_year is None:
        return pd.DataFrame()

    classification_pattern = re.compile(r'([ABCMX]\s*\d+\.?\d*)')
    with open(file_path, "r") as f:
        for line in f:
            line = line.replace('\xa0', ' ')
            if not line.strip().startswith("31777") or len(line) < 27:
                continue
            try:
                yy_str, ddd_str = line[5:7].strip(), line[7:10].strip()
                peak_time_str_raw = line[18:22].strip()
                if not (yy_str.isdigit() and ddd_str.isdigit()): continue
                
                event_date = datetime(file_year, 1, 1) + timedelta(days=int(ddd_str) - 1)
                peak_time_str = ''.join(filter(str.isdigit, peak_time_str_raw))
                if len(peak_time_str) != 4: continue

                peak_dt = datetime(event_date.year, event_date.month, event_date.day, int(peak_time_str[:2]), int(peak_time_str[2:]))
                
                match = classification_pattern.search(line[27:])
                if match:
                    classification = match.group(1).replace(' ', '').strip()
                    if classification and classification[0] in "ABCMX":
                        flare_events.append({"datetime": peak_dt, "class": classification})
            except Exception:
                continue
    return pd.DataFrame(flare_events)

def load_noaa_all(verbose=False):
    all_txt_files = sorted(glob.glob(os.path.join(NOAA_DIR, "*.txt")))
    if not all_txt_files: return pd.DataFrame(columns=["datetime", "class"])
    
    dfs = [parse_standard_noaa_txt(fpath, verbose) for fpath in all_txt_files if not any(s in fpath.lower() for s in ["input-ytd", "seldads", "modified"])]
    
    if not dfs: return pd.DataFrame(columns=["datetime", "class"])
    
    return pd.concat(dfs).drop_duplicates().sort_values("datetime")

def load_hek_csv():
    if not os.path.exists(HEK_FILE): return pd.DataFrame(columns=["datetime", "class"])
    df = pd.read_csv(HEK_FILE)
    if "event_peaktime" not in df.columns or "fl_goescls" not in df.columns: return pd.DataFrame(columns=["datetime", "class"])
    
    df = df.rename(columns={"event_peaktime": "datetime", "fl_goescls": "class"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df[["datetime", "class"]]

def main(verbose=False):
    print("Loading NOAA .txt flare data...")
    noaa_df = load_noaa_all(verbose)
    print(f"[✓] NOAA flare events loaded: {len(noaa_df)}")

    print("Loading HEK flare data...")
    hek_df = load_hek_csv()
    print(f"[✓] HEK flare events loaded: {len(hek_df)}")

    if noaa_df.empty and hek_df.empty:
        print("[ERROR] No flare data loaded. Exiting.")
        return

    combined_df = pd.concat([noaa_df, hek_df], ignore_index=True)
    
    # ==============================================================================
    # 🚀 NEW: BEST PRACTICE FOR DE-DUPLICATION
    # ==============================================================================
    print(f"Initial combined count: {len(combined_df)}")
    
    # 1. Create a rank based on flare intensity (X=4, M=3, C=2, B/A=1)
    flare_map = {'X': 4, 'M': 3, 'C': 2, 'B': 1, 'A': 1}
    combined_df['rank'] = combined_df['class'].str[0].map(flare_map).fillna(0)
    
    # 2. Sort by time, then by the most intense flare first
    combined_df = combined_df.sort_values(by=['datetime', 'rank'], ascending=[True, False])
    
    # 3. Drop duplicates based on the timestamp, keeping the first entry (which is the strongest flare)
    deduplicated_df = combined_df.drop_duplicates(subset=['datetime'], keep='first')
    
    print(f"Deduplicated count: {len(deduplicated_df)}")
    # ==============================================================================

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # Save the clean, de-duplicated dataframe without the temporary rank column
    deduplicated_df[['datetime', 'class']].to_parquet(OUTPUT_PATH, index=False)
    print(f"[✓] Combined and deduplicated flare catalogue saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine NOAA and HEK solar flare catalogues.")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")
    args = parser.parse_args()
    main(verbose=args.verbose)