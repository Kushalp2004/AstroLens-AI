# src/data/combine_flare_catalogues.py
import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import re
import argparse

def parse_noaa_txt_robust(file_path, verbose=False):
    """A robust parser for fixed-width and inconsistently spaced NOAA files."""
    flare_events = []
    fname = os.path.basename(file_path)
    year_match = re.search(r'(\d{4})', fname)
    if not year_match: return pd.DataFrame()
    file_year = int(year_match.group(1))
    
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            line = line.replace('\xa0', ' ').strip()
            if not line or not line.startswith(('31777', '1', '2', '3', '4', '5', '6', '7', '8', '9')): continue
            
            try:
                parts = line.split()
                if len(parts) < 6: continue

                # Find date/time and class based on common patterns
                date_part = parts[0]
                if len(date_part) >= 7: # Format like 31777150101
                    yy, ddd = int(date_part[5:7]), int(date_part[7:10])
                else: # Sometimes year is implied
                    yy, ddd = int(parts[0]), int(parts[1])

                peak_time_str = parts[3] # Usually the 4th element
                
                event_date = datetime(file_year, 1, 1) + timedelta(days=ddd - 1)
                peak_dt = datetime(event_date.year, event_date.month, event_date.day, int(peak_time_str[0:2]), int(peak_time_str[2:4]))

                # Find the flare class, which is usually the first single letter after the time info
                flare_class = ""
                for part in parts[4:]:
                    if re.match(r'^[ABCMX]\d*\.?\d*$', part):
                        flare_class = part
                        break
                
                if flare_class:
                    flare_events.append({"datetime": peak_dt, "class": flare_class})
            except (ValueError, IndexError):
                if verbose: print(f"[WARN] Skipping malformed line in {fname}: {line}")
                continue
    return pd.DataFrame(flare_events)

def load_noaa_all(noaa_dir, verbose=False):
    all_txt_files = sorted(glob.glob(os.path.join(noaa_dir, "*.txt")))
    if not all_txt_files: return pd.DataFrame(columns=["datetime", "class"])
    
    dfs = [parse_noaa_txt_robust(f, verbose) for f in all_txt_files if not any(s in f.lower() for s in ["input-ytd", "seldads", "modified"])]
    
    if not any(not df.empty for df in dfs): return pd.DataFrame(columns=["datetime", "class"])
    
    return pd.concat(dfs, ignore_index=True)

def load_hek_csv(hek_file):
    if not os.path.exists(hek_file): return pd.DataFrame(columns=["datetime", "class"])
    df = pd.read_csv(hek_file)
    # FIX: Use the correct column names from the upstream script
    if "event_peaktime" not in df.columns or "fl_goescls" not in df.columns:
        print("[ERROR] HEK file missing 'event_peaktime' or 'fl_goescls'.")
        return pd.DataFrame(columns=["datetime", "class"])
    
    df = df.rename(columns={"event_peaktime": "datetime", "fl_goescls": "class"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df[["datetime", "class"]].dropna()

def main(root_dir, verbose=False):
    NOAA_DIR = os.path.join(root_dir, "data/raw/flare_catalogue/")
    HEK_FILE = os.path.join(NOAA_DIR, "goes_flare_catalogue_fido_hek.csv")
    OUTPUT_PATH = os.path.join(root_dir, "data/interim/flare_catalogue_combined.parquet")

    print("Loading NOAA .txt flare data...")
    noaa_df = load_noaa_all(NOAA_DIR, verbose)
    print(f"[✓] NOAA flare events loaded: {len(noaa_df)}")

    print("Loading HEK flare data...")
    hek_df = load_hek_csv(HEK_FILE)
    print(f"[✓] HEK flare events loaded: {len(hek_df)}")

    if noaa_df.empty and hek_df.empty:
        print("[ERROR] No flare data loaded. Exiting.")
        return

    combined_df = pd.concat([noaa_df, hek_df], ignore_index=True)
    print(f"Initial combined count: {len(combined_df)}")
    
    flare_map = {'X': 4, 'M': 3, 'C': 2, 'B': 1, 'A': 1}
    combined_df['rank'] = combined_df['class'].str[0].map(flare_map).fillna(0)
    combined_df = combined_df.sort_values(by=['datetime', 'rank'], ascending=[True, False])
    deduplicated_df = combined_df.drop_duplicates(subset=['datetime'], keep='first')
    
    print(f"Deduplicated count: {len(deduplicated_df)}")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    deduplicated_df[['datetime', 'class']].to_parquet(OUTPUT_PATH, index=False)
    print(f"[✓] Combined and deduplicated flare catalogue saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine NOAA and HEK solar flare catalogues.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the project.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    main(args.root_dir, verbose=args.verbose)