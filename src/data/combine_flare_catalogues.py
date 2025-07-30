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

def parse_standard_noaa_txt(file_path, verbose=True): # Keep verbose=True for this run to confirm the fix
    flare_events = []
    file_year = get_year_from_filename(file_path)
    if file_year is None:
        if verbose:
            print(f"[WARN] Could not determine year from {os.path.basename(file_path)}. Skipping file.")
        return pd.DataFrame()

    # Regex for classification: finds A,B,C,M,X followed by optional space, digits, optional decimal and digits
    classification_pattern = re.compile(r'([ABCMX]\s*\d+\.?\d*)')

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            line_num = i + 1
            
            # CRITICAL FIX: Replace Non-Breaking Spaces with regular spaces
            line = line.replace('\xa0', ' ') 

            # Initial filter: must start with '31777' and be long enough for core data
            if not line.strip().startswith("31777") or len(line) < 27: # Reduced minimum length check for now
                if verbose and line.strip():
                    print(f"[DEBUG] Line {line_num} skipped (header/too short/no 31777 prefix): '{line.strip()}' in {os.path.basename(file_path)}")
                continue

            try:
                # Fixed-width extraction for YY, DDD, and times (based on re-evaluation)
                yy_str = line[5:7].strip()
                ddd_str = line[7:10].strip()
                start_time_str_raw = line[13:17].strip()
                peak_time_str_raw = line[18:22].strip() # CORRECTED INDEX
                end_time_str_raw = line[23:27].strip()

                if not yy_str.isdigit() or not ddd_str.isdigit():
                    if verbose:
                        print(f"[DEBUG] Line {line_num} skipped (Malformed YY/DDD: '{yy_str}', '{ddd_str}') in {os.path.basename(file_path)}: '{line.strip()}'")
                    continue
                
                yy = int(yy_str)
                ddd = int(ddd_str)

                base_date = datetime(file_year, 1, 1)
                event_date = base_date + timedelta(days=ddd - 1)

                peak_time_str = ''.join(filter(str.isdigit, peak_time_str_raw))

                if not peak_time_str or len(peak_time_str) != 4:
                    if verbose:
                        print(f"[DEBUG] Line {line_num} skipped (Malformed peak time after clean: '{peak_time_str_raw}' -> '{peak_time_str}') in {os.path.basename(file_path)}: '{line.strip()}'")
                    continue

                peak_hour = int(peak_time_str[0:2])
                peak_minute = int(peak_time_str[2:4])

                peak_dt = datetime(event_date.year, event_date.month, event_date.day, peak_hour, peak_minute)
                
                # Search for classification using regex in the rest of the line after end time
                remainder_of_line = line[27:] # Search from after the end time field
                classification_match = classification_pattern.search(remainder_of_line)
                
                if classification_match:
                    classification = classification_match.group(1).replace(' ', '').strip()
                    # Final validation of the extracted classification string
                    if classification and classification[0] in "ABCMX" and re.match(r'[ABCMX]\d+\.?\d*$', classification):
                        flare_events.append({"datetime": peak_dt, "class": classification})
                        if verbose:
                            print(f"[DEBUG] Line {line_num} PARSED OK: dt={peak_dt}, class={classification} from '{line.strip()}'")
                    else:
                        if verbose:
                            print(f"[DEBUG] Line {line_num} skipped (Invalid X-ray class format after regex match: '{classification_match.group(1)}' -> '{classification}') in {os.path.basename(file_path)}: '{line.strip()}'")
                else:
                    if verbose:
                        print(f"[DEBUG] Line {line_num} skipped (No valid X-ray class found by regex) in {os.path.basename(file_path)}: '{line.strip()}'")
                    pass # No classification found in the line
            except ValueError as ve:
                if verbose:
                    print(f"[WARN] Line {line_num} FAILED TO PARSE (ValueError: {ve}) in {os.path.basename(file_path)}: '{line.strip()}'")
                continue
            except Exception as e:
                if verbose:
                    print(f"[ERROR] Line {line_num} FAILED TO PARSE (Unexpected Error: {e}) in {os.path.basename(file_path)}: '{line.strip()}'")
                continue
    return pd.DataFrame(flare_events)

def load_noaa_all(verbose=True): # Keep verbose=True
    all_txt_files = sorted(glob.glob(os.path.join(NOAA_DIR, "*.txt")))
    dfs = []
    if not all_txt_files:
        print(f"[WARN] No .txt files found in {NOAA_DIR}.")
        return pd.DataFrame(columns=["datetime", "class"])
        
    skip_list = ["2017-input-ytd", "seldads", "2015_modifiedreplacedmissingrows"]
    for fpath in all_txt_files:
        fname = os.path.basename(fpath)
        if any(skip in fname.lower() for skip in skip_list):
            if verbose:
                print(f"[INFO] Skipping known problematic file: {fname}.")
            continue
            
        if verbose:
            print(f"\n[INFO] Processing NOAA file: {fname}")
        df = parse_standard_noaa_txt(fpath, verbose)
        if not df.empty:
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No valid NOAA flare events parsed.")
    return pd.concat(dfs).drop_duplicates().sort_values("datetime")

def load_hek_csv():
    if not os.path.exists(HEK_FILE):
        print(f"[ERROR] HEK file not found: {HEK_FILE}")
        return pd.DataFrame(columns=["datetime", "class"])

    df = pd.read_csv(HEK_FILE)
    if "event_peaktime" not in df.columns or "fl_goescls" not in df.columns:
        print(f"[ERROR] Missing expected columns in HEK file.")
        return pd.DataFrame(columns=["datetime", "class"])

    df = df.rename(columns={"event_peaktime": "datetime", "fl_goescls": "class"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df[["datetime", "class"]]

def main(verbose=True): # Keep verbose=True
    print("Loading NOAA .txt flare data...")
    noaa_df = pd.DataFrame(columns=["datetime", "class"])
    try:
        noaa_df = load_noaa_all(verbose)
        print(f"[✓] NOAA flare events loaded: {len(noaa_df)}")
    except Exception as e:
        print(f"[ERROR] Failed loading NOAA data: {e}")
        noaa_df = pd.DataFrame(columns=["datetime", "class"])

    print("Loading HEK flare data...")
    hek_df = load_hek_csv()
    print(f"[✓] HEK flare events loaded: {len(hek_df)}")

    if noaa_df.empty and hek_df.empty:
        print("[ERROR] No flare data loaded. Exiting.")
        return

    combined_df = pd.concat([noaa_df, hek_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates().sort_values("datetime").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[✓] Combined flare catalogue saved: {OUTPUT_PATH} ({len(combined_df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine NOAA and HEK solar flare catalogues.")
    parser.add_argument("--verbose", action="store_true", help="Print debug info", default=True) # Keep default=True
    args = parser.parse_args()
    main(verbose=args.verbose)