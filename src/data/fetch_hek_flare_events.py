# src/data/fetch_hek_flare_events_fido.py

from sunpy.net import Fido, attrs as a
import pandas as pd
import os
import datetime

def fetch_goes_flare_events_fido(start_year=2018, end_year=None, min_class='A0.1', output_filename='goes_flare_catalogue_fido_hek.csv'):
    all_results = []

    if end_year is None:
        end_year = datetime.datetime.now().year

    output_dir = "data/raw/flare_catalogue/"
    os.makedirs(output_dir, exist_ok=True)
    full_output_csv_path = os.path.join(output_dir, output_filename)

    expected_cols = [
        'event_starttime', 'event_peaktime', 'event_endtime',
        'fl_goescls', 'ar_noaanum', 'event_coord1', 'event_coord2',
        'event_id', 'event_type', 'obs_instrument', 'event_channel'
    ]

    for year in range(start_year, end_year + 1):
        start_date_str = f"{year}-01-01"
        if year == datetime.datetime.now().year:
            end_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        else:
            end_date_str = f"{year}-12-31"
            
        print(f"[INFO] Querying GOES flares from {start_date_str} to {end_date_str} (via Fido.search)...")

        try:
            # Use Fido.search with general attrs for time and HEK-specific attrs for event type/class
            results = Fido.search(
                a.Time(start_date_str, end_date_str),
                a.hek.EventType("FL"),
                a.hek.FL.GOESCls >= min_class
            )

            # --- THE FINAL FIX: Correctly process the HEKTable directly ---
            if results: # Check if the UnifiedResponse object is not empty
                # results[0] directly gives the HEKResponse object (which is an astropy.table.Table)
                hek_table = results[0] 
                
                print(f"[INFO] Found {len(hek_table)} flares in {year}.")
                
                # Convert the Astropy Table to a Pandas DataFrame, then to a list of dictionaries
                # This is the correct way to get the data into a format suitable for pd.DataFrame(all_results)
                all_results.extend(hek_table.to_pandas().to_dict('records'))
            else:
                print(f"[INFO] No results found by Fido.search for {year}.")
            # --------------------------------------------------------------

        except IndexError: # Catches if results is empty and results[0] is attempted
            print(f"[INFO] No results found by Fido.search for {year} (empty UnifiedResponse).")
        except Exception as e:
            print(f"[ERROR] Query failed for {year}: {e}")

    print(f"[INFO] Converting {len(all_results)} events to DataFrame...")

    if not all_results:
        print("[WARN] No flare events retrieved for the entire period. Creating an empty DataFrame with expected columns.")
        df = pd.DataFrame(columns=expected_cols)
    else:
        df = pd.DataFrame(all_results)
        final_cols_present = [col for col in expected_cols if col in df.columns]
        df = df[final_cols_present]
        df = df.reindex(columns=expected_cols)

    df.to_csv(full_output_csv_path, index=False)
    print(f"[SUCCESS] Flare catalogue saved to: {full_output_csv_path}")

# Example usage:
if __name__ == "__main__":
    current_year = datetime.datetime.now().year
    fetch_goes_flare_events_fido(start_year=2018, end_year=current_year, min_class="A0.1")