from sunpy.net import Fido, attrs as a
import os
import astropy.units as u # Keep this if you have it from previous suggestions

def download_goes_xrs(start_date: str, end_date: str, output_dir: str):
    """Download GOES XRS data from NOAA using sunpy."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Querying GOES XRS from {start_date} to {end_date}...")
    try:
        query = Fido.search(
            a.Time(start_date, end_date),
            a.Instrument('XRS') # Capitalize XRS as per sunpy examples
            # Remove: a.goes.XRS.detector('long')
        )
        print(f"[INFO] Found {len(query)} files.")
        files = Fido.fetch(query, path=os.path.join(output_dir, "{file}"))
        print(f"[INFO] Downloaded {len(files)} files to {output_dir}")
        return files
    except Exception as e:
        print(f"[ERROR] Failed to download GOES XRS data: {e}")
        return []

if __name__ == "__main__":
    download_goes_xrs("2012-01-01", "2025-07-26", "data/raw/goes_xrs/")