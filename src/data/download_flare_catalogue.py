#src\data\download_flare_catalogue.py

import ftplib
import os
import socket # Add for timeout

def download_flare_catalogue(ftp_host, ftp_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ftp = None # Initialize ftp to None for finally block

    try:
        # Added timeout for robustness against hanging connections
        ftp = ftplib.FTP(ftp_host, timeout=30)
        ftp.login()
        ftp.cwd(ftp_path)

        files = ftp.nlst()
        # More specific filtering to ensure we get the main historical file
        flare_files = [f for f in files if f.lower() == 'goes-xray-flares.txt' or f.lower() == 'goes_xray_flares_1975-present.txt']

        if not flare_files:
            print(f"[WARN] No specific flare catalogue file found matching 'goes-xray-flares.txt' or 'goes_xray_flares_1975-present.txt' in {ftp_path}. Please verify filename on FTP.")
            print(f"[INFO] Available files: {files}")
            return # Exit if no target file found

        for fname in flare_files: # Loop added in case there are multiple matching files
            local_path = os.path.join(output_dir, fname)
            print(f"[INFO] Attempting to download {fname} to {local_path}")
            with open(local_path, 'wb') as f:
                ftp.retrbinary('RETR ' + fname, f.write)
                print(f"[INFO] Successfully downloaded {fname}")
    except ftplib.all_errors as e: # Catch all FTP specific errors
        print(f"[ERROR] FTP Error during flare catalogue download: {e}")
    except socket.timeout: # Catch explicit timeout errors
        print(f"[ERROR] FTP connection timed out when downloading flare catalogue.")
    except Exception as e: # Catch any other unexpected errors
        print(f"[ERROR] An unexpected error occurred during flare catalogue download: {e}")
    finally:
        if ftp:
            ftp.quit() # Ensure FTP connection is closed

if __name__ == "__main__":
    download_flare_catalogue(
        ftp_host="ftp.ngdc.noaa.gov",
        ftp_path="/STP/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/",
        output_dir="data/raw/flare_catalogue/"
    )