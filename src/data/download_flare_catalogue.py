# src/data/download_flare_catalogue.py

import ftplib
import os
import socket

def download_flare_catalogue(ftp_host, ftp_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ftp = None

    try:
        ftp = ftplib.FTP(ftp_host, timeout=30)
        ftp.login()
        ftp.cwd(ftp_path)

        files = ftp.nlst()
        
        # --- CHANGE THIS BLOCK ---
        # Modified to find yearly GOES XRS report files
        flare_files = [f for f in files if f.lower().startswith('goes-xrs-report_') and f.lower().endswith('.txt')]
        # -------------------------

        if not flare_files:
            print(f"[WARN] No GOES XRS yearly report files found (e.g., 'goes-xrs-report_YYYY.txt') in {ftp_path}. Please verify filename pattern on FTP.")
            print(f"[INFO] Available files in this directory: {files}")
            return # Exit if no target files found

        print(f"[INFO] Found {len(flare_files)} yearly GOES XRS report files. Attempting to download...")
        for fname in flare_files:
            local_path = os.path.join(output_dir, fname)
            print(f"[INFO] Downloading {fname} to {local_path}")
            with open(local_path, 'wb') as f:
                ftp.retrbinary('RETR ' + fname, f.write)
                print(f"[INFO] Successfully downloaded {fname}")
    except ftplib.all_errors as e:
        print(f"[ERROR] FTP Error during flare catalogue download: {e}")
    except socket.timeout:
        print(f"[ERROR] FTP connection timed out when downloading flare catalogue.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during flare catalogue download: {e}")
    finally:
        if ftp:
            ftp.quit()

if __name__ == "__main__":
    download_flare_catalogue(
        ftp_host="ftp.ngdc.noaa.gov",
        ftp_path="/STP/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/",
        output_dir="data/raw/flare_catalogue/"
    )