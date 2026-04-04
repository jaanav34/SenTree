import os
import urllib.request
from pathlib import Path

# Always download into the repo's `data/raw` directory, regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
raw_dir = REPO_ROOT / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)
print(f"Downloading into: {raw_dir}")

# The base URL for the ISIMIP3b atmospheric forcings
base_url = "https://files.isimip.org/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily/ssp370/GFDL-ESM4/"

files = [
    "gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_daily_2015_2020.nc",
    "gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_daily_2021_2030.nc",
    "gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_daily_2031_2040.nc",
    "gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_daily_2041_2050.nc",
    "gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_daily_2051_2060.nc",
    "gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_daily_2061_2070.nc",
    "gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_daily_2071_2080.nc",
    "gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_daily_2081_2090.nc",
    "gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_global_daily_2091_2100.nc"
]

print("Starting ISIMIP Data Fetch...")
for filename in files:
    file_path = raw_dir / filename
    url = base_url + filename
    
    if not file_path.exists():
        print(f"Downloading: {filename}...")
        try:
            urllib.request.urlretrieve(url, str(file_path))
            print(f"Finished: {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
    else:
        print(f"Skipped (already exists): {filename}")
        
print("All files acquired!")

