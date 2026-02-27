import s3fs
import os
import datetime
from concurrent.futures import ThreadPoolExecutor

# Settings
BUCKET_NAME = 'noaa-goes16'
PRODUCT = 'ABI-L1b-RadC' # CONUS (Continental US) - Change to RadF for Full Disk if needed
YEAR = 2023
DAY_OF_YEAR = 200 # Arbitrary day in summer for good cloud activity
START_HOUR = 12
END_HOUR = 16
CHANNEL = 'C13' # Band 13 (Clean IR)
OUTPUT_DIR = 'dataset'

def download_file(s3_path, local_path):
    fs = s3fs.S3FileSystem(anon=True)
    try:
        if not os.path.exists(local_path):
            print(f"Downloading {s3_path}...")
            fs.get(s3_path, local_path)
        else:
            print(f"Skipping {local_path}, already exists.")
    except Exception as e:
        print(f"Error downloading {s3_path}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    fs = s3fs.S3FileSystem(anon=True)
    
    tasks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for hour in range(START_HOUR, END_HOUR + 1):
            # Construct path: bucket/product/year/day_of_year/hour/
            s3_dir = f"{BUCKET_NAME}/{PRODUCT}/{YEAR}/{DAY_OF_YEAR:03d}/{hour:02d}/"
            
            try:
                files = fs.ls(s3_dir)
                for f in files:
                    if f"M6{CHANNEL}" in f or f"M3{CHANNEL}" in f: # Mode 6 or Mode 3 scan, Channel 13
                        filename = f.split('/')[-1]
                        local_path = os.path.join(OUTPUT_DIR, filename)
                        tasks.append(executor.submit(download_file, f, local_path))
            except Exception as e:
                print(f"Could not list directory {s3_dir}: {e}")

    print("Download complete.")

if __name__ == "__main__":
    main()
