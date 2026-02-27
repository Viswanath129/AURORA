import os
import xarray as xr
import numpy as np
import cv2
import glob

# Settings
INPUT_DIR = 'dataset'
OUTPUT_DIR = 'frames'
IMG_SIZE = (256, 256)

def preprocess():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.nc')))
    
    if not files:
        print("No .nc files found in dataset/ directory.")
        return

    print(f"Found {len(files)} files. Processing...")

    for i, fpath in enumerate(files):
        try:
            ds = xr.open_dataset(fpath)
            
            # Extract Radiance
            if 'Rad' in ds:
                img = ds['Rad'].values
            else:
                print(f"Skipping {fpath}: 'Rad' variable not found.")
                ds.close()
                continue
                
            # Close dataset to free resources
            ds.close()

            # Normalize to 0-1 range (approximate min/max for IR)
            # Band 13 min/max values typically range from 0 to ~150-200 depending on units (mW/m2/sr/cm-1)
            # We use a robust min-max or fixed scaling. 
            # Let's simple min-max per frame for now, or fixed for consistency.
            # Fixed is better for ML. Let's assume max around 150 for Rad.
            
            img = np.nan_to_num(img)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            
            # Resize
            img_resized = cv2.resize(img, IMG_SIZE)
            
            # Identify valid timestamp or just use index
            out_name = f"frame_{i:03d}.npy"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            
            np.save(out_path, img_resized)
            print(f"Saved {out_path}")

        except Exception as e:
            print(f"Error processing {fpath}: {e}")

if __name__ == "__main__":
    preprocess()
