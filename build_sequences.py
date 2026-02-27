import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Settings
FRAMES_DIR = 'frames'
SEQ_LEN = 4 # Input frames
PRED_LEN = 1 # Output frame (t + 10 min)
SAVE_PATH = 'sequences.pth'

class CloudDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # Allow retrieving items if needed purely as dataset
        seq, target = self.data_list[idx]
        return torch.from_numpy(seq).float(), torch.from_numpy(target).float()

def build_sequences():
    files = sorted(glob.glob(os.path.join(FRAMES_DIR, '*.npy')))
    
    if len(files) < SEQ_LEN + PRED_LEN:
        print("Not enough frames to build sequences.")
        return

    sequences = []
    targets = []
    
    print(f"Building sequences from {len(files)} frames...")

    for i in range(len(files) - SEQ_LEN - PRED_LEN + 1):
        # Input: t, t+1, t+2, t+3
        input_files = files[i : i+SEQ_LEN]
        # Target: t+4
        target_file = files[i+SEQ_LEN]
        
        # Load arrays
        imgs = [np.load(f) for f in input_files]
        target_img = np.load(target_file)
        
        # Stack inputs -> (SEQ_LEN, H, W)
        seq_stack = np.stack(imgs, axis=0) # Shape: (4, 256, 256)
        
        sequences.append(seq_stack)
        targets.append(target_img)

    # Convert to large array or save list
    # For efficiency with limited RAM, usually we save paths or use a generator.
    # But for ~200 frames of 256x256, it fits in memory easily.
    # 200 * 256 * 256 * 4 bytes ~ 50MB. Safe.
    
    all_inputs = np.array(sequences) 
    all_targets = np.array(targets)
    
    print(f"Created {len(all_inputs)} samples.")
    print(f"Input shape: {all_inputs.shape}")
    print(f"Target shape: {all_targets.shape}")
    
    # Save as PyTorch tensors dict
    data_dict = {
        'inputs': torch.from_numpy(all_inputs).float(),
        'targets': torch.from_numpy(all_targets).float()
    }
    
    torch.save(data_dict, SAVE_PATH)
    print(f"Saved dataset to {SAVE_PATH}")

import glob
if __name__ == "__main__":
    build_sequences()
