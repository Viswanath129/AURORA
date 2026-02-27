
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import load_data
from optical_flow import OpticalFlowExpert
from convlstm import ConvLSTMExpert
from diffusion import DiffusionExpert
from morphology import MorphologyExpert
from routing_net import RoutingNetwork

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_image(tensor, path, cmap='gray', vmin=0, vmax=1):
    """Save a tensor as an image."""
    arr = tensor.squeeze().cpu().numpy()
    plt.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)

def generate_dashboard_data(num_samples=10):
    print(f"Generating dashboard data for {num_samples} samples...")
    
    # Create data directory structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    samples_dir = os.path.join(data_dir, 'samples')
    
    os.makedirs(samples_dir, exist_ok=True)
    
    # Load Data
    _, val_loader = load_data()
    
    # Load Models
    expert_flow = OpticalFlowExpert()
    
    expert_morph = MorphologyExpert().to(DEVICE)
    expert_morph.load_state_dict(torch.load(os.path.join(base_dir, '../expert_morph.pth'), map_location=DEVICE))
    expert_morph.eval()
    
    expert_diff = DiffusionExpert().to(DEVICE)
    expert_diff.load_state_dict(torch.load(os.path.join(base_dir, '../expert_diff.pth'), map_location=DEVICE))
    expert_diff.eval()
    
    expert_lstm = ConvLSTMExpert().to(DEVICE)
    expert_lstm.load_state_dict(torch.load(os.path.join(base_dir, '../expert_lstm.pth'), map_location=DEVICE))
    expert_lstm.eval()
    
    routing_model = RoutingNetwork().to(DEVICE)
    routing_model.load_state_dict(torch.load(os.path.join(base_dir, '../routing_net.pth'), map_location=DEVICE))
    routing_model.eval()
    
    # Initialize Metrics
    metrics = {
        'Advection': {'ssim': [], 'psnr': []},
        'Morphology': {'ssim': [], 'psnr': []},
        'Emergence': {'ssim': [], 'psnr': []},
        'Temporal': {'ssim': [], 'psnr': []},
        'AURORA': {'ssim': [], 'psnr': []}
    }
    
    samples_metadata = []
    
    with torch.no_grad():
        count = 0
        for x_batch, y_batch in val_loader:
            if count >= num_samples:
                break
                
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # Process batch (size might be > 1, take first sample if needed, or iterate batch)
            # Actually let's iterate through the batch to get individual samples until we reach num_samples
            batch_size = x_batch.size(0)
            
            # Run Predictions for the whole batch first for efficiency
            p_flow, u_flow = expert_flow.predict(x_batch)
            p_flow = p_flow.to(DEVICE); u_flow = u_flow.to(DEVICE)
            
            p_morph, u_morph = expert_morph(x_batch)
            p_diff, u_diff = expert_diff(x_batch)
            
            x_seq = x_batch.unsqueeze(2)
            p_lstm, u_lstm = expert_lstm.predict_with_uncertainty(x_seq, num_samples=3)
            p_lstm = p_lstm.to(DEVICE); u_lstm = u_lstm.to(DEVICE)
            
            curr_frame = x_batch[:, -1].unsqueeze(1)
            p_aurora, w_map = routing_model(
                p_flow, u_flow, 
                p_morph, u_morph, 
                p_diff, u_diff, 
                p_lstm, u_lstm, 
                curr_frame
            )
            
            preds_map = {
                'Advection': p_flow,
                'Morphology': p_morph,
                'Emergence': p_diff,
                'Temporal': p_lstm,
                'AURORA': p_aurora
            }
            
            # Iterate through batch
            for i in range(batch_size):
                if count >= num_samples:
                    break
                
                sample_id = f"sample_{count}"
                sample_dir = os.path.join(samples_dir, sample_id)
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save Inputs (Sequence t-3 to t)
                input_seq = x_batch[i].cpu().numpy() # (4, H, W)
                for t in range(4):
                    plt.imsave(os.path.join(sample_dir, f'input_t{t}.png'), input_seq[t], cmap='gray', vmin=0, vmax=1)
                
                # Save GT
                gt = y_batch[i].cpu().numpy()
                plt.imsave(os.path.join(sample_dir, 'gt.png'), gt, cmap='gray', vmin=0, vmax=1)
                
                sample_metrics = {}
                
                # Save Predictions and Calculate Metrics
                for key, pred_tensor in preds_map.items():
                    pred = pred_tensor[i, 0].cpu().numpy()
                    p_clamped = np.clip(pred, 0, 1)
                    
                    # Save Image
                    plt.imsave(os.path.join(sample_dir, f'pred_{key}.png'), p_clamped, cmap='inferno' if key != 'AURORA' else 'gray', vmin=0, vmax=1)
                    
                    # Metrics
                    s = ssim(gt, p_clamped, data_range=1.0)
                    p = psnr(gt, p_clamped, data_range=1.0)
                    
                    metrics[key]['ssim'].append(s)
                    metrics[key]['psnr'].append(p)
                    
                    sample_metrics[key] = {'ssim': float(s), 'psnr': float(p)}
                
                # Save Weights
                weights = w_map[i].cpu().numpy() # (4, H, W)
                weight_names = ['Advection', 'Morphology', 'Emergence', 'Temporal']
                for w_idx, w_name in enumerate(weight_names):
                    plt.imsave(os.path.join(sample_dir, f'weight_{w_name}.png'), weights[w_idx], cmap='hot', vmin=0, vmax=1)
                
                # Save Sample Metadata
                samples_metadata.append({
                    'id': sample_id,
                    'metrics': sample_metrics
                })
                
                count += 1

    # Aggregate Metrics
    final_metrics = {}
    for key, vals in metrics.items():
        final_metrics[key] = {
            'avg_ssim': float(np.mean(vals['ssim'])),
            'avg_psnr': float(np.mean(vals['psnr']))
        }
        
    # Save JSON
    output_data = {
        'summary': final_metrics,
        'samples': samples_metadata
    }
    
    with open(os.path.join(data_dir, 'dashboard_data.json'), 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Done! Generated data for {count} samples in {data_dir}")

if __name__ == "__main__":
    generate_dashboard_data(num_samples=10)
