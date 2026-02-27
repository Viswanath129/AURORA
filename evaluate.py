import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from train import load_data
from optical_flow import OpticalFlowExpert
from convlstm import ConvLSTMExpert
from diffusion import DiffusionExpert
from morphology import MorphologyExpert
from routing_net import RoutingNetwork

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():
    _, val_loader = load_data()
    
    # Load Models
    expert_flow = OpticalFlowExpert()
    
    expert_morph = MorphologyExpert().to(DEVICE)
    expert_morph.load_state_dict(torch.load('expert_morph.pth'))
    expert_morph.eval()
    
    expert_diff = DiffusionExpert().to(DEVICE)
    expert_diff.load_state_dict(torch.load('expert_diff.pth'))
    expert_diff.eval()
    
    expert_lstm = ConvLSTMExpert().to(DEVICE)
    expert_lstm.load_state_dict(torch.load('expert_lstm.pth'))
    expert_lstm.eval()
    
    routing_model = RoutingNetwork().to(DEVICE)
    routing_model.load_state_dict(torch.load('routing_net.pth'))
    routing_model.eval()
    
    metrics = {
        'Advection': {'ssim': [], 'psnr': []},
        'Morphology': {'ssim': [], 'psnr': []},
        'Emergence': {'ssim': [], 'psnr': []},
        'Temporal': {'ssim': [], 'psnr': []},
        'AURORA': {'ssim': [], 'psnr': []}
    }
    
    print("Evaluating AURORA vs Experts...")
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_np = y.cpu().numpy()
            
            # Predictions
            p_flow, u_flow = expert_flow.predict(x)
            p_flow = p_flow.to(DEVICE); u_flow = u_flow.to(DEVICE)
            
            p_morph, u_morph = expert_morph(x)
            
            p_diff, u_diff = expert_diff(x)
            
            x_seq = x.unsqueeze(2)
            p_lstm, u_lstm = expert_lstm.predict_with_uncertainty(x_seq, num_samples=3)
            p_lstm = p_lstm.to(DEVICE); u_lstm = u_lstm.to(DEVICE)
            
            curr_frame = x[:, -1].unsqueeze(1)
            p_aurora, _ = routing_model(
                p_flow, u_flow, 
                p_morph, u_morph, 
                p_diff, u_diff, 
                p_lstm, u_lstm, 
                curr_frame
            )
            
            # Metrics
            preds_map = {
                'Advection': p_flow,
                'Morphology': p_morph,
                'Emergence': p_diff,
                'Temporal': p_lstm,
                'AURORA': p_aurora
            }
            
            for key, pred_tensor in preds_map.items():
                pred_np = pred_tensor.squeeze(1).cpu().numpy()
                
                for i in range(len(pred_np)):
                    gt = y_np[i]
                    p = np.clip(pred_np[i], 0, 1)
                    
                    metrics[key]['ssim'].append(ssim(gt, p, data_range=1.0))
                    metrics[key]['psnr'].append(psnr(gt, p, data_range=1.0))
                    
    print(f"\n{'Model / Path':<15} | {'SSIM':<10} | {'PSNR':<10}")
    print("-" * 45)
    for name, vals in metrics.items():
        avg_s = np.mean(vals['ssim'])
        avg_p = np.mean(vals['psnr'])
        print(f"{name:<15} | {avg_s:.4f}     | {avg_p:.4f}")

if __name__ == "__main__":
    evaluate()
