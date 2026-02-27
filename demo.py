import torch
import numpy as np
import matplotlib.pyplot as plt
from train import load_data
from optical_flow import OpticalFlowExpert
from convlstm import ConvLSTMExpert
from diffusion import DiffusionExpert
from morphology import MorphologyExpert
from routing_net import RoutingNetwork

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_demo():
    print("Generating AURORA Demo with Routing Visualization...")
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
    
    # Get Sample
    x_batch, y_batch = next(iter(val_loader))
    x = x_batch.to(DEVICE)
    y = y_batch.to(DEVICE)
    
    with torch.no_grad():
        p_flow, u_flow = expert_flow.predict(x)
        p_flow = p_flow.to(DEVICE); u_flow = u_flow.to(DEVICE)
        
        p_morph, u_morph = expert_morph(x)
        p_diff, u_diff = expert_diff(x)
        
        x_seq = x.unsqueeze(2)
        p_lstm, u_lstm = expert_lstm.predict_with_uncertainty(x_seq, num_samples=3)
        p_lstm = p_lstm.to(DEVICE); u_lstm = u_lstm.to(DEVICE)
        
        curr_frame = x[:, -1].unsqueeze(1)
        p_aurora, w_map = routing_model(
            p_flow, u_flow, 
            p_morph, u_morph, 
            p_diff, u_diff, 
            p_lstm, u_lstm, 
            curr_frame
        )
    
    # Visualization setup
    idx = 0
    in_seq = x[idx].cpu().numpy()
    gt = y[idx].cpu().numpy()
    
    # Experts
    pf = p_flow[idx, 0].cpu().numpy()
    pm = p_morph[idx, 0].cpu().numpy()
    pd = p_diff[idx, 0].cpu().numpy()
    pl = p_lstm[idx, 0].cpu().numpy()
    
    # Result
    final = p_aurora[idx, 0].cpu().numpy()
    
    # Weights (Routing Map)
    # w_map shape: (B, 4, H, W) -> (4, H, W)
    weights = w_map[idx].cpu().numpy()
    
    # Plotting
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 5)
    
    # Row 1: Inputs
    ax_in1 = fig.add_subplot(gs[0, 0]); ax_in1.imshow(in_seq[0], cmap='gray'); ax_in1.set_title("t-3")
    ax_in2 = fig.add_subplot(gs[0, 1]); ax_in2.imshow(in_seq[1], cmap='gray'); ax_in2.set_title("t-2")
    ax_in3 = fig.add_subplot(gs[0, 2]); ax_in3.imshow(in_seq[2], cmap='gray'); ax_in3.set_title("t-1")
    ax_in4 = fig.add_subplot(gs[0, 3]); ax_in4.imshow(in_seq[3], cmap='gray'); ax_in4.set_title("t (Current)")
    ax_gt  = fig.add_subplot(gs[0, 4]); ax_gt.imshow(gt, cmap='gray'); ax_gt.set_title("Ground Truth (t+1)")
    
    # Row 2: Expert Predictions
    ax_e1 = fig.add_subplot(gs[1, 0]); ax_e1.imshow(pf, cmap='jet'); ax_e1.set_title("Path 1: Advection")
    ax_e2 = fig.add_subplot(gs[1, 1]); ax_e2.imshow(pm, cmap='jet'); ax_e2.set_title("Path 2: Morphology")
    ax_e3 = fig.add_subplot(gs[1, 2]); ax_e3.imshow(pd, cmap='jet'); ax_e3.set_title("Path 3: Emergence")
    ax_e4 = fig.add_subplot(gs[1, 3]); ax_e4.imshow(pl, cmap='jet'); ax_e4.set_title("Path 4: Temporal")
    
    # Row 3: Routing Logic & Final
    # Create an RGB Routing Map where R=Advection, G=Morphology, B=Emergence (Temporal mixing or neglect for visualization simplicity, or use specific colors)
    # Let's show separate weight maps.
    
    ax_w1 = fig.add_subplot(gs[2, 0]); ax_w1.imshow(weights[0], cmap='hot', vmin=0, vmax=1); ax_w1.set_title("Routing W: Advection")
    ax_w2 = fig.add_subplot(gs[2, 1]); ax_w2.imshow(weights[1], cmap='hot', vmin=0, vmax=1); ax_w2.set_title("Routing W: Morphology")
    ax_w3 = fig.add_subplot(gs[2, 2]); ax_w3.imshow(weights[2], cmap='hot', vmin=0, vmax=1); ax_w3.set_title("Routing W: Emergence")
    
    # Final Result
    ax_final = fig.add_subplot(gs[2, 3]); ax_final.imshow(final, cmap='gray'); ax_final.set_title("AURORA Prediction", color='blue', fontweight='bold')
    
    # Error Map
    err = np.abs(final - gt)
    ax_err = fig.add_subplot(gs[2, 4]); ax_err.imshow(err, cmap='magma'); ax_err.set_title("Error Map")
    
    for ax in fig.axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('aurora_demo.png')
    print("AURORA demo saved to aurora_demo.png")
    plt.show()

if __name__ == "__main__":
    run_demo()
