
import torch
import numpy as np
import os
from train import load_data
from optical_flow import OpticalFlowExpert
from convlstm import ConvLSTMExpert
from diffusion import DiffusionExpert
from morphology import MorphologyExpert
from routing_net import RoutingNetwork
from analysis import MeteorologicalAnalyzer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_report():
    print("Initializing Weather Report Generation...")
    
    # 1. Load Data
    _, val_loader = load_data()
    if val_loader is None:
        return
        
    iter_loader = iter(val_loader)
    x_batch, y_batch = next(iter_loader) # Get first batch
    
    # Pick a random sample from the batch
    idx = 0
    x = x_batch[idx:idx+1].to(DEVICE) # (1, T, H, W)
    y = y_batch[idx:idx+1].to(DEVICE) # (1, H, W)
    
    # 2. Load Models
    print("Loading AURORA models...")
    expert_flow = OpticalFlowExpert()
    
    expert_morph = MorphologyExpert().to(DEVICE)
    expert_morph.load_state_dict(torch.load('expert_morph.pth', map_location=DEVICE))
    expert_morph.eval()
    
    expert_diff = DiffusionExpert().to(DEVICE)
    expert_diff.load_state_dict(torch.load('expert_diff.pth', map_location=DEVICE))
    expert_diff.eval()
    
    expert_lstm = ConvLSTMExpert().to(DEVICE)
    expert_lstm.load_state_dict(torch.load('expert_lstm.pth', map_location=DEVICE))
    expert_lstm.eval()
    
    routing_model = RoutingNetwork().to(DEVICE)
    routing_model.load_state_dict(torch.load('routing_net.pth', map_location=DEVICE))
    routing_model.eval()
    
    # 3. Model Inference (AURORA)
    print("Running Inference...")
    with torch.no_grad():
        # Experts
        p_flow, u_flow = expert_flow.predict(x)
        p_flow = p_flow.to(DEVICE); u_flow = u_flow.to(DEVICE)
        
        p_morph, u_morph = expert_morph(x)
        p_diff, u_diff = expert_diff(x)
        
        x_seq = x.unsqueeze(2) # (1, T, 1, H, W)
        p_lstm, u_lstm = expert_lstm.predict_with_uncertainty(x_seq, num_samples=5)
        p_lstm = p_lstm.to(DEVICE); u_lstm = u_lstm.to(DEVICE)
        
        # Router
        curr_frame = x[:, -1].unsqueeze(1) # (1, 1, H, W)
        p_aurora, w_map = routing_model(
            p_flow, u_flow, 
            p_morph, u_morph, 
            p_diff, u_diff, 
            p_lstm, u_lstm, 
            curr_frame
        )
        
    # 4. Analysis
    print("Analyzing Meteorological Signals...")
    analyzer = MeteorologicalAnalyzer()
    
    # Note: flow from 'expert_flow' is computed on CPU and returned as tensor.
    # In `predict`, it computes flow between t-1 and t.
    # We want flow vector at 't' for the prompt.
    # The expert returns 'p_flow' which is warps, but we need the actual flow vectors if possible?
    # Wait, `OpticalFlowExpert.predict` returns prediction and uncertainty map. It doesn't return the raw flow field (u, v).
    # We should re-compute flow or modify `OpticalFlowExpert` to return it.
    # For now, let's re-compute flow briefly here using the same logic, or simpler.
    
    # Extract frames for flow
    prev_frame_np = x[0, -2].cpu().numpy()
    curr_frame_np = x[0, -1].cpu().numpy()
    
    # Re-calc flow (using opencv)
    prv = (prev_frame_np * 255).astype(np.uint8)
    cur = (curr_frame_np * 255).astype(np.uint8)
    flow_cv = cv2.calcOpticalFlowFarneback(
        prv, cur, None, 
        pyr_scale=0.5, levels=3, winsize=15, 
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    # flow_cv is (H, W, 2)
    # We need (2, H, W)
    flow_tensor = torch.tensor(flow_cv.transpose(2, 0, 1)).unsqueeze(0) # (1, 2, H, W)
    
    # Signals
    signals = analyzer.analyze_signals(
        flow_tensor=flow_tensor,
        curr_frame=curr_frame, 
        pred_frame=p_aurora, 
        uncertainty_map=u_diff # Using Emergence uncertainty as proxy or average? The prompt asks for "Model uncertainty".
        # Let's use the routing uncertainty (weighted sum?) or just one.
        # AURORA doesn't output a single uncertainty map directly in `demo.py`, but `RoutingNetwork` could.
        # For now, let's use the max uncertainty across experts or just the LSTM one (temporal).
        # Better: calculate weighted uncertainty.
    )
    
    # 5. Generate Prompt
    prompt = analyzer.generate_prompt(signals)
    
    print("\n" + "="*50)
    print("  GENERATED LLM PROMPT FOR METEOROLOGY AGENT")
    print("="*50)
    print(prompt)
    print("="*50)
    
    # Save
    with open("weather_forecast_prompt.txt", "w") as f:
        f.write(prompt)
    print("\nPrompt saved to 'weather_forecast_prompt.txt'")

if __name__ == "__main__":
    import cv2 # Ensure imported for local flow calc
    generate_report()
