import cv2
import numpy as np
import torch

class OpticalFlowExpert:
    def __init__(self):
        pass

    def predict(self, frames_sequence):
        """
        frames_sequence: (B, T, H, W) numpy array or tensor, normalized 0-1
        T should be at least 2 to calculate flow.
        We will use the last 2 frames to predict motion.
        """
        if torch.is_tensor(frames_sequence):
            frames_sequence = frames_sequence.cpu().numpy()
            
        B, T, H, W = frames_sequence.shape
        
        # Use last two frames: t-1 and t
        prev_frame = frames_sequence[:, -2, :, :]
        curr_frame = frames_sequence[:, -1, :, :]
        
        predictions = []
        uncertainties = []
        
        for b in range(B):
            # Convert to uint8 for OpenCV (0-255)
            prv = (prev_frame[b] * 255).astype(np.uint8)
            cur = (curr_frame[b] * 255).astype(np.uint8)
            
            # Farneback Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                prv, cur, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Warp current frame using flow to get next frame prediction
            # Flow is (dx, dy). We want to map pixels from 'cur' to 'next'.
            # A simple approximation is forward warping 'cur' by 'flow'.
            # Or assume flow is constant and warp 'cur' by 'flow' to get 'next'.
            
            h, w = flow.shape[:2]
            flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
            dst_map = flow_map + flow.reshape(-1, 2) # Apply displacement
            
            # Remap requires inverse map usually, but for simple interpolation:
            # We construct a flow map for remap: map_x, map_y where each pixel tells *where to sample from*.
            # If we assume constant velocity: Next = Current shifted by Flow.
            # So Pixel(x,y) in Next comes from Pixel(x-dx, y-dy) in Current.
            
            flow_inv = -flow
            map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
            map_y = np.repeat(np.arange(h), w).reshape(h, w).astype(np.float32)
            
            map_x += flow_inv[..., 0]
            map_y += flow_inv[..., 1]
            
            warped_next = cv2.remap(cur, map_x, map_y, interpolation=cv2.INTER_LINEAR)
            
            # Normalize back to 0-1
            pred = warped_next.astype(np.float32) / 255.0
            
            # Uncertainty: Magnitude of flow? Or temporal consistency?
            # High flow magnitude might imply higher uncertainty for simple linear extrapolation.
            # Local variance in flow field is a good proxy for uncertainty (turbulence).
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            uncertainty_map = mag / (mag.max() + 1e-6) # Normalize relative to max flow in frame
            
            predictions.append(pred)
            uncertainties.append(uncertainty_map)
            
        return torch.tensor(np.array(predictions)).float().unsqueeze(1), \
               torch.tensor(np.array(uncertainties)).float().unsqueeze(1)

if __name__ == "__main__":
    # Test
    expert = OpticalFlowExpert()
    dummy_input = np.random.rand(2, 4, 256, 256).astype(np.float32)
    pred, unc = expert.predict(dummy_input)
    print(f"Prediction shape: {pred.shape}, Uncertainty shape: {unc.shape}")
