
import numpy as np
import torch
import cv2

class MeteorologicalAnalyzer:
    def __init__(self):
        pass
        
    def analyze_signals(self, flow_tensor, curr_frame, pred_frame, uncertainty_map):
        """
        Extracts structured meteorological signals from model outputs.
        
        Args:
            flow_tensor (torch.Tensor): Optical flow (B, 2, H, W)
            curr_frame (torch.Tensor): Current frame (B, 1, H, W)
            pred_frame (torch.Tensor): Predicted frame (B, 1, H, W)
            uncertainty_map (torch.Tensor): Uncertainty (B, 1, H, W)
            
        Returns:
            dict: Structured signals for LLM prompt
        """
        # Move to CPU and numpy
        if torch.is_tensor(flow_tensor): flow = flow_tensor.detach().cpu().numpy()
        else: flow = flow_tensor
            
        if torch.is_tensor(curr_frame): curr = curr_frame.detach().cpu().numpy()
        else: curr = curr_frame
            
        if torch.is_tensor(pred_frame): pred = pred_frame.detach().cpu().numpy()
        else: pred = pred_frame
            
        if torch.is_tensor(uncertainty_map): unc = uncertainty_map.detach().cpu().numpy()
        else: unc = uncertainty_map
        
        # Batch size 1 for description usually, or take average
        # We'll assume B=1 or take the first sample for the narrative
        flow = flow[0] # (2, H, W)
        curr = curr[0, 0] # (H, W)
        pred = pred[0, 0] # (H, W)
        unc = unc[0, 0] # (H, W)
        
        # 1. Motion Analysis
        u = flow[0]
        v = flow[1]
        
        # Speed & Direction
        mag, ang = cv2.cartToPolar(u, v)
        avg_speed = np.mean(mag)
        avg_angle_rad = np.mean(ang) 
        avg_angle_deg = np.degrees(avg_angle_rad)
        
        # Convert angle to cardinal direction
        direction_str = self._get_cardinal_direction(avg_angle_deg)
        
        # Divergence (simple gradients)
        dy_u, dx_u = np.gradient(u)
        dy_v, dx_v = np.gradient(v)
        divergence = dx_u + dy_v
        mean_div = np.mean(divergence)
        
        div_pattern = "Stable"
        if mean_div > 0.05: div_pattern = "Strong Divergence (Spreading)"
        elif mean_div < -0.05: div_pattern = "Strong Convergence (Concentrating)"
        elif mean_div > 0.01: div_pattern = "Slight Divergence"
        elif mean_div < -0.01: div_pattern = "Slight Convergence"
        
        # 2. Cloud Evolution
        diff = pred - curr
        
        # Thickening/Thinning
        thickening_mask = diff > 0.1
        thinning_mask = diff < -0.1
        
        thickening_ratio = np.sum(thickening_mask) / diff.size
        thinning_ratio = np.sum(thinning_mask) / diff.size
        
        # Growth Rate (Mass change)
        curr_mass = np.sum(curr)
        pred_mass = np.sum(pred)
        growth_rate = ((pred_mass - curr_mass) / (curr_mass + 1e-6)) * 100
        
        # 3. Intensity / "Cold Tops"
        # Assuming normalized 0-1 where 1 is cloud (and usually bright/cold in IR)
        cold_tops_mask = pred > 0.8
        cold_tops_ratio = np.sum(cold_tops_mask) / pred.size
        
        # 4. Uncertainty
        avg_unc = np.mean(unc)
        unc_level = "Low"
        if avg_unc > 0.2: unc_level = "Medium"
        if avg_unc > 0.5: unc_level = "High"
        
        # Regions Helper
        def get_region_desc(mask):
            h, w = mask.shape
            y_c, x_c = np.argwhere(mask).mean(axis=0) if np.any(mask) else (h/2, w/2)
            
            # Simple grid location
            v_loc = "North" if y_c < h/3 else ("South" if y_c > 2*h/3 else "Central")
            h_loc = "West" if x_c < w/3 else ("East" if x_c > 2*w/3 else "Central")
            
            if not np.any(mask): return "None"
            return f"{v_loc}-{h_loc}"

        thick_region = get_region_desc(thickening_mask)
        thin_region = get_region_desc(thinning_mask)
        cold_region = get_region_desc(cold_tops_mask)
        unc_region = get_region_desc(unc > 0.2)
        
        return {
            "direction": direction_str,
            "speed": f"{avg_speed:.2f} px/frame",
            "thickening_zones": thick_region,
            "thinning_zones": thin_region,
            "cold_tops": cold_region,
            "growth_rate": f"{growth_rate:+.1f}%",
            "divergence": div_pattern,
            "uncertainty_level": unc_level,
            "uncertainty_regions": unc_region,
            "raw_metrics": {
                "avg_speed": avg_speed,
                "thickening_ratio": thickening_ratio,
                "cold_tops_ratio": cold_tops_ratio
            }
        }
        
    def _get_cardinal_direction(self, angle_deg):
        # Angle is usually 0=East? Depends on cv2.cartToPolar
        # cv2: 0 is East (x+), 90 is South (y+) if y goes down?
        # Let's assume standard math: 0=East, 90=North??
        # IR images: y is usually down.
        # Flow (u,v): u=right, v=down.
        # angle 0 -> u=+, v=0 -> Right (East)
        # angle 90 -> u=0, v=+ -> Down (South)
        # angle 180 -> u=-, v=0 -> Left (West)
        # angle 270 -> u=0, v=- -> Up (North)
        # Normalize to 0-360
        angle = angle_deg % 360
        
        dirs = ["East", "South-East", "South", "South-West", "West", "North-West", "North", "North-East"]
        ix = int((angle + 22.5) // 45) % 8
        return dirs[ix]

    def generate_prompt(self, signals):
        return f"""
You are a meteorology reasoning assistant.

You are given analyzed satellite cloud-motion data derived from infrared imagery.

Based on the structured signals below, describe in natural language what is likely to happen in the next 30–60 minutes in this region.

Cloud Motion Data:
- Dominant motion direction: {signals['direction']}
- Motion speed: {signals['speed']}
- Cloud thickening zones: {signals['thickening_zones']}
- Cloud thinning zones: {signals['thinning_zones']}
- Cold cloud tops concentration: {signals['cold_tops']}
- Cloud growth rate: {signals['growth_rate']}
- Divergence/Convergence pattern: {signals['divergence']}
- Model uncertainty areas: {signals['uncertainty_regions']} ({signals['uncertainty_level']} confidence)

Tasks:
1. Describe the expected cloud movement.
2. Predict areas likely to experience rainfall or storm intensification.
3. Predict areas where clouds may dissipate.
4. Mention confidence level based on uncertainty.
5. Explain the reasoning in simple meteorological terms.

Output format:
- Movement summary
- Rainfall/storm prediction
- Dissipation zones
- Confidence note
- Reasoning
"""

if __name__ == "__main__":
    # Test
    analyzer = MeteorolgicalAnalyzer()
    dummy_flow = torch.randn(1, 2, 64, 64)
    dummy_curr = torch.rand(1, 1, 64, 64)
    dummy_pred = torch.rand(1, 1, 64, 64)
    dummy_unc = torch.rand(1, 1, 64, 64) * 0.1
    
    sigs = analyzer.analyze_signals(dummy_flow, dummy_curr, dummy_pred, dummy_unc)
    print(sigs)
    print("\nPROMPT:")
    print(analyzer.generate_prompt(sigs))
