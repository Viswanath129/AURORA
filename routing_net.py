import torch
import torch.nn as nn
import torch.nn.functional as F

class RoutingNetwork(nn.Module):
    def __init__(self, num_experts=4):
        """
        Inputs per expert: Prediction (1ch), Uncertainty (1ch) -> 2 channels
        Total Expert Inputs: 4 * 2 = 8 channels
        
        Context Input: Last frame (1ch) -> processed by Context Encoder
        """
        super().__init__()
        
        # Context Encoder (Simple CNN to extract features from the image itself)
        # Why? To know if we are looking at edges, smooth areas, etc.
        self.context_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Routing Logic
        # Input: 8 (Experts) + 32 (Context) = 40 channels
        self.router = nn.Sequential(
            nn.Conv2d(40, 64, kernel_size=1), # 1x1 conv to mix info
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, num_experts, kernel_size=1) # Output logits for weights
        )
        
    def forward(self, 
                pred_flow, unc_flow, 
                pred_morph, unc_morph,
                pred_diff, unc_diff,
                pred_lstm, unc_lstm,
                current_frame):
        """
        All preds/uncs: (B, 1, H, W)
        current_frame: (B, 1, H, W) - The 't' frame
        """
        
        # 1. Stack Experts
        experts_stack = torch.cat([
            pred_flow, unc_flow,
            pred_morph, unc_morph,
            pred_diff, unc_diff,
            pred_lstm, unc_lstm
        ], dim=1) # (B, 8, H, W)
        
        # 2. Get Context
        context_feat = self.context_encoder(current_frame) # (B, 32, H, W)
        
        # 3. Combine
        combined_in = torch.cat([experts_stack, context_feat], dim=1) # (B, 40, H, W)
        
        # 4. Compute Weights
        logits = self.router(combined_in) # (B, 4, H, W)
        weights = F.softmax(logits, dim=1) # (B, 4, H, W), sum across dim 1 is 1.0
        
        # 5. Fuse
        # Weighted sum of predictions
        # Preds: Flow(0), Morph(1), Diff(2), LSTM(3)
        final_pred = (
            weights[:, 0:1] * pred_flow +
            weights[:, 1:2] * pred_morph +
            weights[:, 2:3] * pred_diff +
            weights[:, 3:4] * pred_lstm
        )
        
        return final_pred, weights

if __name__ == "__main__":
    model = RoutingNetwork()
    b, h, w = 2, 128, 128
    dummy_input = torch.randn(b, 1, h, w)
    
    # Simulate expert outputs
    p1, u1 = torch.randn(b, 1, h, w), torch.rand(b, 1, h, w)
    p2, u2 = torch.randn(b, 1, h, w), torch.rand(b, 1, h, w)
    p3, u3 = torch.randn(b, 1, h, w), torch.rand(b, 1, h, w)
    p4, u4 = torch.randn(b, 1, h, w), torch.rand(b, 1, h, w)
    
    out, w_map = model(p1, u1, p2, u2, p3, u3, p4, u4, dummy_input)
    print(f"Fused Output: {out.shape}, Routing Map: {w_map.shape}")
