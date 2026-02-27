import torch
import torch.nn as nn

class FusionNet(nn.Module):
    def __init__(self, in_channels=6):
        """
        Input: 6 Channels
        1. Optical Flow Pred
        2. Optical Flow Uncertainty
        3. ConvLSTM Pred
        4. ConvLSTM Uncertainty
        5. Diffusion Pred
        6. Diffusion Uncertainty
        
        Output: 1 Channel (Final Prediction)
        """
        super(FusionNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() # Output 0-1 cloud cover
        )
        
    def forward(self, pred_flow, unc_flow, pred_lstm, unc_lstm, pred_diff, unc_diff):
        # Concatenate all inputs along channel dimension
        # Inputs shape: (B, 1, H, W)
        x = torch.cat([
            pred_flow, unc_flow, 
            pred_lstm, unc_lstm, 
            pred_diff, unc_diff
        ], dim=1)
        
        return self.net(x)

if __name__ == "__main__":
    model = FusionNet()
    # Dummy inputs
    b, h, w = 2, 256, 256
    inputs = [torch.randn(b, 1, h, w) for _ in range(6)]
    out = model(*inputs)
    print(f"Fusion output shape: {out.shape}")
