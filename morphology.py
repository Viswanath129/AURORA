import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class MorphologyExpert(nn.Module):
    def __init__(self, in_channels=1, out_channels=2): 
        """
        Input: Only the LAST frame (t). 
        Architecture: U-Net
        Output: Prediction (t+1) + Uncertainty (LogVar)
        Focus: Shape/Texture evolution, ignoring temporal history.
        """
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, T, H, W) or (B, 1, H, W). We take only the last frame.
        if x.dim() == 4 and x.size(1) > 1:
            x = x[:, -1:, :, :] # Take last frame (B, 1, H, W)
            
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        output = self.outc(x)
        
        mean = torch.sigmoid(output[:, 0:1, :, :])
        log_var = output[:, 1:2, :, :]
        var = torch.exp(log_var)
        
        return mean, var

if __name__ == "__main__":
    model = MorphologyExpert()
    dummy = torch.randn(2, 4, 128, 128) # B, T, H, W
    m, v = model(dummy)
    print(f"Morphology Output: {m.shape}, Var: {v.shape}")
