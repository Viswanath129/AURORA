import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DiffusionExpert(nn.Module):
    def __init__(self, in_channels=4, out_channels=2): 
        """
        in_channels: 4 (sequence length)
        out_channels: 2 (Mean prediction + Log Variance) -> Probabilistic UNet
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, T, H, W) -> flattened to (B, T, H, W) treated as channels for 2D UNet
        # If input is (B, T, 1, H, W), we view as (B, T, H, W)
        
        if x.dim() == 5:
            b, t, c, h, w = x.size()
            x = x.view(b, t*c, h, w)
            
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
        
        # Split into Mean and Log Variance
        # We output Log Variance for numerical stability
        mean = torch.sigmoid(output[:, 0:1, :, :]) # Cloud values 0-1
        log_var = output[:, 1:2, :, :]
        var = torch.exp(log_var)
        
        return mean, var

if __name__ == "__main__":
    model = DiffusionExpert(in_channels=4, out_channels=2)
    dummy_input = torch.randn(2, 4, 256, 256) # B, T, H, W
    mean, var = model(dummy_input)
    print(f"Mean shape: {mean.shape}, Var shape: {var.shape}")
