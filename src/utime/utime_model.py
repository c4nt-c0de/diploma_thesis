import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=2, dropout=0.0):
        super(ConvBlock, self).__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class UTime(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, dropout=0.0):
        super(UTime, self).__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 16, dropout=dropout); self.pool1 = nn.MaxPool1d(10)
        self.enc2 = ConvBlock(16, 32, dropout=dropout); self.pool2 = nn.MaxPool1d(8)
        self.enc3 = ConvBlock(32, 64, dropout=dropout); self.pool3 = nn.MaxPool1d(6)
        self.enc4 = ConvBlock(64, 128, dropout=dropout); self.pool4 = nn.MaxPool1d(4)
        
        self.bottleneck = ConvBlock(128, 256, dropout=dropout)
        
        # Decoder
        self.up4 = nn.ConvTranspose1d(256, 128, 4, stride=4)
        self.dec4 = ConvBlock(256, 128, dropout=dropout)
        self.up3 = nn.ConvTranspose1d(128, 64, 6, stride=6)
        self.dec3 = ConvBlock(128, 64, dropout=dropout)
        self.up2 = nn.ConvTranspose1d(64, 32, 8, stride=8)
        self.dec2 = ConvBlock(64, 32, dropout=dropout)
        self.up1 = nn.ConvTranspose1d(32, 16, 10, stride=10)
        self.dec1 = ConvBlock(32, 16, dropout=dropout)
        self.final_conv = nn.Conv1d(16, n_classes, 1)

    def forward(self, x):
        orig_len = x.shape[2]
        multiple = 1920
        if orig_len % multiple != 0:
            pad_len = multiple - (orig_len % multiple)
            x = F.pad(x, (0, pad_len))
        
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)
        
        b = self.bottleneck(p4)
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        out = self.final_conv(d1)
        return out[:, :, :orig_len]