import torch
import torch.nn as nn
import os

class UNetConvBlock(nn.Module):
    """
    Standard U-Net Convolutional Block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """
    def __init__(self, dim, in_ch, out_ch):
        super().__init__()
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        norm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        
        self.block = nn.Sequential(
            conv(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm(out_ch),
            nn.ReLU(inplace=True),
            conv(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)


class UNetBaseline(nn.Module):
    """
    Classic U-Net Architecture (Supports 2D and 3D).
    This serves as the pure ablation baseline to prove the efficacy of the nnU-Net + CyL-Adapter.
    It is explicitly decoupled into encode() and decode() so that it can also be tested 
    with Latent Transformations if required.
    """
    def __init__(self, dim=2, in_channels=1, num_classes=4, base_filters=64, num_stages=4):
        super().__init__()
        self.dim = dim
        self.num_stages = num_stages
        
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2) if dim == 2 else nn.MaxPool3d(2)
        
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # ---------------------
        # ENCODER
        # ---------------------
        filters = base_filters
        self.encoder.append(UNetConvBlock(dim, in_channels, filters))
        
        for i in range(num_stages - 1):
            self.encoder.append(UNetConvBlock(dim, filters, filters * 2))
            filters *= 2
            
        # ---------------------
        # DECODER
        # ---------------------
        conv_trans = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
        for i in range(num_stages - 1):
            self.upsamples.append(conv_trans(filters, filters // 2, kernel_size=2, stride=2))
            self.decoder.append(UNetConvBlock(dim, filters, filters // 2))
            filters //= 2
            
        # ---------------------
        # CLASSIFICATION HEAD
        # ---------------------
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        self.head = conv(filters, num_classes, kernel_size=1)

    def encode(self, x):
        """
        Standard U-Net Contracting Path.
        """
        skips = []
        out = x
        for i in range(self.num_stages - 1):
            out = self.encoder[i](out)
            skips.append(out)
            out = self.pool(out)
            
        z = self.encoder[-1](out) # Bottleneck Latent Space
        return skips, z
        
    def decode(self, skips, z):
        """
        Standard U-Net Expanding Path.
        """
        out = z
        for i in range(self.num_stages - 1):
            out = self.upsamples[i](out)
            skip = skips[-(i+1)]
            
            # Pad if sizes mismatch due to pooling
            diff = [skip.size(d) - out.size(d) for d in range(2, skip.dim())]
            if any(d > 0 for d in diff):
                import torch.nn.functional as F
                pad = []
                for d in reversed(diff):
                    pad.extend([d // 2, d - d // 2])
                out = F.pad(out, pad)
                
            out = torch.cat([out, skip], dim=1)
            out = self.decoder[i](out)
            
        features = out
        logits = self.head(features)
        return logits, features
        
    def forward(self, x):
        skips, z = self.encode(x)
        logits, _ = self.decode(skips, z)
        return logits


def get_unet_baseline(dim, in_channels, num_classes, pretrained_path=None, device="cuda:0"):
    """
    Instantiates the UNetBaseline model (2D or 3D) and optionally loads pretrained weights.
    """
    model = UNetBaseline(
        dim=dim, 
        in_channels=in_channels, 
        num_classes=num_classes, 
        base_filters=64, 
        num_stages=4
    )
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained Baseline U-Net weights from {pretrained_path}...")
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    elif pretrained_path:
        print(f"WARNING: Pretrained path {pretrained_path} not found. Returning randomly initialized UNet.")
        
    model = model.to(device)
    return model
