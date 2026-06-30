import torch
import torch.nn as nn
import os

class ConvBlock(nn.Module):
    """
    Standard nnU-Net Convolutional Block: Conv -> InstanceNorm -> LeakyReLU -> Conv -> InstanceNorm -> LeakyReLU.
    Automatically handles 2D or 3D based on the 'dim' argument.
    """
    def __init__(self, dim, in_ch, out_ch):
        super().__init__()
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        norm = nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d
        
        self.block = nn.Sequential(
            conv(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            conv(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)

class Downsample(nn.Module):
    def __init__(self, dim, in_ch, out_ch):
        super().__init__()
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        self.down = conv(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        
    def forward(self, x):
        return self.down(x)

class Upsample(nn.Module):
    def __init__(self, dim, in_ch, out_ch):
        super().__init__()
        conv_trans = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
        self.up = conv_trans(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        
    def forward(self, x):
        return self.up(x)


class NNUNetFoundation(nn.Module):
    """
    Modular nnU-Net architecture strictly following the official PlainConvUNet design.
    Explicitly decoupled into encode() and decode() to expose the bottleneck (Z_source) 
    for CyL-Adapter Latent Transformations.
    """
    def __init__(self, dim=2, in_channels=1, num_classes=4, base_filters=32, num_stages=5):
        super().__init__()
        self.dim = dim
        self.num_stages = num_stages
        
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # ---------------------
        # ENCODER
        # ---------------------
        filters = base_filters
        self.encoder.append(ConvBlock(dim, in_channels, filters))
        
        for i in range(num_stages - 1):
            self.downsamples.append(Downsample(dim, filters, filters * 2))
            filters *= 2
            self.encoder.append(ConvBlock(dim, filters, filters))
            
        # ---------------------
        # DECODER
        # ---------------------
        for i in range(num_stages - 1):
            self.upsamples.append(Upsample(dim, filters, filters // 2))
            # decoder block concatenates skip connection, so in_ch is `filters` (filters//2 from upsample + filters//2 from skip)
            self.decoder.append(ConvBlock(dim, filters, filters // 2))
            filters //= 2
            
        # ---------------------
        # CLASSIFICATION HEAD
        # ---------------------
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        self.head = conv(filters, num_classes, kernel_size=1)

    def encode(self, x):
        """
        Compresses input to the latent bottleneck space.
        Returns skip connections and the bottleneck tensor Z.
        """
        skips = []
        out = x
        for i in range(self.num_stages - 1):
            out = self.encoder[i](out)
            skips.append(out)
            out = self.downsamples[i](out)
            
        z = self.encoder[-1](out) # The highly semantic bottleneck
        return skips, z
        
    def decode(self, skips, z):
        """
        Reconstructs the latent tensor Z back into anatomical features and logits.
        """
        out = z
        for i in range(self.num_stages - 1):
            out = self.upsamples[i](out)
            skip = skips[-(i+1)]
            
            # Concatenate along channel dimension
            out = torch.cat([out, skip], dim=1)
            out = self.decoder[i](out)
            
        features = out
        logits = self.head(features)
        return logits, features
        
    def forward(self, x):
        """Standard full forward pass for pre-training."""
        skips, z = self.encode(x)
        logits, _ = self.decode(skips, z)
        return logits


def get_nnunet_model(dim, in_channels, num_classes, pretrained_path=None, device="cuda:0"):
    """
    Instantiates the NNUNetFoundation model (2D or 3D) and optionally loads pretrained weights.
    
    Expected naming convention for pretrained models:
    FM_{exp}_{trainedonseq}_{trainedcenters}.pth
    e.g., FM_1_LGE_ABC.pth
    """
    model = NNUNetFoundation(
        dim=dim, 
        in_channels=in_channels, 
        num_classes=num_classes, 
        base_filters=32, 
        num_stages=5
    )
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained Foundation Model weights from {pretrained_path}...")
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    elif pretrained_path:
        print(f"WARNING: Pretrained path {pretrained_path} not found. Returning randomly initialized model.")
        
    model = model.to(device)
    return model
