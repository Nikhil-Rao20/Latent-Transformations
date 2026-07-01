import torch
import torch.nn as nn

class DynamicConvNet(nn.Module):
    """
    A simple bottleneck Convolutional Network used to parameterize the Scale (s) 
    and Translation (t) of the Affine Coupling Layer.
    Dynamically supports 2D or 3D inputs.
    """
    def __init__(self, dim, in_channels, out_channels, hidden_channels=256):
        super().__init__()
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        
        # 1x1 or 1x1x1 convolutions are used to ensure that the flow transformation
        # is strictly channel-wise and preserves the spatial/topological structure perfectly.
        self.net = nn.Sequential(
            conv(in_channels, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            conv(hidden_channels, out_channels, kernel_size=1)
        )
        
        # Initialize the final layer to zero so the flow starts as a pure Identity function.
        # This prevents the adapter from destroying the pre-trained features on epoch 1.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class DynamicAffineCouplingLayer(nn.Module):
    """
    RealNVP-style Affine Coupling Layer.
    Splits the latent tensor along the channel dimension and transforms one half 
    based on the other half.
    """
    def __init__(self, dim, channels):
        super().__init__()
        self.split_dim = 1 # Split along the channel dimension
        assert channels % 2 == 0, "Channels must be divisible by 2 for the coupling layer."
        half_channels = channels // 2
        
        # Scale (s) and Translation (t) networks
        self.s_net = DynamicConvNet(dim, half_channels, half_channels)
        self.t_net = DynamicConvNet(dim, half_channels, half_channels)

    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=self.split_dim)
        
        # Bound scale to [-1, 1] using tanh to prevent torch.exp() from exploding
        s = torch.tanh(self.s_net(x1))
        t = self.t_net(x1)

        if not reverse:
            # Forward mapping: Z_target -> Z_source
            y1 = x1
            y2 = x2 * torch.exp(s) + t
        else:
            # Inverse mapping: Z_source -> Z_target (Only used if cycle-consistency is tested)
            y1 = x1
            y2 = (x2 - t) * torch.exp(-s)
            
        return torch.cat([y1, y2], dim=self.split_dim)


class CyLNormalizingFlow(nn.Module):
    """
    The Core Latent Transformation Engine (T_flow).
    Stacks multiple coupling layers, flipping the channels between each layer 
    so that all channels are eventually transformed.
    """
    def __init__(self, dim=2, channels=256, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([DynamicAffineCouplingLayer(dim, channels) for _ in range(num_layers)])

    def forward(self, x, reverse=False):
        if not reverse:
            # Forward pass: warp Target Latent into Source Latent
            for layer in self.layers:
                x = layer(x, reverse=False)
                # Flip channels so the unmodified half gets transformed in the next layer
                x = torch.flip(x, dims=[1])
        else:
            # Inverse pass (Optional): warp Source Latent back into Target Latent
            for layer in reversed(self.layers):
                x = layer(x, reverse=True)
                x = torch.flip(x, dims=[1])
                
        return x
