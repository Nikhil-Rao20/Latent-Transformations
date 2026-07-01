import torch
import torch.nn as nn
from .normalizing_flow import CyLNormalizingFlow

class CyLAdapterModel(nn.Module):
    """
    Cross-Modality Latent Adapter Wrapper.
    Takes a pre-trained Foundation Model (e.g., nnU-Net), freezes its parameters, 
    and injects a Dynamic Normalizing Flow at the bottleneck to align Target latent 
    representations with the Source domain.
    """
    def __init__(self, foundation_model, dim=2, num_flow_layers=3):
        super().__init__()
        
        self.foundation = foundation_model
        
        # Explicitly Freeze the Foundation Model to prevent catastrophic forgetting
        for param in self.foundation.parameters():
            param.requires_grad = False
        self.foundation.eval() # Ensure BatchNorms don't update running stats
            
        # Initialize the dynamic Forward Flow engine
        self.T_flow = CyLNormalizingFlow(dim=dim, channels=self.foundation.bottleneck_channels, num_layers=num_flow_layers)
        
    def forward_foundation(self, x):
        """
        Standard forward pass for pure Source Data.
        Bypasses the adapter entirely.
        """
        return self.foundation(x)
        
    def forward(self, x):
        """
        Target Inference via T_flow Latent Transformation.
        1. Encodes target image to get Z_target.
        2. Warps Z_target -> Z_source_aligned using the Flow.
        3. Decodes the aligned latent using the frozen Source Decoder.
        """
        # 1. Encode Target to Latent Bottleneck
        skips, z_target = self.foundation.encode(x)
        
        # 2. Warp Target Latent into Source Domain
        z_source_aligned = self.T_flow(z_target, reverse=False)
        
        # 3. Decode Aligned Latent using Frozen Source Decoder
        logits, _ = self.foundation.decode(skips, z_source_aligned)
        
        return logits, z_target, z_source_aligned

    def train(self, mode=True):
        """
        Override the train method to ensure the Foundation Model STAYS in eval mode 
        even when we call adapter.train() for Phase 2.
        """
        super().train(mode)
        # Force foundation to stay in eval mode
        self.foundation.eval()
        return self
