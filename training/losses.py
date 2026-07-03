import torch
import torch.nn as nn
import torch.nn.functional as F

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / band) for band in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        
        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = XX + YY - XY - YX
        return loss

class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes)
        dims = [0, targets_oh.dim() - 1] + list(range(1, targets_oh.dim() - 1))
        targets_oh = targets_oh.permute(*dims).float()
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        tgt_flat   = targets_oh.view(targets_oh.shape[0], targets_oh.shape[1], -1)
        intersection = (probs_flat * tgt_flat).sum(-1)
        dice_per_class = (2 * intersection + self.smooth) / \
                         (probs_flat.sum(-1) + tgt_flat.sum(-1) + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()
        return (1 - self.ce_weight) * dice_loss + self.ce_weight * ce_loss

class CyLAdapterLoss(nn.Module):
    def __init__(self, num_classes, mmd_weight=0.01, cycle_weight=0.01, use_pure_ce=False):
        super().__init__()
        self.mmd_loss = MMDLoss()
        
        if use_pure_ce:
            self.seg_loss_fn = nn.CrossEntropyLoss()
        else:
            self.seg_loss_fn = DiceCELoss(num_classes=num_classes)
            
        self.mmd_weight = mmd_weight
        self.cycle_weight = cycle_weight
        
    def forward(self, preds, targets, z_aligned, cine_centroid, z_lge_original, T_flow_module):
        # 1. Supervised Segmentation Loss (CE + Dice)
        seg_loss = self.seg_loss_fn(preds, targets)
        
        # 2. Latent Alignment Loss (MMD)
        if cine_centroid is not None:
            # Spatially average to get (B, C)
            spatial_dims = list(range(2, z_aligned.dim()))
            z_aligned_flat = z_aligned.mean(dim=spatial_dims)
            
            if cine_centroid.dim() > 2:
                cine_centroid_flat = cine_centroid.mean(dim=spatial_dims)
            else:
                cine_centroid_flat = cine_centroid
                
            # Ensure it is at least 2D: (1, C)
            if cine_centroid_flat.dim() == 1:
                cine_centroid_flat = cine_centroid_flat.unsqueeze(0)
                
            if cine_centroid_flat.size(0) == 1:
                cine_centroid_flat = cine_centroid_flat.expand(z_aligned_flat.size(0), -1)
                
            align_loss = self.mmd_loss(z_aligned_flat, cine_centroid_flat)
        else:
            align_loss = torch.tensor(0.0, device=preds.device)
            
        # 3. Cycle Consistency Loss
        if self.cycle_weight > 0.0:
            z_reconstructed = T_flow_module(z_aligned, reverse=True)
            cycle_loss = F.mse_loss(z_reconstructed, z_lge_original)
        else:
            cycle_loss = torch.tensor(0.0, device=preds.device)
            
        total_loss = seg_loss + (self.mmd_weight * align_loss) + (self.cycle_weight * cycle_loss)
        return total_loss, seg_loss, align_loss, cycle_loss
