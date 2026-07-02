"""
train_phase1_foundation.py
==========================
Phase 1 Training: Foundation Model (NNUNetFoundation) on Source Domain Data.

This script trains the NNUNetFoundation from scratch on a specified
experiment split. After training, the frozen weights are used in Phase 2
by the CyL-Adapter.

Supports:
  - 2D Myocardium (CARE Left Myocardium dataset)
  - 3D Whole Heart (CARE Whole Heart dataset)

Saves per run (under outputs/<exp_name>_<timestamp>/):
  - best_model.pth          : best val Dice checkpoint
  - last_model.pth          : end-of-training checkpoint
  - train_log.csv           : per-epoch metrics (loss, dice per class, mean dice)
  - config_used.yaml        : copy of the experiment config for reproducibility
  - centroid_cache.npz      : per-class bottleneck centroids (computed post-training,
                              used by Phase 2)

Usage (2D):
    python training/train_phase1_foundation.py \
        --mode 2d \
        --slices_csv data/metadata_myo2d_slices.csv \
        --config configs/2d/exp_a1_lge_intercenter.yaml \
        --output_dir outputs/ \
        --epochs 200 \
        --batch_size 16 \
        --lr 1e-3 \
        --num_workers 4 \
        --device cuda:0

Usage (3D):
    python training/train_phase1_foundation.py \
        --mode 3d \
        --volumes_csv data/metadata_wholeheart3d.csv \
        --config configs/3d/exp1_ct_intercenter.yaml \
        --npy_cache_dir data/3d_npy_cache \
        --output_dir outputs/ \
        --epochs 300 \
        --batch_size 2 \
        --lr 1e-3 \
        --num_workers 2 \
        --device cuda:0
"""

import argparse
import csv
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader.myo2d_dataset import Myo2DDataset
from dataloader.wholeheart3d_dataset import WholeHeart3DDataset

# All models import
from models.nnunet_foundation import NNUNetFoundation
from models.unet_baseline import UNetBaseline

MODEL_REGISTRY = {
    "nnunet": NNUNetFoundation,
    "unet":   UNetBaseline,
}

# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

class DiceCELoss(nn.Module):
    """
    Combined Dice + Cross-Entropy loss, standard for medical segmentation.
    Dice handles class imbalance (scar is tiny); CE stabilises gradients early.
    """
    def __init__(self, num_classes: int, ce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: (B, C, ...), targets: (B, ...)
        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes)           # (B, ..., C)
        # move class dim to position 1
        dims = [0, targets_oh.dim() - 1] + list(range(1, targets_oh.dim() - 1))
        targets_oh = targets_oh.permute(*dims).float()              # (B, C, ...)

        # flatten spatial dims
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        tgt_flat   = targets_oh.view(targets_oh.shape[0], targets_oh.shape[1], -1)

        intersection = (probs_flat * tgt_flat).sum(-1)              # (B, C)
        dice_per_class = (2 * intersection + self.smooth) / \
                         (probs_flat.sum(-1) + tgt_flat.sum(-1) + self.smooth)
        dice_loss = 1.0 - dice_per_class.mean()

        return (1 - self.ce_weight) * dice_loss + self.ce_weight * ce_loss


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_dice_per_class(preds: torch.Tensor, targets: torch.Tensor,
                           num_classes: int, smooth: float = 1e-5):
    """
    preds:   (B, ...) integer class predictions
    targets: (B, ...) integer class targets
    Returns: numpy array of shape (num_classes,) — Dice per class
    """
    dice_scores = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        pred_c = (preds == c).float()
        tgt_c  = (targets == c).float()
        intersection = (pred_c * tgt_c).sum().item()
        union = pred_c.sum().item() + tgt_c.sum().item()
        dice_scores[c] = (2 * intersection + smooth) / (union + smooth)
    return dice_scores


# ─────────────────────────────────────────────
# Centroid computation (run once after training)
# ─────────────────────────────────────────────

@torch.no_grad()
def compute_global_centroid(model, loader, device):
    """
    Forward-passes all training samples through the frozen encoder,
    accumulates a SINGLE global mean bottleneck feature vector.
    """
    model.eval()
    sum_z = None
    count_z = 0

    for batch in loader:
        images  = batch["image"].to(device)
        _, z    = model.encode(images)             # z: (B, C, h, w) or (B, C, h, w, d)

        # Spatial average to get (B, C)
        z_vec = z.mean(dim=list(range(2, z.dim())))
        
        batch_sum = z_vec.sum(dim=0).cpu()         # (C,)

        if sum_z is None:
            sum_z = batch_sum
        else:
            sum_z += batch_sum
            
        count_z += z_vec.size(0)

    return (sum_z / count_z).numpy()


# ─────────────────────────────────────────────
# Train / Val loops
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, num_classes):
    model.train()
    total_loss = 0.0
    all_dice   = np.zeros(num_classes, dtype=np.float64)
    n_batches  = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        dice  = compute_dice_per_class(preds.cpu(), labels.cpu(), num_classes)

        total_loss += loss.item()
        all_dice   += dice
        n_batches  += 1

    return total_loss / n_batches, all_dice / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_dice   = np.zeros(num_classes, dtype=np.float64)
    n_batches  = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)
        dice   = compute_dice_per_class(preds.cpu(), labels.cpu(), num_classes)

        total_loss += loss.item()
        all_dice   += dice
        n_batches  += 1

    return total_loss / n_batches, all_dice / n_batches


# ─────────────────────────────────────────────
# CSV logger
# ─────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path: Path, fieldnames: list):
        self.path = path
        self.fieldnames = fieldnames
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",         required=True, choices=["2d", "3d"])
    p.add_argument("--config",       required=True, help="YAML experiment config path")
    p.add_argument("--output_dir",   default="outputs/", help="Root output directory")

    # 2D-specific
    p.add_argument("--slices_csv",   default=None, help="metadata_myo2d_slices.csv")

    # 3D-specific
    p.add_argument("--volumes_csv",  default=None, help="metadata_wholeheart3d.csv")
    p.add_argument("--npy_cache_dir",default=None, help="3D npy cache directory")

    # Model Selection 
    p.add_argument("--model", default="nnunet", choices=["nnunet", "unet"], help="Foundation model architecture to train")

    # Centroid extraction only mode
    p.add_argument("--extract_centroid_only", action="store_true", help="Skip training, just load checkpoint and extract global centroid")
    p.add_argument("--checkpoint_path", default=None, help="Path to best_model.pth (required if --extract_centroid_only)")

    # Training hyperparams
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--val_split",    type=float, default=0.15,
                   help="Fraction of train data used for validation (random split)")
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--seed",         type=int,   default=42)

    # Model
    p.add_argument("--base_filters", type=int,   default=32)
    p.add_argument("--num_stages",   type=int,   default=5)
    return p.parse_args()


def main():
    args = parse_args()

    # ── reproducibility ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── output directory ──
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    exp_name  = cfg["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path(args.output_dir) / f"phase1_{args.model}_{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config_used.yaml")
    print(f"\n{'='*60}")
    print(f"  Phase 1 Foundation Training")
    print(f"  Experiment : {exp_name}")
    print(f"  Mode       : {args.mode.upper()}")
    print(f"  Output     : {run_dir}")
    print(f"{'='*60}\n")

    # ── dataset ──
    if args.mode == "2d":
        assert args.slices_csv, "--slices_csv required for 2d mode"
        full_ds = Myo2DDataset(args.slices_csv, args.config, split="train")
    else:
        assert args.volumes_csv and args.npy_cache_dir, \
            "--volumes_csv and --npy_cache_dir required for 3d mode"
        full_ds = WholeHeart3DDataset(args.volumes_csv, args.config,
                                      split="train",
                                      npy_cache_dir=args.npy_cache_dir)

    full_ds.summary()
    num_classes = len(full_ds.classes)

    # ── train / val split (random, stratified by case_id for 2D) ──
    total      = len(full_ds)
    n_val      = max(1, int(total * args.val_split))
    n_train    = total - n_val
    g          = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val],
                                                      generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}")
    print(f"  Classes ({num_classes}): {full_ds.classes}\n")

    # ── model ──
    dim = 2 if args.mode == "2d" else 3

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(
        dim=dim,
        in_channels=1,
        num_classes=num_classes,
        base_filters=args.base_filters,
        num_stages=args.num_stages,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params  : {n_params:,}")

    # ── optimiser & scheduler ──
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = DiceCELoss(num_classes=num_classes)

    # ── early exit for centroid extraction ──
    if args.extract_centroid_only:
        assert args.checkpoint_path, "--checkpoint_path is required with --extract_centroid_only"
        print(f"\n  [Centroid Extraction Mode] Skipping training. Loading from: {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        
        centroid_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)
        global_cent = compute_global_centroid(model, centroid_loader, device)
        save_dir = Path(args.checkpoint_path).parent
        np.save(save_dir / "global_centroid.npy", global_cent)
        print(f"  Global centroid saved to: {save_dir / 'global_centroid.npy'} (norm={np.linalg.norm(global_cent):.4f})\n")
        return

    # ── CSV logger ──
    class_names = full_ds.classes
    fieldnames  = (
        ["epoch", "phase", "loss", "mean_dice", "lr"]
        + [f"dice_{c}" for c in class_names]
        + ["epoch_time_s"]
    )
    logger = CSVLogger(run_dir / "train_log.csv", fieldnames)

    # ── training loop ──
    best_val_dice = -1.0
    best_epoch    = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, num_classes)
        val_loss,   val_dice   = validate(
            model, val_loader,   criterion,           device, num_classes)

        scheduler.step()
        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        train_mean = float(np.mean(train_dice[1:]))   # skip background
        val_mean   = float(np.mean(val_dice[1:]))

        # ── log ──
        for phase, loss, dice in [("train", train_loss, train_dice),
                                   ("val",   val_loss,   val_dice)]:
            row = {
                "epoch":        epoch,
                "phase":        phase,
                "loss":         round(loss, 6),
                "mean_dice":    round(float(np.mean(dice[1:])), 6),
                "lr":           round(current_lr, 8),
                "epoch_time_s": round(epoch_time, 2),
            }
            for i, cname in enumerate(class_names):
                row[f"dice_{cname}"] = round(float(dice[i]), 6)
            logger.log(row)

        # ── checkpoint: last ──
        torch.save({
            "epoch":      epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_dice":    val_mean,
            "config":      cfg,
            "args":        vars(args),
        }, run_dir / "last_model.pth")

        # ── checkpoint: best ──
        if val_mean > best_val_dice:
            best_val_dice = val_mean
            best_epoch    = epoch
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_dice":    val_mean,
                "config":      cfg,
                "args":        vars(args),
            }, run_dir / "best_model.pth")

        # ── console ──
        if epoch % 10 == 0 or epoch == 1:
            dice_str = "  ".join(
                f"{c}:{d:.3f}" for c, d in zip(class_names[1:], val_dice[1:]))
            print(f"[{epoch:>4d}/{args.epochs}] "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_mean_dice={val_mean:.4f}  [{dice_str}]  "
                  f"lr={current_lr:.6f}  t={epoch_time:.1f}s")

    print(f"\n  Best val Dice: {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"  Models saved to: {run_dir}")

    # ── compute & save global centroid ──
    print("\n  Computing global bottleneck centroid (for Phase 2)...")
    best_ckpt = torch.load(run_dir / "best_model.pth", map_location="cpu")
    model.load_state_dict(best_ckpt["model_state"])
    model.to(device)

    # use the full training set (no val split) for centroid computation
    centroid_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    global_cent = compute_global_centroid(model, centroid_loader, device)

    centroid_path = run_dir / "global_centroid.npy"
    np.save(centroid_path, global_cent)
    print(f"  Global centroid saved to: {centroid_path} (norm={np.linalg.norm(global_cent):.4f})")

    print(f"\n  Done. All outputs in: {run_dir}\n")


if __name__ == "__main__":
    main()
