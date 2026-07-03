"""
train_phase2_adapter.py
=======================
Phase 2 Training: CyL-Adapter (Normalizing Flow) on Target Domain Data.

The Foundation Model is FULLY FROZEN. Only the CyL-Adapter (T_flow) is
trained. The adapter learns to warp the target domain bottleneck
(Z_target) into alignment with the source domain bottleneck (Z_source),
guided by:

    L_total = L_seg  (Dice + CE on segmentation output)
            + λ_align * L_align  (MSE between Z_aligned and per-class centroids)

L_align uses the pre-computed per-class centroids from Phase 1
(centroid_cache.npz). This is the key signal that pulls the target
latent distribution toward the source distribution.

What this script expects:
  - A trained Phase 1 checkpoint (best_model.pth)
  - The centroid_cache.npz from the same Phase 1 run
  - A target domain dataset (uses the "test" split of the config, which
    IS the target domain — we train the adapter on a SMALL PORTION of
    target data in a one-shot or few-shot manner)

One-shot vs few-shot:
  Set --adapter_train_cases to limit how many target cases the adapter
  sees during Phase 2. Default is 1 (true one-shot). Set to -1 to use
  all available target cases (upper bound).

Saves per run (under outputs/<exp_name>_<timestamp>/):
  - best_adapter.pth    : best val Dice checkpoint (adapter weights only)
  - last_adapter.pth    : end-of-training adapter weights
  - adapter_log.csv     : per-epoch metrics
  - config_used.yaml    : experiment config copy

Usage:
    python training/train_phase2_adapter.py \
        --mode 2d \
        --slices_csv data/metadata_myo2d_slices.csv \
        --config configs/2d/exp_a1_lge_intercenter.yaml \
        --foundation_ckpt outputs/phase1_exp_a1_LGE_ABC_.../best_model.pth \
        --centroid_cache outputs/phase1_exp_a1_LGE_ABC_.../centroid_cache.npz \
        --output_dir outputs/ \
        --adapter_train_cases 1 \
        --epochs 100 \
        --batch_size 8 \
        --lr 1e-4 \
        --lambda_align 0.5 \
        --device cuda:0
"""

import argparse
import csv
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader.myo2d_dataset import Myo2DDataset
from dataloader.wholeheart3d_dataset import WholeHeart3DDataset
from models.model_adapter import CyLAdapterModel

# Models Importing
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


class GlobalAlignmentLoss(nn.Module):
    """
    Combined alignment loss (MSE + L1 + Cosine) between the aligned bottleneck 
    and the single global source-domain centroid.
    """
    def __init__(self, global_centroid: np.ndarray, device: torch.device):
        super().__init__()
        self.centroid = torch.tensor(global_centroid, dtype=torch.float32, device=device)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, z_aligned: torch.Tensor, labels=None):
        """
        z_aligned: (B, C, h, w) or (B, C, h, w, d)
        labels: ignored (kept for API compatibility with old loops)
        """
        # Spatially average to get (B, C)
        spatial_dims = list(range(2, z_aligned.dim()))
        z_vec = z_aligned.mean(dim=spatial_dims)

        # Expand centroid to batch size: (B, C)
        c_batch = self.centroid.unsqueeze(0).expand(z_vec.size(0), -1)

        loss_mse = self.mse(z_vec, c_batch)
        loss_l1  = self.l1(z_vec, c_batch)
        loss_cos = 1.0 - F.cosine_similarity(z_vec, c_batch, dim=1).mean()

        return loss_mse + loss_l1 + loss_cos


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_dice_per_class(preds, targets, num_classes, smooth=1e-5):
    dice_scores = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        pred_c = (preds == c).float()
        tgt_c  = (targets == c).float()
        intersection = (pred_c * tgt_c).sum().item()
        union = pred_c.sum().item() + tgt_c.sum().item()
        dice_scores[c] = (2 * intersection + smooth) / (union + smooth)
    return dice_scores


# ─────────────────────────────────────────────
# Case-level subset selection (one-shot / few-shot)
# ─────────────────────────────────────────────

def select_adapter_train_subset(dataset, n_cases: int, seed: int):
    """
    Randomly selects n_cases unique case_ids from the dataset (for
    one-shot / few-shot adapter training). Returns train indices and
    val indices (remaining cases).
    """
    if n_cases == -1:
        # use all cases for training, no val subset
        return list(range(len(dataset))), []

    # gather all case_ids with their indices
    case_to_indices: dict = {}
    for i, row in enumerate(dataset.rows):
        cid = row.get("case_id", f"vol_{i}")
        case_to_indices.setdefault(cid, []).append(i)

    all_cases = sorted(case_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(all_cases)

    n_cases = min(n_cases, len(all_cases))
    train_cases = set(all_cases[:n_cases])
    val_cases   = set(all_cases[n_cases:])

    train_idx = [i for c in train_cases for i in case_to_indices[c]]
    val_idx   = [i for c in val_cases   for i in case_to_indices[c]]

    return train_idx, val_idx


# ─────────────────────────────────────────────
# Train / Val loops
# ─────────────────────────────────────────────

def train_one_epoch(adapter, loader, optimizer, seg_loss_fn,
                    align_loss_fn, lambda_align, device, num_classes):
    adapter.train()   # foundation stays in eval via overridden train()
    total_seg   = 0.0
    total_align = 0.0
    total_total = 0.0
    all_dice    = np.zeros(num_classes, dtype=np.float64)
    n_batches   = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, z_target, z_aligned = adapter(images)

        l_seg   = seg_loss_fn(logits, labels)
        l_align = align_loss_fn(z_aligned, labels)
        loss    = l_seg + lambda_align * l_align

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.T_flow.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        dice  = compute_dice_per_class(preds.cpu(), labels.cpu(), num_classes)

        total_seg   += l_seg.item()
        total_align += l_align.item()
        total_total += loss.item()
        all_dice    += dice
        n_batches   += 1

    return (total_total / n_batches,
            total_seg   / n_batches,
            total_align / n_batches,
            all_dice    / n_batches)


@torch.no_grad()
def validate(adapter, loader, seg_loss_fn, align_loss_fn,
             lambda_align, device, num_classes):
    adapter.eval()
    total_seg   = 0.0
    total_align = 0.0
    total_total = 0.0
    all_dice    = np.zeros(num_classes, dtype=np.float64)
    n_batches   = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits, _, z_aligned = adapter(images)

        l_seg   = seg_loss_fn(logits, labels)
        l_align = align_loss_fn(z_aligned, labels)
        loss    = l_seg + lambda_align * l_align

        preds = logits.argmax(dim=1)
        dice  = compute_dice_per_class(preds.cpu(), labels.cpu(), num_classes)

        total_seg   += l_seg.item()
        total_align += l_align.item()
        total_total += loss.item()
        all_dice    += dice
        n_batches   += 1

    return (total_total / n_batches,
            total_seg   / n_batches,
            total_align / n_batches,
            all_dice    / n_batches)


@torch.no_grad()
def validate_foundation(model, loader, criterion, device, num_classes):
    """Direct evaluation of frozen foundation with no adapter — for ablation."""
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
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",              required=True, choices=["2d", "3d"])
    p.add_argument("--config",            required=True)
    p.add_argument("--foundation_ckpt",   required=True,
                   help="Path to best_model.pth from Phase 1")
    p.add_argument("--centroid_cache",    default=None,
                   help="Path to global_centroid.npy from Phase 1. If not provided, pure task loss is used.")
    p.add_argument("--output_dir",        default="outputs/")

    # 2D
    p.add_argument("--slices_csv",        default=None)
    # 3D
    p.add_argument("--volumes_csv",       default=None)
    p.add_argument("--npy_cache_dir",     default=None)

    # Model Selection
    p.add_argument("--model", default="nnunet", choices=list(MODEL_REGISTRY.keys()), help="Must match the model used in Phase 1")
    p.add_argument("--no_adapter", action="store_true", help="Run frozen foundation only, no flow adapter (ablation baseline)")
    p.add_argument("--use_pure_ce", action="store_true", help="Use pure CrossEntropyLoss (like the old CyL-Adapter code) instead of DiceCELoss.")

    # Adapter hyperparams
    p.add_argument("--adapter_train_cases", type=int, default=1,
                   help="Number of target cases for adapter training. -1 = all.")
    p.add_argument("--num_flow_layers",   type=int,   default=3)
    p.add_argument("--epochs",            type=int,   default=100)
    p.add_argument("--batch_size",        type=int,   default=8)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--weight_decay",      type=float, default=1e-5)
    p.add_argument("--lambda_align",      type=float, default=0.5,
                   help="Weight for centroid alignment loss")
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--device",            default="cuda:0")
    p.add_argument("--seed",              type=int,   default=42)

    # Foundation model architecture (must match Phase 1)
    p.add_argument("--base_filters",      type=int,   default=32)
    p.add_argument("--num_stages",        type=int,   default=5)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    exp_name  = cfg["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_suffix = "no_adapter" if args.no_adapter else "adapter"
    run_dir = Path(args.output_dir) / f"phase2_{args.model}_{run_dir_suffix}_{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config_used.yaml")

    print(f"\n{'='*60}")
    print(f"  Phase 2 Adapter Training")
    print(f"  Experiment  : {exp_name}")
    print(f"  Mode        : {args.mode.upper()}")
    print(f"  Target cases: {args.adapter_train_cases} (one-shot=1)")
    print(f"  λ_align     : {args.lambda_align}")
    print(f"  Output      : {run_dir}")
    print(f"{'='*60}\n")

    # ── target dataset (the "test" split in the config IS the target domain) ──
    if args.mode == "2d":
        assert args.slices_csv
        target_ds = Myo2DDataset(args.slices_csv, args.config, split="test")
    else:
        assert args.volumes_csv and args.npy_cache_dir
        target_ds = WholeHeart3DDataset(args.volumes_csv, args.config,
                                        split="test",
                                        npy_cache_dir=args.npy_cache_dir)
    target_ds.summary()
    num_classes = len(target_ds.classes)
    class_names = target_ds.classes

    # ── one-shot / few-shot case selection ──
    train_idx, val_idx = select_adapter_train_subset(
        target_ds, args.adapter_train_cases, args.seed)

    if len(train_idx) == 0:
        raise RuntimeError("No training samples selected — check --adapter_train_cases")

    train_subset = Subset(target_ds, train_idx)
    val_subset   = Subset(target_ds, val_idx) if val_idx else None

    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=len(train_subset) > args.batch_size)
    val_loader   = DataLoader(val_subset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True) if val_subset else None

    print(f"  Adapter train slices/volumes : {len(train_subset)}")
    print(f"  Adapter val   slices/volumes : {len(val_subset) if val_subset else 0}\n")

    # ── load foundation model ──

    # validate model choice
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(MODEL_REGISTRY.keys())}")

    # validate checkpoint path
    if not os.path.exists(args.foundation_ckpt):
        raise FileNotFoundError(
            f"Foundation checkpoint not found: {args.foundation_ckpt}\n"
            f"Run Phase 1 first with --model {args.model}"
        )

    # load and verify the checkpoint was trained with the same model
    ckpt = torch.load(args.foundation_ckpt, map_location="cpu")
    ckpt_model = ckpt.get("args", {}).get("model", "unknown")
    if ckpt_model != "unknown" and ckpt_model != args.model:
        raise ValueError(
            f"Model mismatch: checkpoint was trained with '{ckpt_model}' "
            f"but --model is '{args.model}'. Use the correct --model flag."
        )
    dim = 2 if args.mode == "2d" else 3
    model_cls = MODEL_REGISTRY[args.model]
    foundation = model_cls(
        dim=dim, in_channels=1, num_classes=num_classes,
        base_filters=args.base_filters, num_stages=args.num_stages
    )

    try:
        foundation.load_state_dict(ckpt["model_state"])
        print(f"  Loaded {args.model} foundation from: {args.foundation_ckpt}")
        print(f"  Phase 1 best val Dice: {ckpt.get('val_dice', 'N/A')}")
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load weights into {args.model} — architecture mismatch.\n"
            f"Check --base_filters and --num_stages match Phase 1 settings.\n"
            f"Original error: {e}"
        )

    if args.no_adapter:
        print("  Mode: NO ADAPTER (frozen foundation only — ablation baseline)")
        foundation = foundation.to(device)
        for param in foundation.parameters():
            param.requires_grad = False
        foundation.eval()

        if args.use_pure_ce:
            seg_loss_fn = nn.CrossEntropyLoss()
        else:
            seg_loss_fn = DiceCELoss(num_classes=num_classes)

        eval_rows = []
        train_loss, train_dice = validate_foundation(
            foundation, train_loader, seg_loss_fn, device, num_classes
        )
        eval_rows.append(("train", train_loss, train_dice))

        if val_loader is not None:
            val_loss, val_dice = validate_foundation(
                foundation, val_loader, seg_loss_fn, device, num_classes
            )
            eval_rows.append(("val", val_loss, val_dice))
        else:
            val_loss, val_dice = train_loss, train_dice

        fieldnames = (
            ["epoch", "phase", "loss_total", "loss_seg", "loss_align",
             "mean_dice", "lr", "epoch_time_s"]
            + [f"dice_{c}" for c in class_names]
        )
        logger = CSVLogger(run_dir / "no_adapter_log.csv", fieldnames)

        for phase, loss, dice in eval_rows:
            row = {
                "epoch": 0,
                "phase": phase,
                "loss_total": round(float(loss), 6),
                "loss_seg": round(float(loss), 6),
                "loss_align": 0.0,
                "mean_dice": round(float(np.mean(dice[1:])), 6),
                "lr": 0.0,
                "epoch_time_s": 0.0,
            }
            for i, cname in enumerate(class_names):
                row[f"dice_{cname}"] = round(float(dice[i]), 6)
            logger.log(row)

        summary = {
            "mode": args.mode,
            "model": args.model,
            "no_adapter": True,
            "foundation_ckpt": args.foundation_ckpt,
            "train_loss": float(train_loss),
            "train_mean_dice": float(np.mean(train_dice[1:])),
            "val_loss": float(val_loss),
            "val_mean_dice": float(np.mean(val_dice[1:])),
            "config": cfg,
            "args": vars(args),
        }
        with open(run_dir / "no_adapter_summary.yaml", "w") as f:
            yaml.dump(summary, f)

        print(f"  Frozen foundation train Dice : {float(np.mean(train_dice[1:])):.4f}")
        print(f"  Frozen foundation val   Dice : {float(np.mean(val_dice[1:])):.4f}")
        print(f"  All outputs saved to        : {run_dir}\n")
        return

    # bottleneck channels = base_filters * 2^(num_stages-1)
    bottleneck_ch = args.base_filters * (2 ** (args.num_stages - 1))
    print(f"  Bottleneck channels : {bottleneck_ch}")

    adapter = CyLAdapterModel(
        foundation_model=foundation,
        dim=dim,
        # bottleneck_channels=bottleneck_ch,
        num_flow_layers=args.num_flow_layers,
    ).to(device)

    n_flow_params = sum(p.numel() for p in adapter.T_flow.parameters())
    print(f"  Flow params (trainable): {n_flow_params:,}\n")

    # ── load centroids and setup loss ──
    if args.use_pure_ce:
        print("  Using PURE CrossEntropyLoss for segmentation (old CyL-Adapter method).")
        seg_loss_fn = nn.CrossEntropyLoss()
    else:
        print("  Using DiceCELoss for segmentation.")
        seg_loss_fn = DiceCELoss(num_classes=num_classes)
    
    if args.centroid_cache and os.path.exists(args.centroid_cache):
        centroid_data = np.load(args.centroid_cache)
        print(f"  Loaded global centroid from: {args.centroid_cache}")
        align_loss_fn = GlobalAlignmentLoss(centroid_data, device)
    else:
        print("  No centroid provided. Using PURE task loss (lambda_align = 0.0)")
        args.lambda_align = 0.0
        align_loss_fn = lambda z, y: torch.tensor(0.0, device=device)

    # ── optimiser — ONLY flow params ──
    optimizer = Adam(adapter.T_flow.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── CSV logger ──
    fieldnames = (
        ["epoch", "phase", "loss_total", "loss_seg", "loss_align",
         "mean_dice", "lr", "epoch_time_s"]
        + [f"dice_{c}" for c in class_names]
    )
    logger = CSVLogger(run_dir / "adapter_log.csv", fieldnames)

    # save args
    with open(run_dir / "train_args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    # ── training loop ──
    best_val_dice  = -1.0
    best_epoch     = 0
    has_val        = val_loader is not None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        (tr_loss, tr_seg, tr_align,
         tr_dice) = train_one_epoch(
            adapter, train_loader, optimizer,
            seg_loss_fn, align_loss_fn, args.lambda_align,
            device, num_classes)

        if has_val:
            (vl_loss, vl_seg, vl_align,
             vl_dice) = validate(
                adapter, val_loader,
                seg_loss_fn, align_loss_fn, args.lambda_align,
                device, num_classes)
            monitor_dice = float(np.mean(vl_dice[1:]))
        else:
            # no val cases (pure one-shot with only 1 case total):
            # monitor training dice instead
            vl_loss = vl_seg = vl_align = tr_loss
            vl_dice = tr_dice
            monitor_dice = float(np.mean(tr_dice[1:]))

        scheduler.step()
        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # ── log ──
        for phase, loss, seg, align, dice in [
            ("train", tr_loss, tr_seg, tr_align, tr_dice),
            ("val",   vl_loss, vl_seg, vl_align, vl_dice),
        ]:
            row = {
                "epoch":        epoch,
                "phase":        phase,
                "loss_total":   round(loss,  6),
                "loss_seg":     round(seg,   6),
                "loss_align":   round(align, 6),
                "mean_dice":    round(float(np.mean(dice[1:])), 6),
                "lr":           round(current_lr, 8),
                "epoch_time_s": round(epoch_time, 2),
            }
            for i, cname in enumerate(class_names):
                row[f"dice_{cname}"] = round(float(dice[i]), 6)
            logger.log(row)

        # ── checkpoint: last ──
        torch.save({
            "epoch":            epoch,
            "flow_state":       adapter.T_flow.state_dict(),
            "foundation_ckpt":  args.foundation_ckpt,
            "val_dice":         monitor_dice,
            "config":           cfg,
            "args":             vars(args),
        }, run_dir / "last_adapter.pth")

        # ── checkpoint: best ──
        if monitor_dice > best_val_dice:
            best_val_dice = monitor_dice
            best_epoch    = epoch
            torch.save({
                "epoch":            epoch,
                "flow_state":       adapter.T_flow.state_dict(),
                "foundation_ckpt":  args.foundation_ckpt,
                "val_dice":         monitor_dice,
                "config":           cfg,
                "args":             vars(args),
            }, run_dir / "best_adapter.pth")

        # ── console ──
        if epoch % 10 == 0 or epoch == 1:
            tr_mean = float(np.mean(tr_dice[1:]))
            dice_str = "  ".join(
                f"{c}:{d:.3f}" for c, d in zip(class_names[1:], vl_dice[1:]))
            print(f"[{epoch:>4d}/{args.epochs}] "
                  f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
                  f"tr_dice={tr_mean:.4f}  vl_dice={monitor_dice:.4f}  "
                  f"align={tr_align:.4f}  [{dice_str}]  "
                  f"lr={current_lr:.6f}  t={epoch_time:.1f}s")

    print(f"\n  Best adapter val Dice : {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"  All outputs saved to  : {run_dir}\n")


if __name__ == "__main__":
    main()
