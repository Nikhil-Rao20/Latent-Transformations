"""
wholeheart3d_dataset.py

The single Dataset class used for EVERY 3D Whole Heart experiment.
Same resolution pattern as Myo2DDataset: read YAML -> filter master CSV
by (center, modality) -> load.

Unlike the 2D case, we do NOT pre-extract anything to a flat cache here.
Volumes are resampled to target_spacing and cached as .npy ONCE per case
on first access (lazy disk cache keyed by case_id), since repeated NIfTI
loads + resampling are the actual bottleneck, not slicing. After first
epoch, every subsequent access reads the npy cache.

Training mode returns a single random foreground-biased patch per call
(standard nnU-Net-style sampling: 1/3 of patches purely random, 2/3
forced to contain at least one foreground voxel). Use a full sliding-
window inference function separately at eval time (not in this class).
"""

import csv
import yaml
import hashlib
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom


class WholeHeart3DDataset(Dataset):
    def __init__(self, volumes_csv: str, config_path: str, split: str,
                 npy_cache_dir: str, fg_oversample_prob: float = 0.66,
                 transform=None):
        assert split in ("train", "test")
        self.split = split
        self.transform = transform
        self.fg_oversample_prob = fg_oversample_prob

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        with open(volumes_csv) as f:
            all_volumes = list(csv.DictReader(f))

        self.classes = self.config.get("classes")
        self.patch_size = tuple(self.config.get("patch_size", [128, 128, 128]))
        self.target_spacing = tuple(self.config.get("target_spacing", [1.0, 1.0, 1.0]))

        self.cache_dir = Path(npy_cache_dir)
        (self.cache_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "labels").mkdir(parents=True, exist_ok=True)

        self.rows = self._resolve_rows(all_volumes)
        if len(self.rows) == 0:
            raise RuntimeError(
                f"No volumes matched config '{self.config['name']}' split='{split}'."
            )

    def _resolve_rows(self, all_volumes):
        wanted = self.config[self.split]  # list of {center, modality}
        wanted_pairs = {(w["center"], w["modality"]) for w in wanted}
        return [r for r in all_volumes if (r["center"], r["modality"]) in wanted_pairs]

    def __len__(self):
        return len(self.rows)

    # ---------- caching ----------

    def _cache_paths(self, row):
        key = f"{row['center']}_{row['modality']}_{row['case_id']}_{self.target_spacing}"
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        img_p = self.cache_dir / "images" / f"{row['case_id']}_{h}.npy"
        lbl_p = self.cache_dir / "labels" / f"{row['case_id']}_{h}.npy"
        return img_p, lbl_p

    def _load_or_build_cache(self, row):
        img_p, lbl_p = self._cache_paths(row)
        if img_p.exists() and lbl_p.exists():
            return np.load(img_p), np.load(lbl_p)

        img_nii = nib.load(row["image_path"])
        lbl_nii = nib.load(row["label_path"])
        img = img_nii.get_fdata().astype(np.float32)
        lbl = lbl_nii.get_fdata().astype(np.uint8)

        orig_spacing = img_nii.header.get_zooms()[:3]
        zoom_factors = [o / t for o, t in zip(orig_spacing, self.target_spacing)]
        img = zoom(img, zoom_factors, order=3)             # cubic for image
        lbl = zoom(lbl, zoom_factors, order=0)              # nearest-neighbor for label

        # intensity normalization
        if row["modality"] == "CT":
            img = np.clip(img, -1000, 1000)
            img = (img + 1000) / 2000.0
        else:  # MRI
            p1, p99 = np.percentile(img, [1, 99])
            img = np.clip(img, p1, p99)
            mean, std = img.mean(), img.std()
            img = (img - mean) / (std + 1e-6)

        img = img.astype(np.float32)
        lbl = lbl.astype(np.uint8)
        np.save(img_p, img)
        np.save(lbl_p, lbl)
        return img, lbl

    # ---------- patch sampling ----------

    def _sample_patch(self, img, lbl):
        H, W, D = img.shape
        ph, pw, pd = self.patch_size

        force_fg = np.random.rand() < self.fg_oversample_prob
        fg_coords = None
        if force_fg:
            fg_coords = np.argwhere(lbl > 0)

        if force_fg and len(fg_coords) > 0:
            cz, cy, cx = fg_coords[np.random.randint(len(fg_coords))]
        else:
            cz = np.random.randint(0, max(H, 1))
            cy = np.random.randint(0, max(W, 1))
            cx = np.random.randint(0, max(D, 1))

        def clamp_start(center, patch, full):
            start = center - patch // 2
            start = max(0, min(start, full - patch))
            return start if full >= patch else 0

        z0 = clamp_start(cz, ph, H)
        y0 = clamp_start(cy, pw, W)
        x0 = clamp_start(cx, pd, D)

        img_patch = self._pad_or_crop(img[z0:z0+ph, y0:y0+pw, x0:x0+pd], self.patch_size)
        lbl_patch = self._pad_or_crop(lbl[z0:z0+ph, y0:y0+pw, x0:x0+pd], self.patch_size, is_label=True)
        return img_patch, lbl_patch

    @staticmethod
    def _pad_or_crop(arr, target_shape, is_label=False):
        pad_width = [(0, max(0, t - s)) for s, t in zip(arr.shape, target_shape)]
        if any(p[1] > 0 for p in pad_width):
            arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
        slices = tuple(slice(0, t) for t in target_shape)
        return arr[slices]

    def __getitem__(self, idx):
        row = self.rows[idx]
        img, lbl = self._load_or_build_cache(row)

        if self.split == "train":
            img_patch, lbl_patch = self._sample_patch(img, lbl)
        else:
            # test/eval: return the FULL resampled volume; use a separate
            # sliding-window inference function at eval time, not patches here
            img_patch, lbl_patch = img, lbl

        image = torch.from_numpy(img_patch).unsqueeze(0).float()  # (1, H, W, D)
        label = torch.from_numpy(lbl_patch).long()                 # (H, W, D)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return {
            "image": image,
            "label": label,
            "center": row["center"],
            "modality": row["modality"],
            "case_id": row["case_id"],
        }

    def summary(self):
        from collections import Counter
        counts = Counter((r["center"], r["modality"]) for r in self.rows)
        print(f"[{self.config['name']}] split={self.split} total_volumes={len(self.rows)}")
        for (c, m), n in sorted(counts.items()):
            print(f"   center={c:6s} modality={m:6s} volumes={n}")
