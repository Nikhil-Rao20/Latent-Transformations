"""
extract_slices_myo2d.py

One-time preprocessing: reads metadata_myo2d_volumes.csv (output of
index_myo2d.py), loads every NIfTI volume, and writes each depth slice
out as a raw float32 .npy pair (image, label) into a flat cache dir.
Also writes metadata_myo2d_slices.csv — one row per slice — which is
what the actual Dataset class reads from during training.

Label remapping happens HERE, once, not at every __getitem__ call.
We keep only the three finalized classes: background=0, LV=1, Myo=2, Scar=3.
You MUST fill in LABEL_MAP below with the organizers' actual integer
codes per dataset (MyoPS vs CineMyoPS may use different codes — check
the CARE2026 documentation / a sample volume's unique() values).

Run once:
    python DATALOADERS/extract_slices_myo2d.py \
        --volumes_csv Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_volumes.csv \
        --cache_dir Datasets/CURE2026-Left-Myocardium/2D_NPY_Cache \
        --out Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_slices.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm

# ---------------------------------------------------------------------
# IMPORTANT: verify these against the actual CARE2026 label conventions
# before running on the full dataset. Different challenge tracks often
# use different raw integer codes for the same anatomical class.
# Run `python check_label_values.py` (see below) on a few volumes first.
# ---------------------------------------------------------------------
LABEL_MAP = {
    # raw_value: remapped_value
    0: 0,   # background
    500: 1,   # LV
    200: 2,   # Myo
    2221: 3,   # Scar
    # add edema / RV codes here if you later expand finalized labels
}
FINAL_CLASSES = {0: "background", 1: "LV", 2: "Myo", 3: "Scar"}


def remap_label(label_slice: np.ndarray) -> np.ndarray:
    out = np.zeros_like(label_slice, dtype=np.uint8)
    for raw_val, new_val in LABEL_MAP.items():
        out[label_slice == raw_val] = new_val
    return out


def normalize_image(img_slice: np.ndarray) -> np.ndarray:
    # Per-slice z-score normalization, robust to outliers via percentile clip.
    p1, p99 = np.percentile(img_slice, [1, 99])
    img_slice = np.clip(img_slice, p1, p99)
    mean, std = img_slice.mean(), img_slice.std()
    if std < 1e-6:
        return np.zeros_like(img_slice, dtype=np.float32)
    return ((img_slice - mean) / std).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes_csv", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out", default="metadata_myo2d_slices.csv")
    ap.add_argument("--min_fg_pixels", type=int, default=0,
                     help="Skip slices with fewer than this many foreground "
                          "pixels (0 = keep all slices including empty ones)")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    img_dir = cache_dir / "images"
    lbl_dir = cache_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(args.volumes_csv) as f:
        volumes = list(csv.DictReader(f))

    slice_rows = []
    for vol in tqdm(volumes, desc="Extracting volumes"):
        try:
            img_nii = nib.load(vol["image_path"])
            lbl_nii = nib.load(vol["label_path"])
        except Exception as e:
            print(f"  SKIP (load error) {vol['image_path']}: {e}")
            continue

        img_data = img_nii.get_fdata().astype(np.float32)
        lbl_data = lbl_nii.get_fdata().astype(np.int16)

        if img_data.shape != lbl_data.shape:
            print(f"  SKIP (shape mismatch) {vol['case_id']} {vol['sequence']}: "
                  f"{img_data.shape} vs {lbl_data.shape}")
            continue

        n_slices = img_data.shape[2]
        for z in range(n_slices):
            img_slice = normalize_image(img_data[:, :, z])
            lbl_slice = remap_label(lbl_data[:, :, z])

            if args.min_fg_pixels > 0 and (lbl_slice > 0).sum() < args.min_fg_pixels:
                continue

            slice_id = f"{vol['center']}_{vol['sequence']}_{vol['case_id']}_z{z:03d}"
            img_out = img_dir / f"{slice_id}.npy"
            lbl_out = lbl_dir / f"{slice_id}.npy"
            np.save(img_out, img_slice)
            np.save(lbl_out, lbl_slice)

            slice_rows.append({
                "slice_id": slice_id,
                "center": vol["center"],
                "sequence": vol["sequence"],
                "case_id": vol["case_id"],
                "z_index": z,
                "image_npy": str(img_out),
                "label_npy": str(lbl_out),
                "has_fg": int((lbl_slice > 0).any()),
            })

    fieldnames = ["slice_id", "center", "sequence", "case_id", "z_index",
                  "image_npy", "label_npy", "has_fg"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(slice_rows)

    print(f"Extracted {len(slice_rows)} slices -> {args.out}")
    print(f"npy cache at: {cache_dir}")


if __name__ == "__main__":
    main()
