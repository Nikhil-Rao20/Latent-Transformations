"""
index_wholeheart3d.py

One-time script: walks the raw CURE2026-Whole-Heart directory and writes
a master metadata CSV — one row per case volume, with center and
modality columns. Same role as index_myo2d.py but for the 3D dataset
(no slicing here — these stay full volumes).

Folder names in the raw download don't matter for downstream experiment
logic (we ignore "_train" suffixes etc.) — only center letter + modality
matter, both of which we parse from the folder name itself.

Run once:
    python DATALOADERS/index_wholeheart3d.py --root Datasets/CURE2026-Whole-Heart --out Datasets/CURE2026-Whole-Heart/Metadata/metadata_wholeheart3d.csv
"""

import argparse
import re
import csv
from pathlib import Path


def parse_center_modality(folder_name: str):
    """
    'A ct_train' -> ('A', 'CT')
    'C and D mr_train' -> ('C&D', 'MRI')
    'E mr_train' -> ('E', 'MRI')
    """
    name = folder_name.lower()
    modality = "CT" if "ct" in name else ("MRI" if "mr" in name else "UNKNOWN")

    if "and" in name:
        # "c and d mr_train" -> "C&D"
        letters = re.findall(r"\b([a-z])\s+and\s+([a-z])\b", name)
        if letters:
            a, b = letters[0]
            return f"{a.upper()}&{b.upper()}", modality

    m = re.match(r"^([a-z])\s", folder_name.strip(), re.IGNORECASE)
    center = m.group(1).upper() if m else folder_name.split()[0]
    return center, modality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to CURE2026-Whole-Heart")
    ap.add_argument("--out", default="metadata_wholeheart3d.csv")
    args = ap.parse_args()

    root = Path(args.root)
    train_root = root / "Wholeheart_Train_Dataset"

    rows = []
    for center_dir in sorted(train_root.iterdir()):
        if not center_dir.is_dir():
            continue
        center, modality = parse_center_modality(center_dir.name)

        image_files = sorted(center_dir.glob("*_image.nii.gz"))
        for img_path in image_files:
            case_id = img_path.name.replace("_image.nii.gz", "")
            label_path = center_dir / f"{case_id}_label.nii.gz"
            if not label_path.exists():
                print(f"  WARN: missing label for {case_id} in {center_dir.name}")
                continue
            rows.append({
                "center": center,
                "modality": modality,
                "case_id": case_id,
                "image_path": str(img_path),
                "label_path": str(label_path),
            })

    fieldnames = ["center", "modality", "case_id", "image_path", "label_path"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Indexed {len(rows)} volumes -> {args.out}")
    from collections import Counter
    counts = Counter((r["center"], r["modality"]) for r in rows)
    for (center, mod), n in sorted(counts.items()):
        print(f"  center={center:6s} modality={mod:6s} -> {n} volumes")


if __name__ == "__main__":
    main()
