"""
index_myo2d.py

One-time script: walks the raw CURE2026-Left-Myocardium directory tree,
finds every case, and writes a master metadata CSV — one row per
(center, sequence, case) volume. This CSV is the single source of truth
that the slice-extraction script and the Dataset class both read from.

Run once after downloading the dataset:
    python index_myo2d.py --root Datasets/CURE2026-Left-Myocardium --out Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_volumes.csv
"""

import argparse
import os
import re
import csv
from pathlib import Path

# Sequence file suffixes we look for inside each case folder / cine folder.
# Add new ones here if the organizers add a sequence later — nothing else
# in the pipeline needs to change.
SEQUENCE_SUFFIXES = {
    "LGE": "_LGE.nii.gz",
    "T2": "_T2.nii.gz",
    "bSSFP": "_C0.nii.gz",   # CURE naming: C0 == bSSFP/cine-balanced-SSFP
    "Cine": "_Cine.nii.gz",
}
LABEL_SUFFIX = "_gd.nii.gz"


def find_case_id(filename: str) -> str:
    # filenames look like CaseXXXX_LGE.nii.gz -> CaseXXXX
    m = re.match(r"(Case\d+)", filename)
    return m.group(1) if m else None


def index_cine_centers(myops_cine_root: Path, rows: list):
    """center_alpha / center_beta — flat folder, no per-case subfolder."""
    for center_dir in sorted(myops_cine_root.iterdir()):
        if not center_dir.is_dir():
            continue
        center_name = center_dir.name.replace("center_", "").capitalize()  # alpha -> Alpha
        files = list(center_dir.glob("*.nii.gz"))
        case_ids = sorted({find_case_id(f.name) for f in files if find_case_id(f.name)})
        for case_id in case_ids:
            img_path = center_dir / f"{case_id}_Cine.nii.gz"
            label_path = center_dir / f"{case_id}{LABEL_SUFFIX}"
            if img_path.exists() and label_path.exists():
                rows.append({
                    "center": center_name,
                    "sequence": "Cine",
                    "case_id": case_id,
                    "image_path": str(img_path),
                    "label_path": str(label_path),
                })


def index_myops_centers(myops_root: Path, rows: list):
    """CenterA..H — each case has its own subfolder with 1-3 sequence files."""
    for center_dir in sorted(myops_root.iterdir()):
        if not center_dir.is_dir():
            continue
        center_name = center_dir.name.replace("Center", "")  # CenterA -> A
        for case_dir in sorted(center_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            label_path = case_dir / f"{case_id}{LABEL_SUFFIX}"
            if not label_path.exists():
                continue
            for seq_name, suffix in SEQUENCE_SUFFIXES.items():
                img_path = case_dir / f"{case_id}{suffix}"
                if img_path.exists():
                    rows.append({
                        "center": center_name,
                        "sequence": seq_name,
                        "case_id": case_id,
                        "image_path": str(img_path),
                        "label_path": str(label_path),
                    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to CURE2026-Left-Myocardium")
    ap.add_argument("--out", default="metadata_myo2d_volumes.csv")
    args = ap.parse_args()

    root = Path(args.root)
    myo_train = root / "Myo_train"
    cine_root = myo_train / "CineMyoPS_train"
    myops_root = myo_train / "MyoPS_train"

    rows = []
    if cine_root.exists():
        index_cine_centers(cine_root, rows)
    if myops_root.exists():
        index_myops_centers(myops_root, rows)

    if not rows:
        raise RuntimeError(f"No cases found under {root} — check --root path / folder names")

    fieldnames = ["center", "sequence", "case_id", "image_path", "label_path"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Indexed {len(rows)} volumes -> {args.out}")
    # quick sanity summary
    from collections import Counter
    counts = Counter((r["center"], r["sequence"]) for r in rows)
    for (center, seq), n in sorted(counts.items()):
        print(f"  center={center:6s} sequence={seq:6s} -> {n} volumes")


if __name__ == "__main__":
    main()
