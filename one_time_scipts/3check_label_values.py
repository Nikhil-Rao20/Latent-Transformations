"""
check_label_values.py

Run this BEFORE extract_slices_myo2d.py. Different CARE2026 sub-datasets
(MyoPS vs CineMyoPS) and even different centers can use different raw
integer codes for the same anatomical class. This script prints the
unique label values found per (center, sequence) so you can fill in
LABEL_MAP correctly instead of guessing.

Usage:
    python check_label_values.py --volumes_csv metadata_myo2d_volumes.csv
"""

import argparse
import csv
from collections import defaultdict

import nibabel as nib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes_csv", required=True)
    ap.add_argument("--sample_per_group", type=int, default=2,
                     help="How many volumes to check per (center, sequence) group")
    args = ap.parse_args()

    with open(args.volumes_csv) as f:
        volumes = list(csv.DictReader(f))

    groups = defaultdict(list)
    for v in volumes:
        groups[(v["center"], v["sequence"])].append(v)

    print(f"{'center':8s} {'sequence':8s} {'unique label values found'}")
    print("-" * 60)
    for (center, seq), vols in sorted(groups.items()):
        all_values = set()
        for v in vols[:args.sample_per_group]:
            lbl = nib.load(v["label_path"]).get_fdata()
            all_values |= set(np.unique(lbl).astype(int).tolist())
        print(f"{center:8s} {seq:8s} {sorted(all_values)}")


if __name__ == "__main__":
    main()
