"""
make_patient_split.py

Generates a reproducible patient-level train/test split per center, used
to fill in the `patient_split` block in any exp_b*.yaml config. This
exists as a separate, run-once script (not done randomly inside the
Dataset class) so that the same split is reused every time that
experiment is trained — required for fair comparison across model
variants and reruns.

Usage:
    python DATALOADERS/make_patient_split.py --volumes_csv Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_volumes.csv \
        --centers B C --test_frac 0.2 --seed 42
"""

import argparse
import csv
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes_csv", required=True)
    ap.add_argument("--centers", nargs="+", required=True)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.volumes_csv) as f:
        volumes = list(csv.DictReader(f))

    case_ids_per_center = defaultdict(set)
    for v in volumes:
        if v["center"] in args.centers:
            case_ids_per_center[v["center"]].add(v["case_id"])

    rng = random.Random(args.seed)
    print("patient_split:")
    for center in args.centers:
        case_ids = sorted(case_ids_per_center[center])
        rng.shuffle(case_ids)
        n_test = max(1, int(len(case_ids) * args.test_frac))
        test_pool = sorted(case_ids[:n_test])
        train_pool = sorted(case_ids[n_test:])
        print(f"  {center}:")
        print(f"    train_pool: {train_pool}")
        print(f"    test_pool: {test_pool}")


if __name__ == "__main__":
    main()
