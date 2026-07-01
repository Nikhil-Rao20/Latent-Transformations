"""
myo2d_dataset.py

The single Dataset class used for EVERY 2D Myocardium experiment.
Nothing in here changes between experiments — only the YAML config path
changes. Resolution logic:

    1. Read the YAML config -> get list of {center, sequence} dicts for
       the requested split ("train" or "test").
    2. Filter the master slice CSV (metadata_myo2d_slices.csv) to rows
       matching those (center, sequence) pairs.
    3. If the config has a `patient_split` block, further filter by
       case_id according to train_pool/test_pool for that split.
    4. Load npy pairs lazily in __getitem__.

Usage:
    train_ds = Myo2DDataset(
        slices_csv="metadata_myo2d_slices.csv",
        config_path="configs/2d/exp_a1_lge_intercenter.yaml",
        split="train",
        transform=train_transforms,
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
"""

import csv
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Myo2DDataset(Dataset):
    def __init__(self, slices_csv: str, config_path: str, split: str,
                 transform=None):
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.split = split
        self.transform = transform

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        with open(slices_csv) as f:
            all_slices = list(csv.DictReader(f))

        self.classes = self.config.get("classes", ["background", "LV", "Myo", "Scar"])
        self.rows = self._resolve_rows(all_slices)

        if len(self.rows) == 0:
            raise RuntimeError(
                f"No slices matched config '{self.config['name']}' split='{split}'. "
                f"Check center/sequence names match metadata CSV exactly."
            )

    def _resolve_rows(self, all_slices):
        wanted = self.config[self.split]  # list of {center, sequence}
        wanted_pairs = {(w["center"], w["sequence"]) for w in wanted}

        patient_split = self.config.get("patient_split")  # optional

        rows = []
        for r in all_slices:
            pair = (r["center"], r["sequence"])
            if pair not in wanted_pairs:
                continue

            if patient_split and r["center"] in patient_split:
                pool_key = "train_pool" if self.split == "train" else "test_pool"
                allowed_cases = set(patient_split[r["center"]].get(pool_key, []))
                if r["case_id"] not in allowed_cases:
                    continue

            rows.append(r)

        max_per_case = self.config.get("max_slices_per_case")
        if max_per_case:
            rows = self._cap_per_case(rows, max_per_case)

        return rows

    @staticmethod
    def _cap_per_case(rows, max_per_case):
        from collections import defaultdict
        by_case = defaultdict(list)
        for r in rows:
            by_case[r["case_id"]].append(r)
        capped = []
        for case_rows in by_case.values():
            capped.extend(case_rows[:max_per_case])
        return capped

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = np.load(row["image_npy"])   # (H, W) float32, already normalized
        label = np.load(row["label_npy"])   # (H, W) uint8, already remapped

        image = torch.from_numpy(image).unsqueeze(0).float()   # (1, H, W)
        label = torch.from_numpy(label).long()                 # (H, W)

        # Resize to 256x256
        image = F.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        label = F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest-exact').squeeze(0).squeeze(0).long()

        if self.transform is not None:
            image, label = self.transform(image, label)

        return {
            "image": image,
            "label": label,
            "center": row["center"],
            "sequence": row["sequence"],
            "case_id": row["case_id"],
            "slice_id": row["slice_id"],
        }

    def summary(self):
        """Quick sanity-check printout — call after construction."""
        from collections import Counter
        counts = Counter((r["center"], r["sequence"]) for r in self.rows)
        print(f"[{self.config['name']}] split={self.split} total_slices={len(self.rows)}")
        for (c, s), n in sorted(counts.items()):
            print(f"   center={c:6s} sequence={s:6s} slices={n}")
