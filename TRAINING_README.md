# Training Guide


## Commands

**Phase 1:**
```bash
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
```

**Phase 2 — LGE target (inter-center shift):**
```bash
python training/train_phase2_adapter.py \
    --mode 2d \
    --slices_csv data/metadata_myo2d_slices.csv \
    --config configs/2d/exp_a1_lge_intercenter.yaml \
    --foundation_ckpt outputs/phase1_exp_a1_lge_intercenter_<timestamp>/best_model.pth \
    --centroid_cache outputs/phase1_exp_a1_lge_intercenter_<timestamp>/centroid_cache.npz \
    --output_dir outputs/ \
    --adapter_train_cases 1 \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --lambda_align 0.5 \
    --num_workers 4 \
    --device cuda:0
```

For T2 and bSSFP targets you need separate configs since they have different `test:` blocks. Create two more configs:

`configs/2d/exp_a1_lge_to_t2.yaml` — same train block (LGE A,B,C), test block changed to T2(B,C):
```yaml
name: exp_a1_lge_to_t2
train:
  - {center: A, sequence: LGE}
  - {center: B, sequence: LGE}
  - {center: C, sequence: LGE}
test:
  - {center: B, sequence: T2}
  - {center: C, sequence: T2}
classes: [background, LV, Myo, Scar]
```

`configs/2d/exp_a1_lge_to_bssfp.yaml` — same but test is bSSFP(B,C).

Then Phase 2 for T2:
```bash
python training/train_phase2_adapter.py \
    --mode 2d \
    --slices_csv data/metadata_myo2d_slices.csv \
    --config configs/2d/exp_a1_lge_to_t2.yaml \
    --foundation_ckpt outputs/phase1_exp_a1_lge_intercenter_<timestamp>/best_model.pth \
    --centroid_cache outputs/phase1_exp_a1_lge_intercenter_<timestamp>/centroid_cache.npz \
    --output_dir outputs/ \
    --adapter_train_cases 1 \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-4 \
    --lambda_align 0.5 \
    --num_workers 4 \
    --device cuda:0
```

And for bSSFP same command with `--config configs/2d/exp_a1_lge_to_bssfp.yaml`.

---

## Output structure for your dummy experiment

**Experiment:** Train LGE(A,B,C) → Test LGE(E,F,G,H) + T2(B,C) + bSSFP(B,C)

---

### Phase 1 — one run total

```
outputs/
└── phase1_exp_a1_lge_intercenter_20250702_143022/
    ├── best_model.pth          # best val Dice weights (what Phase 2 loads)
    ├── last_model.pth          # final epoch weights
    ├── centroid_cache.npz      # μ_LV, μ_Myo, μ_Scar (what Phase 2 loads)
    ├── config_used.yaml        # copy of exp_a1_lge_intercenter.yaml
    └── train_log.csv           # columns below, 2 rows per epoch (train + val)
```

`train_log.csv` columns:
```
epoch | phase | loss | mean_dice | lr | dice_background | dice_LV | dice_Myo | dice_Scar | epoch_time_s
```

---

### Phase 2 — three separate runs (one per target domain)

```
outputs/
├── phase2_exp_a1_lge_intercenter_LGE_target_20250702_160000/
│   ├── best_adapter.pth        # flow weights for LGE→LGE shift
│   ├── last_adapter.pth
│   ├── adapter_log.csv
│   ├── config_used.yaml
│   └── train_args.yaml
│
├── phase2_exp_a1_lge_intercenter_T2_target_20250702_161500/
│   ├── best_adapter.pth        # flow weights for LGE→T2 shift
│   ├── last_adapter.pth
│   ├── adapter_log.csv
│   ├── config_used.yaml
│   └── train_args.yaml
│
└── phase2_exp_a1_lge_intercenter_bSSFP_target_20250702_163000/
    ├── best_adapter.pth        # flow weights for LGE→bSSFP shift
    ├── last_adapter.pth
    ├── adapter_log.csv
    ├── config_used.yaml
    └── train_args.yaml
```

`adapter_log.csv` columns:
```
epoch | phase | loss_total | loss_seg | loss_align | mean_dice | lr | dice_background | dice_LV | dice_Myo | dice_Scar | epoch_time_s
```

---