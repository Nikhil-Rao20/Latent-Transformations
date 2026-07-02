# Phase 2 Adaptation Update: The Global Centroid Shift

This document serves as a historical record and guide for the recent updates made to the Phase 1 and Phase 2 training pipelines. It explains the transition away from class-wise centroids toward a robust global centroid anchor, and provides the exact commands needed to run the ablation study on your other systems.

## 1. The Problem: Collapsing Dice Scores

In our initial runs on the target domain, we observed that using a **class-conditional centroid MSE loss** during Phase 2 caused the validation Dice score to plummet from `~0.30` down to `0.04`. 

### Why did this happen?
In semantic segmentation, the latent bottleneck `Z` is a spatial tensor (e.g., `B x 512 x 8 x 8`). Each "pixel" in that `8x8` bottleneck represents a massive `32x32` patch in the original image. Therefore, a single spatial coordinate in the latent space contains features for multiple mixed classes (e.g., Background, LV, and Myocardium all in one pixel). 

By attempting to force an `8x8` spatial tensor toward single, isolated class centroids based on label masks, the `CentroidAlignmentLoss` inadvertently ripped apart the spatial topology of the heart. The Normalizing Flow became confused, collapsing the anatomy into a flat featureless blob.

## 2. The Solution: Reverting to the Global Anchor

During our previous, highly successful Cine-to-LGE domain adaptation experiments (which achieved a `0.72` Dice score on a single shot), we did **not** use class-wise centroids. Instead, we relied heavily on the **Task Loss** (Cross-Entropy/Dice passing back through the frozen decoder) and a **Single Global Centroid**.

### What is the Global Centroid?
Instead of slicing the latent space by class, we take the average of the *entire latent space* across all spatial dimensions, resulting in a single vector representing "the average channel activations of a Source Image." 

When we apply a loss against this global anchor, it tells the Normalizing Flow: *"Translate these features however you need to for the segmentation loss, but make sure the **overall average** feature intensity looks like the Source Domain."* This prevents the Flow from warping the features into extreme, unstable values while preserving the spatial topology perfectly.

## 3. Architectural Updates

To implement this robust strategy and allow for a clean ablation study, the following updates were deployed:

### Phase 1: `train_phase1_foundation.py`
- **Global Extraction:** The post-training centroid calculation was modified to compute a single `global_centroid.npy` instead of 4 class-wise arrays.
- **Fast Extraction Mode:** Added an `--extract_centroid_only` flag. If passed alongside a `--checkpoint_path`, the script skips all training and immediately computes the global centroid. This means **you do not have to retrain your existing Foundation Models!**

### Phase 2: `train_phase2_adapter.py`
- **Optional Anchors:** The `--centroid_cache` argument is now optional. If omitted, the adapter will train using **Pure Task Loss** (`lambda_align = 0.0`).
- **The Combo Loss (`GlobalAlignmentLoss`):** We replaced the standard MSE loss with a powerful combination loss: `MSE + L1 + Cosine Similarity`. 
  - **MSE:** Aligns raw magnitudes.
  - **L1:** Provides robust alignment that ignores outliers.
  - **Cosine:** Perfectly aligns the high-dimensional vector *direction*. 

---

## 4. Execution Guide (Ablation Study)

You can run these commands directly on your other systems. 

*(Note: Replace `outputs_exp_a1/.../best_model.pth` with your actual trained Phase 1 checkpoint path).*

### Step 1: Extract the Global Centroid
Run this to generate `global_centroid.npy` in your existing checkpoint directory. It takes ~1 minute.
```bash
python training/train_phase1_foundation.py --mode 2d \
  --slices_csv Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_slices.csv \
  --config configs/exp_a1_lge_intercenter.yaml \
  --extract_centroid_only \
  --checkpoint_path outputs_exp_a1/.../best_model.pth
```

### Step 2: Run Experiment A (Pure Task Loss)
Train the adapter using *only* the Cross-Entropy/Dice loss. (Notice there is no `--centroid_cache` flag here).
```bash
python training/train_phase2_adapter.py --mode 2d \
  --slices_csv Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_slices.csv \
  --config configs/exp_a1_lge_intercenter.yaml \
  --foundation_ckpt outputs_exp_a1/.../best_model.pth \
  --output_dir outputs_exp_a1/ \
  --adapter_train_cases 1 --epochs 100 --batch_size 8 --lr 1e-4 --num_worker 4 --device cuda:0
```

### Step 3: Run Experiment B (Combo Global Anchor)
Train the adapter using the Task Loss + the Combo Global Anchor loss.
```bash
python training/train_phase2_adapter.py --mode 2d \
  --slices_csv Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_slices.csv \
  --config configs/exp_a1_lge_intercenter.yaml \
  --foundation_ckpt outputs_exp_a1/.../best_model.pth \
  --centroid_cache outputs_exp_a1/.../global_centroid.npy \
  --lambda_align 0.5 \
  --output_dir outputs_exp_a1/ \
  --adapter_train_cases 1 --epochs 100 --batch_size 8 --lr 1e-4 --num_worker 4 --device cuda:0
```
