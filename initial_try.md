(nikhil-conda) nikhil@tanuh:~/Desktop/Research Works/Papers/Latent-Transformations$ python training/train_phase1_foundation.py --mode 2d --slices_csv Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_slices.csv --config configs/exp_a1_lge_intercenter.yaml --output_dir outputs_exp_a1 --epochs 50 --batc
h_size 16 --lr 1e-3 --num_workers 4 --device cuda:0 --model nnunet

============================================================
  Phase 1 Foundation Training
  Experiment : exp_a1_lge_intercenter
  Mode       : 2D
  Output     : outputs_exp_a1/phase1_nnunet_exp_a1_lge_intercenter_20260702_175324
============================================================

[exp_a1_lge_intercenter] split=train total_slices=1764
   center=A      sequence=LGE    slices=1405
   center=B      sequence=LGE    slices=185
   center=C      sequence=LGE    slices=174
  Train samples : 1500
  Val   samples : 264
  Classes (4): ['background', 'LV', 'Myo', 'Scar']

  Model params  : 10,025,124
[   1/50] train_loss=0.6889  val_loss=0.4567  val_mean_dice=0.2261  [LV:0.622  Myo:0.056  Scar:0.000]  lr=0.000999  t=12.5s
[  10/50] train_loss=0.1621  val_loss=0.1641  val_mean_dice=0.7113  [LV:0.888  Myo:0.706  Scar:0.540]  lr=0.000905  t=42.9s
[  20/50] train_loss=0.1398  val_loss=0.1559  val_mean_dice=0.7465  [LV:0.916  Myo:0.739  Scar:0.585]  lr=0.000655  t=41.7s
[  30/50] train_loss=0.1186  val_loss=0.1496  val_mean_dice=0.7634  [LV:0.922  Myo:0.751  Scar:0.616]  lr=0.000346  t=45.6s
[  40/50] train_loss=0.1031  val_loss=0.1454  val_mean_dice=0.7803  [LV:0.927  Myo:0.764  Scar:0.651]  lr=0.000096  t=46.5s
[  50/50] train_loss=0.0955  val_loss=0.1472  val_mean_dice=0.7826  [LV:0.929  Myo:0.767  Scar:0.652]  lr=0.000001  t=44.0s

  Best val Dice: 0.7830 at epoch 49
  Models saved to: outputs_exp_a1/phase1_nnunet_exp_a1_lge_intercenter_20260702_175324

  Computing per-class bottleneck centroids (for Phase 2)...
  Centroids saved to: outputs_exp_a1/phase1_nnunet_exp_a1_lge_intercenter_20260702_175324/centroid_cache.npz
    class 0 (background): shape=(512,)  norm=2.6896
    class 1 (LV): shape=(512,)  norm=2.6895
    class 2 (Myo): shape=(512,)  norm=2.6879
    class 3 (Scar): shape=(512,)  norm=2.6891

  Done. All outputs in: outputs_exp_a1/phase1_nnunet_exp_a1_lge_intercenter_20260702_175324


(nikhil-conda) nikhil@tanuh:~/Desktop/Research Works/Papers/Latent-Transformations$ python training/train_phase2_adapter.py --mode 2d --slices_csv Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_slices.csv --config configs/exp_a1_lge_intercenter.yaml --foundation_ckpt outputs_exp_a1/phase1_nnunet_exp_a1_lge_intercenter_20260702_175324/best_model.pth --model nnunet --output_dir outputs_exp_a1/ --adapter_train_cases 1 --epochs 100 --batch_size 8 --lr 1e-4 --lambda_align 0.5 --num_worker 4 --device cuda:0 --centroid_cache outputs_exp_a1/phase1_nnunet_exp_a1_lge_intercenter_20260
702_175324/centroid_cache.npz

============================================================
  Phase 2 Adapter Training
  Experiment  : exp_a1_lge_intercenter
  Mode        : 2D
  Target cases: 1 (one-shot=1)
  λ_align     : 0.5
  Output      : outputs_exp_a1/phase2_nnunet_adapter_exp_a1_lge_intercenter_20260702_183803
============================================================

[exp_a1_lge_intercenter] split=test total_slices=441
   center=E      sequence=LGE    slices=38
   center=F      sequence=LGE    slices=74
   center=G      sequence=LGE    slices=56
   center=H      sequence=LGE    slices=273
  Adapter train slices/volumes : 8
  Adapter val   slices/volumes : 433

  Loaded nnunet foundation from: outputs_exp_a1/phase1_nnunet_exp_a1_lge_intercenter_20260702_175324/best_model.pth
  Phase 1 best val Dice: 0.7829973767785465
  Bottleneck channels : 512
  Flow params (trainable): 789,504

  Loaded 4 class centroids from: outputs_exp_a1/phase1_nnunet_exp_a1_lge_intercenter_20260702_175324/centroid_cache.npz
[   1/100] tr_loss=0.4145  vl_loss=0.3704  tr_dice=0.0045  vl_dice=0.2965  align=0.0130  [LV:0.305  Myo:0.268  Scar:0.317]  lr=0.000100  t=6.5s
[  10/100] tr_loss=0.4093  vl_loss=0.3799  tr_dice=0.0035  vl_dice=0.2726  align=0.0111  [LV:0.274  Myo:0.249  Scar:0.295]  lr=0.000098  t=36.1s
[  20/100] tr_loss=0.4013  vl_loss=0.3979  tr_dice=0.0000  vl_dice=0.2285  align=0.0097  [LV:0.224  Myo:0.211  Scar:0.251]  lr=0.000091  t=22.3s
[  30/100] tr_loss=0.3928  vl_loss=0.4250  tr_dice=0.0000  vl_dice=0.1556  align=0.0091  [LV:0.148  Myo:0.156  Scar:0.163]  lr=0.000080  t=23.0s
[  40/100] tr_loss=0.3868  vl_loss=0.4495  tr_dice=0.0000  vl_dice=0.0877  align=0.0090  [LV:0.091  Myo:0.097  Scar:0.075]  lr=0.000066  t=22.7s
[  50/100] tr_loss=0.3839  vl_loss=0.4582  tr_dice=0.0000  vl_dice=0.0580  align=0.0081  [LV:0.072  Myo:0.065  Scar:0.037]  lr=0.000051  t=22.9s
[  60/100] tr_loss=0.3824  vl_loss=0.4591  tr_dice=0.0000  vl_dice=0.0508  align=0.0064  [LV:0.069  Myo:0.056  Scar:0.028]  lr=0.000035  t=23.5s
[  70/100] tr_loss=0.3815  vl_loss=0.4597  tr_dice=0.0000  vl_dice=0.0485  align=0.0050  [LV:0.063  Myo:0.055  Scar:0.027]  lr=0.000021  t=22.9s
[  80/100] tr_loss=0.3810  vl_loss=0.4601  tr_dice=0.0000  vl_dice=0.0470  align=0.0043  [LV:0.061  Myo:0.054  Scar:0.026]  lr=0.000010  t=23.5s
[  90/100] tr_loss=0.3809  vl_loss=0.4602  tr_dice=0.0000  vl_dice=0.0462  align=0.0041  [LV:0.060  Myo:0.053  Scar:0.026]  lr=0.000003  t=23.2s
[ 100/100] tr_loss=0.3808  vl_loss=0.4603  tr_dice=0.0000  vl_dice=0.0459  align=0.0040  [LV:0.060  Myo:0.053  Scar:0.025]  lr=0.000001  t=22.6s

  Best adapter val Dice : 0.2965 at epoch 1
  All outputs saved to  : outputs_exp_a1/phase2_nnunet_adapter_exp_a1_lge_intercenter_20260702_183803


(nikhil-conda) nikhil@tanuh:~/Desktop/Research Works/Papers/Latent-Transformations$ python training/train_phase2_adapter.py --mode 2d --slices_csv Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_slices.csv --config configs/exp_a1_lge_intercenter.yaml --foundation_ckpt outputs_exp_a1/phase1_nnunet_exp_a1_lge_intercenter_20260702_175324/best_model.pth --model nnunet --output_dir outputs_exp_a1/ --adapter_train_cases 1 --epochs 100 --batch_size 8 --lr 1e-4 --lambda_align 0.5 --num_worker 4 --device cuda:0 --centroid_cache outputs_exp_a1/phase1_nnunet_exp_a1_lge_intercenter_20260702_175324/centroid_cache.npz --no_adapter

============================================================
  Phase 2 Adapter Training
  Experiment  : exp_a1_lge_intercenter
  Mode        : 2D
  Target cases: 1 (one-shot=1)
  λ_align     : 0.5
  Output      : outputs_exp_a1/phase2_nnunet_no_adapter_exp_a1_lge_intercenter_20260702_191651
============================================================

[exp_a1_lge_intercenter] split=test total_slices=441
   center=E      sequence=LGE    slices=38
   center=F      sequence=LGE    slices=74
   center=G      sequence=LGE    slices=56
   center=H      sequence=LGE    slices=273
  Adapter train slices/volumes : 8
  Adapter val   slices/volumes : 433

  Loaded nnunet foundation from: outputs_exp_a1/phase1_nnunet_exp_a1_lge_intercenter_20260702_175324/best_model.pth
  Phase 1 best val Dice: 0.7829973767785465
  Mode: NO ADAPTER (frozen foundation only — ablation baseline)
  Frozen foundation train Dice : 0.0131
  Frozen foundation val   Dice : 0.3749
  All outputs saved to        : outputs_exp_a1/phase2_nnunet_no_adapter_exp_a1_lge_intercenter_20260702_191651

---

(nikhil-conda) nikhil@tanuh:~/Desktop/Research Works/Papers/Latent-Transformations$ python training/train_phase1_foundation.py --mode 2d --slices_csv Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_slices.csv --config configs/exp_a1_lge_intercenter.yaml --output_dir outputs_exp_a1 --epochs 50 --batch_size 16 --lr 1e-3 --num_workers 4 --device cuda:0 --model unet

============================================================
  Phase 1 Foundation Training
  Experiment : exp_a1_lge_intercenter
  Mode       : 2D
  Output     : outputs_exp_a1/phase1_unet_exp_a1_lge_intercenter_20260702_175345
============================================================

[exp_a1_lge_intercenter] split=train total_slices=1764
   center=A      sequence=LGE    slices=1405
   center=B      sequence=LGE    slices=185
   center=C      sequence=LGE    slices=174
  Train samples : 1500
  Val   samples : 264
  Classes (4): ['background', 'LV', 'Myo', 'Scar']

  Model params  : 7,762,564
[   1/50] train_loss=0.5248  val_loss=0.3810  val_mean_dice=0.3175  [LV:0.722  Myo:0.231  Scar:0.000]  lr=0.000999  t=45.2s
[  10/50] train_loss=0.1560  val_loss=0.1660  val_mean_dice=0.7157  [LV:0.900  Myo:0.722  Scar:0.526]  lr=0.000905  t=41.2s
[  20/50] train_loss=0.1340  val_loss=0.1541  val_mean_dice=0.7502  [LV:0.920  Myo:0.742  Scar:0.589]  lr=0.000655  t=41.4s
[  30/50] train_loss=0.1132  val_loss=0.1433  val_mean_dice=0.7779  [LV:0.926  Myo:0.757  Scar:0.651]  lr=0.000346  t=44.8s
[  40/50] train_loss=0.0951  val_loss=0.1411  val_mean_dice=0.7895  [LV:0.929  Myo:0.767  Scar:0.672]  lr=0.000096  t=46.2s
[  50/50] train_loss=0.0871  val_loss=0.1419  val_mean_dice=0.7917  [LV:0.932  Myo:0.771  Scar:0.671]  lr=0.000001  t=33.0s

  Best val Dice: 0.7918 at epoch 45
  Models saved to: outputs_exp_a1/phase1_unet_exp_a1_lge_intercenter_20260702_175345

  Computing per-class bottleneck centroids (for Phase 2)...
  Centroids saved to: outputs_exp_a1/phase1_unet_exp_a1_lge_intercenter_20260702_175345/centroid_cache.npz
    class 0 (background): shape=(512,)  norm=2.3798
    class 1 (LV): shape=(512,)  norm=2.3891
    class 2 (Myo): shape=(512,)  norm=2.3874
    class 3 (Scar): shape=(512,)  norm=2.3879

  Done. All outputs in: outputs_exp_a1/phase1_unet_exp_a1_lge_intercenter_20260702_175345

(nikhil-conda) nikhil@tanuh:~/Desktop/Research Works/Papers/Latent-Transformations$ python training/train_phase2_adapter.py --mode 2d --slices_csv Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_slices.csv --config configs/exp_a1_lge_intercenter.yaml --foundation_ckpt outputs_exp_a1/phase1_unet_exp_a1_lge_intercenter_20260702_175345/best_model.pth --model unet --output_dir outputs_exp_a1/ --adapter_train_cases 1 --epochs 100 --batch_size 8 --lr 1e-4 --lambda_align 0.5 --num_worker 4 --device cuda:0 --centroid_cache outputs_exp_a1/phase1_unet_exp_a1_lge_intercenter_20260702_17
5345/centroid_cache.npz

============================================================
  Phase 2 Adapter Training
  Experiment  : exp_a1_lge_intercenter
  Mode        : 2D
  Target cases: 1 (one-shot=1)
  λ_align     : 0.5
  Output      : outputs_exp_a1/phase2_unet_adapter_exp_a1_lge_intercenter_20260702_183839
============================================================

[exp_a1_lge_intercenter] split=test total_slices=441
   center=E      sequence=LGE    slices=38
   center=F      sequence=LGE    slices=74
   center=G      sequence=LGE    slices=56
   center=H      sequence=LGE    slices=273
  Adapter train slices/volumes : 8
  Adapter val   slices/volumes : 433

  Loaded unet foundation from: outputs_exp_a1/phase1_unet_exp_a1_lge_intercenter_20260702_175345/best_model.pth
  Phase 1 best val Dice: 0.7918056635295643
  Bottleneck channels : 512
  Flow params (trainable): 789,504

  Loaded 4 class centroids from: outputs_exp_a1/phase1_unet_exp_a1_lge_intercenter_20260702_175345/centroid_cache.npz
[   1/100] tr_loss=0.3866  vl_loss=0.4595  tr_dice=0.0000  vl_dice=0.1189  align=0.0123  [LV:0.102  Myo:0.100  Scar:0.155]  lr=0.000100  t=23.6s
[  10/100] tr_loss=0.3843  vl_loss=0.4689  tr_dice=0.0000  vl_dice=0.0964  align=0.0108  [LV:0.081  Myo:0.078  Scar:0.129]  lr=0.000098  t=21.3s
[  20/100] tr_loss=0.3822  vl_loss=0.4770  tr_dice=0.0000  vl_dice=0.0770  align=0.0092  [LV:0.065  Myo:0.057  Scar:0.108]  lr=0.000091  t=22.9s
[  30/100] tr_loss=0.3807  vl_loss=0.4821  tr_dice=0.0000  vl_dice=0.0702  align=0.0075  [LV:0.071  Myo:0.044  Scar:0.095]  lr=0.000080  t=22.1s
[  40/100] tr_loss=0.3796  vl_loss=0.4848  tr_dice=0.0000  vl_dice=0.0676  align=0.0060  [LV:0.080  Myo:0.038  Scar:0.085]  lr=0.000066  t=23.2s
[  50/100] tr_loss=0.3787  vl_loss=0.4860  tr_dice=0.3333  vl_dice=0.0750  align=0.0047  [LV:0.112  Myo:0.034  Scar:0.079]  lr=0.000051  t=22.6s
[  60/100] tr_loss=0.3781  vl_loss=0.4864  tr_dice=0.3333  vl_dice=0.0785  align=0.0037  [LV:0.128  Myo:0.032  Scar:0.076]  lr=0.000035  t=22.5s
[  70/100] tr_loss=0.3778  vl_loss=0.4866  tr_dice=0.3333  vl_dice=0.0771  align=0.0030  [LV:0.126  Myo:0.031  Scar:0.074]  lr=0.000021  t=22.5s
[  80/100] tr_loss=0.3776  vl_loss=0.4868  tr_dice=0.3333  vl_dice=0.0763  align=0.0027  [LV:0.125  Myo:0.030  Scar:0.073]  lr=0.000010  t=22.4s
[  90/100] tr_loss=0.3775  vl_loss=0.4868  tr_dice=0.3333  vl_dice=0.0761  align=0.0026  [LV:0.125  Myo:0.030  Scar:0.073]  lr=0.000003  t=22.0s
[ 100/100] tr_loss=0.3775  vl_loss=0.4868  tr_dice=0.3333  vl_dice=0.0760  align=0.0025  [LV:0.125  Myo:0.030  Scar:0.073]  lr=0.000001  t=5.2s

  Best adapter val Dice : 0.1189 at epoch 1
  All outputs saved to  : outputs_exp_a1/phase2_unet_adapter_exp_a1_lge_intercenter_20260702_183839

(nikhil-conda) nikhil@tanuh:~/Desktop/Research Works/Papers/Latent-Transformations$ 

(nikhil-conda) nikhil@tanuh:~/Desktop/Research Works/Papers/Latent-Transformations$ python training/train_phase2_adapter.py --mode 2d --slices_csv Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_slices.csv --config configs/exp_a1_lge_intercenter.yaml --foundation_ckpt outputs_exp_a1/phase1_unet_exp_a1_lge_intercenter_20260702_175345/best_model.pth --model unet --output_dir outputs_exp_a1/ --adapter_train_cases 1 --epochs 100 --batch_size 8 --lr 1e-4 --lambda_align 0.5 --num_worker 4 --device cuda:0 --centroid_cache outputs_exp_a1/phase1_unet_exp_a1_lge_intercenter_20260702_175345/centroid_cache.npz --no_adapter

============================================================
  Phase 2 Adapter Training
  Experiment  : exp_a1_lge_intercenter
  Mode        : 2D
  Target cases: 1 (one-shot=1)
  λ_align     : 0.5
  Output      : outputs_exp_a1/phase2_unet_no_adapter_exp_a1_lge_intercenter_20260702_191658
============================================================

[exp_a1_lge_intercenter] split=test total_slices=441
   center=E      sequence=LGE    slices=38
   center=F      sequence=LGE    slices=74
   center=G      sequence=LGE    slices=56
   center=H      sequence=LGE    slices=273
  Adapter train slices/volumes : 8
  Adapter val   slices/volumes : 433

  Loaded unet foundation from: outputs_exp_a1/phase1_unet_exp_a1_lge_intercenter_20260702_175345/best_model.pth
  Phase 1 best val Dice: 0.7918056635295643
  Mode: NO ADAPTER (frozen foundation only — ablation baseline)
  Frozen foundation train Dice : 0.0000
  Frozen foundation val   Dice : 0.3541
  All outputs saved to        : outputs_exp_a1/phase2_unet_no_adapter_exp_a1_lge_intercenter_20260702_191658