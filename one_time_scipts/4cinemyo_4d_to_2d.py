import os
import nibabel as nib
import numpy as np

input_root = "Datasets/CURE2026-Left-Myocardium/Myo_train/CineMyoPS_train/center_beta_4d"
output_root = "Datasets/CURE2026-Left-Myocardium/Myo_train/CineMyoPS_train/center_beta"

os.makedirs(output_root, exist_ok=True)

frame_idx = 0   # take 0th frame from Cine

for fname in os.listdir(input_root):
    if not fname.endswith(".nii.gz"):
        continue

    # Process only Cine files
    if fname.endswith("_Cine.nii.gz"):
        cine_path = os.path.join(input_root, fname)
        gd_name = fname.replace("_Cine.nii.gz", "_gd.nii.gz")
        gd_path = os.path.join(input_root, gd_name)

        # --- Load and convert Cine 4D -> 3D ---
        cine_nii = nib.load(cine_path)
        cine_data = cine_nii.get_fdata()

        print(f"Processing Cine: {fname}, shape: {cine_data.shape}")

        if cine_data.ndim != 4:
            print(f"Skipping {fname} because it is not 4D")
            continue

        if frame_idx >= cine_data.shape[3]:
            print(f"Skipping {fname} because frame_idx={frame_idx} is out of range")
            continue

        cine_3d = cine_data[:, :, :, frame_idx]   # shape: (256, 256, 14)

        cine_out = nib.Nifti1Image(np.asarray(cine_3d), affine=cine_nii.affine, header=cine_nii.header)
        cine_out.header.set_data_shape(cine_3d.shape)

        cine_out_path = os.path.join(output_root, fname)
        nib.save(cine_out, cine_out_path)
        print(f"Saved Cine: {cine_out_path} -> {cine_3d.shape}")

        # --- Copy GD as it is ---
        if os.path.exists(gd_path):
            gd_nii = nib.load(gd_path)
            gd_out_path = os.path.join(output_root, gd_name)
            nib.save(gd_nii, gd_out_path)
            print(f"Saved GD:   {gd_out_path} -> {gd_nii.shape}")
        else:
            print(f"GD not found for {fname}")