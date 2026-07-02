# Dataloaders


### Indexing for the metadata csv's

- Firt run the index_myo2d.py to create the global csv for the Myo task and it will create the metadata and store it properly somwhere, i have stored it in the Datasets/CURE2026-Left-Myocardium/Metadata folder. The output is likely like: 

```
Indexed 468 volumes -> Datasets/CURE2026-Left-Myocardium/Metadata/metadata_myo2d_volumes.csv
  center=A      sequence=LGE    -> 81 volumes
  center=Alpha  sequence=Cine   -> 40 volumes
  center=B      sequence=LGE    -> 35 volumes
  center=B      sequence=T2     -> 35 volumes
  center=B      sequence=bSSFP  -> 35 volumes
  center=Beta   sequence=Cine   -> 24 volumes
  center=C      sequence=LGE    -> 45 volumes
  center=C      sequence=T2     -> 45 volumes
  center=C      sequence=bSSFP  -> 45 volumes
  center=E      sequence=LGE    -> 7 volumes
  center=E      sequence=bSSFP  -> 7 volumes
  center=F      sequence=LGE    -> 9 volumes
  center=F      sequence=bSSFP  -> 9 volumes
  center=G      sequence=LGE    -> 8 volumes
  center=G      sequence=bSSFP  -> 8 volumes
  center=H      sequence=LGE    -> 35 volumes
```


---

- Second run the index_wholeheart3d.py to create the global csv for the Myo task and it will create the metadata and store it properly somwhere, i have stored it in the Datasets/CURE2026-Whole-Heart/Metadata folder. The output is likely like: 

```
Indexed 105 volumes -> Datasets/CURE2026-Whole-Heart/Metadata/metadata_wholeheart3d.csv
  center=A      modality=CT     -> 20 volumes
  center=B      modality=CT     -> 20 volumes
  center=C&D    modality=MRI    -> 20 volumes
  center=E      modality=MRI    -> 26 volumes
  center=G      modality=CT     -> 19 volumes
```

---


### Creating the Labels Map

Run the check_label_values.py and point to the Myo2D task metadata csv and run the code, and the output is likely to be like:

```
center   sequence unique label values found
------------------------------------------------------------
A        LGE      [0, 200, 500, 2221]
Alpha    Cine     [0, 200, 500, 2221]
B        LGE      [0, 200, 500, 600, 1220, 2221]
B        T2       [0, 200, 500, 600, 1220, 2221]
B        bSSFP    [0, 200, 500, 600, 1220, 2221]
Beta     Cine     [0, 200, 500, 2221]
C        LGE      [0, 200, 500, 600, 1220, 2221]
C        T2       [0, 200, 500, 600, 1220, 2221]
C        bSSFP    [0, 200, 500, 600, 1220, 2221]
E        LGE      [0, 200, 500, 600, 2221]
E        bSSFP    [0, 200, 500, 600, 2221]
F        LGE      [0, 200, 500, 600, 2221]
F        bSSFP    [0, 200, 500, 600, 2221]
G        LGE      [0, 200, 500, 600, 2221]
G        bSSFP    [0, 200, 500, 600, 2221]
H        LGE      [0, 2221]
```

---


### Converting the 4D Cine images to 3D, matching to the given ground truth

Run the cinemyo_4d_to_2d.py with the both center_alpha and center_beta, first rename the original data from center_alpha to center_alpha_4d, and then run the code, same applies for the center_beta too, because the dataloader takes the folder name as center_alpha only, so we need to do this. You can just run this file and the preprocessing will be done.



---

### Creating 2D NPY images for the Myo2D data

The official is actually like 3d data, but with very low z-index frames, so we treat this as 2D task and for the faster loading purposes, lets save this official data format to 2d npy files. To do this, run this file: extract_slices_myo2d.py attacing the volumes_csv and the cache_dir properly. I have stored the cache in the folder: Datasets/CURE2026-Left-Myocardium/2D_NPY_Cache

---

###

The B and C centers in the Myo2D task needs to have the internal train and test split, becuase we are going to use those centers for the intra center different sequence experiments. To do this run the make_patient_split.py code, and you will get a output of the train and test pool of the cases from B and C centers. Now in the exp_b1*.yaml, paste that total in the place of this below placeholder. But for now this is all going to be pushed to the github. Dont worry! Thank me later.
```
patient_split:
  B:
    train_pool: [Case0001, Case0002, Case0003]   
    test_pool: [Case0030, Case0031]
  C:
    train_pool: [Case0101, Case0102, Case0103]
    test_pool: [Case0140, Case0141]
```

---

### Labels Mapping

- Myocardium 2D Task:

```
0: 0,   # background
500: 1,   # LV
200: 2,   # Myo
2221: 3,   # Scar
```

- Whole Heart 3D Task:

```
0: 0,      # background
500: 1,    # LV
600: 2,    # RV
420: 3,    # LA
550: 4,    # RA
205: 5,    # Myo
820: 6,    # AO
850: 7,    # PA
```

---

## Data Analysis

### Myocardium 2D data

- The image dimension distributions are Below 200: 22, Near to 250: 1568, Above 250: 174, images are there. And majority of them are 256, so lets just resize them to 256 using interpolate.
- Currently the image data statistics, is standardized normal distribution with mean~0 and std=1. The min~0.7 and max~4.0. We need to normalize them to [0,1]
- The labels are good, each mapped according to the labels dictionary that we have. 

---