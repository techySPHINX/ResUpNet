# Dataset Source

This repo targets the Kaggle BraTS 2021 Task 1 dataset:

```text
https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
slug: dschettler8845/brats-2021-task1
```

The training pipeline expects raw BraTS-style patient folders after download/extraction.
For this workspace, the active extracted dataset root is:

```text
data/kaggle_brats2021_task1/extracted
```

Storage note for this machine:

```text
C:/Users/JK/desktop/resupnet/data
  -> junction to E:/ResUpNet/data

C:/Users/JK/desktop/resupnet/experiments/v2_multimodal_roi/processed_splits
  -> junction to E:/ResUpNet/processed_splits/v2_multimodal_roi
```

The repo paths are kept so existing commands still work, but the heavy raw and
preprocessed BraTS files are physically stored on `E:`.

The processed dataset used for training is:

```text
Command path:
experiments/v2_multimodal_roi/processed_splits

Physical path:
E:/ResUpNet/processed_splits/v2_multimodal_roi
```

Current processed split:

```text
X_train: (41868, 160, 160, 4), float16
y_train: (41868, 160, 160, 1), uint8
X_val:   (9024, 160, 160, 4), float16
y_val:   (9024, 160, 160, 1), uint8
X_test:  (9034, 160, 160, 4), float16
y_test:  (9034, 160, 160, 1), uint8
Patients: train=875, val=188, test=188
Patient overlap: none
```

Required files per patient:

```text
*_t1.nii.gz
*_t1ce.nii.gz
*_t2.nii.gz
*_flair.nii.gz
*_seg.nii.gz
```

Training task:

```text
Input:  T1 + T1ce + T2 + FLAIR
Output: binary whole-tumor segmentation mask
```

The model does not train from the old single-channel `processed_splits` folder. That folder is not the desired BraTS multimodal input.

Kaggle credentials:

```text
data/.kaggle/kaggle.json
```

Download command:

```powershell
python download_kaggle_brats2021.py --output-dir data\kaggle_brats2021_task1
```

After download, the Kaggle archive must be extracted so patient folders are visible under
`data\kaggle_brats2021_task1\extracted`.
