# ResUpNet Phase 2: BraTS 2021 Tumor Segmentation

This repository is now prepared for Phase 2 ResUpNet training on the Kaggle
BraTS 2021 Task 1 dataset.

```text
Dataset: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
Task:    binary whole-tumor segmentation
Input:   T1 + T1ce + T2 + FLAIR MRI slices
Output:  binary tumor mask from BraTS seg labels
Model:   PyTorch ResUpNet
```

## Current Train-Ready Dataset

Use this path in training commands:

```text
experiments\v2_multimodal_roi\processed_splits
```

That path is a Windows junction. The heavy files are physically stored here:

```text
E:\ResUpNet\processed_splits\v2_multimodal_roi
```

Current split:

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

The raw extracted BraTS data is available through:

```text
data\kaggle_brats2021_task1\extracted
```

That path is also a junction to:

```text
E:\ResUpNet\data\kaggle_brats2021_task1\extracted
```

## Storage Policy

Heavy generated files must stay on `E:`.

```text
data -> E:\ResUpNet\data
experiments\v2_multimodal_roi\processed_splits -> E:\ResUpNet\processed_splits\v2_multimodal_roi
training outputs -> E:\ResUpNet\runs
runtime caches -> E:\ResUpNet\cache
```

Do not delete the `data` or `processed_splits` folders from the repo. They are
junction links used by the scripts.

## Phase 2 Method

The active pipeline keeps the safest non-deviating improvements:

- all four modalities: `T1`, `T1ce`, `T2`, `FLAIR`
- per-patient modality normalization
- image-intensity ROI crop, not mask-based crop
- compact `160x160x4` input due to local disk constraints
- capped tumor, near-tumor, and hard-negative slice selection
- patient-wise train/validation/test split
- validation-only threshold selection during evaluation
- pixel-micro hard metrics from pooled TP/FP/FN/TN counts

Current validation and test arrays are patient-wise but still use the same
selected-slice policy as training. They are suitable for internal selected-slice
experiments, but not for official full-volume BraTS reporting. For publication
claims, evaluate on full patient volumes or clearly report the selected-slice
protocol.

## Metric Protocol

[`METRICS_PROTOCOL.md`](METRICS_PROTOCOL.md) is the source of truth for metric
computation. The primary result pools TP/FP/FN/TN over every pixel in all
selected slices before calculating Dice, IoU, precision, recall, F1,
specificity, and accuracy. Slice means are diagnostics, and patient-level
averaging is not used.

## Historical Result Artifact

The checked-in result artifacts have a reproducible validation report:

```text
reports\phase2_metrics_validation\RESUPNET_PHASE2_RESULTS_REPORT.md
```

It checks the internal consistency of `resupnet_training_curve.json` and
`training_history_rows.json`, generates plots, and documents their claim
boundary. Those historical files do not contain predictions or confusion
counts, so their aggregation cannot be independently reconstructed and their
values must not be relabeled as verified protocol-1.0 micro metrics.

Recorded epoch-50 selected-slice validation metrics:

```text
Dice: 0.890146
IoU:  0.802039
F1:   0.891956
HD95: 4.8877
ASD:  1.5056
Loss: 0.262043
```

Regenerate the report and plots with:

```powershell
.\.venv\Scripts\python.exe generate_phase2_artifact_report.py
```

## Key Files

- `download_kaggle_brats2021.py`: Kaggle dataset download helper
- `preflight_phase2_dataset.py`: raw dataset validation
- `phase2_input_pipeline.py`: multimodal normalization, ROI crop, slice selection
- `prepare_phase2_dataset.py`: patient-wise split generation
- `resupnet_torch_model.py`: native PyTorch ResUpNet model
- `segmentation_metrics.py`: canonical pooled-count metric implementation
- `METRICS_PROTOCOL.md`: exact formulas, aggregation, edge cases, and output contract
- `train_phase2_resupnet_torch.py`: native Windows CUDA training for RTX GPUs
- `evaluate_phase2_model_torch.py`: native PyTorch evaluation
- `generate_phase2_plots.py`: post-evaluation plots and metric visualizations
- `TRAINING_BACKENDS.md`: active PyTorch backend setup
- `TRAINING_COMMANDS_PHASE2.md`: exact commands to run
- `DATASET_SOURCE.md`: dataset and storage details
- `PHASE2_STRATEGY.md`: research strategy and ablation direction
- `generate_phase2_artifact_report.py`: validates checked-in result artifacts
  and generates the Phase 2 result report

## Start Training

Activate the E-drive virtual environment first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\activate_resupnet.ps1
```

Native Windows RTX 5050 GPU run:

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 150 --batch-size 8 --base-filters 32 --mixed-precision --augmentation-policy conservative --grad-clip-norm 1.0 --ema-decay 0.999 --early-stopping-patience 35 --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb
```

If VRAM is tight:

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 150 --batch-size 4 --base-filters 24 --mixed-precision --augmentation-policy conservative --grad-clip-norm 1.0 --ema-decay 0.999 --early-stopping-patience 35 --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_safe
```

After training, evaluate the best checkpoint:

```powershell
python -B evaluate_phase2_model_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --model-path E:\ResUpNet\runs\resupnet_torch_cuda_8gb\checkpoints\best_model.pt --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_tta_post --tta --postprocess
```

The canonical test result is `primary_test_metrics` in
`evaluation_summary.json`. Dice, IoU, precision, recall, F1, specificity, and
accuracy are micro-averaged by pooling TP/FP/FN/TN across every pixel in all
selected test slices. Per-slice summaries remain available for diagnostics;
metrics are not averaged per patient. HD95 and ASD are per-slice boundary
metrics summarized over slices with defined distances. See
[`METRICS_PROTOCOL.md`](METRICS_PROTOCOL.md) for the complete protocol.

Generate plots:

```powershell
python -B generate_phase2_plots.py --run-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb --evaluation-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_tta_post
```
