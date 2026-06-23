# Phase 2 Commands: Kaggle BraTS 2021 Task 1

Dataset:

```text
https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
slug: dschettler8845/brats-2021-task1
```

Storage layout:

```text
data -> E:\ResUpNet\data
experiments\v2_multimodal_roi\processed_splits -> E:\ResUpNet\processed_splits\v2_multimodal_roi
training outputs -> E:\ResUpNet\runs
runtime caches -> E:\ResUpNet\cache
temporary files -> E:\ResUpNet\tmp
```

Active processed dataset:

```text
Command path:   experiments\v2_multimodal_roi\processed_splits
Physical path:  E:\ResUpNet\processed_splits\v2_multimodal_roi
```

## 1. Activate Environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\activate_resupnet.ps1
```

## 2. Install Strict Requirements

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_brats.txt
```

Verify native GPU:

```powershell
python -B -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))"
```

## 3. Current Prepared Split

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

## 4. Optional Preflight

```powershell
python preflight_phase2_dataset.py --dataset-root data\kaggle_brats2021_task1\extracted --deep-check
```

Expected deep-check shape:

```text
image_shape: [256, 256, 4]
mask_shape:  [256, 256, 1]
```

## 5. Rebuild Processed Split If Needed

Use this only if the processed split needs to be regenerated:

```powershell
python prepare_phase2_dataset.py --dataset-root data\kaggle_brats2021_task1\extracted --output-dir experiments\v2_multimodal_roi\processed_splits --target-size 160 --max-tumor-slices-per-patient 36 --max-near-slices-per-patient 8 --max-hard-negative-slices-per-patient 8
```

## 6. Sanity Training

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 1 --batch-size 2 --base-filters 4 --steps-per-epoch 1 --validation-steps 1 --no-augment --no-balanced-batches --mixed-precision --output-dir E:\ResUpNet\runs\torch_native_sanity
```

## 7. Full Training

Best first run for RTX 5050 8 GB. This keeps the core ResUpNet architecture
unchanged while enabling conservative MRI augmentation, gradient clipping, and
EMA checkpointing:

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 150 --batch-size 8 --base-filters 32 --mixed-precision --augmentation-policy conservative --grad-clip-norm 1.0 --ema-decay 0.999 --early-stopping-patience 35 --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb
```

Memory-safe run:

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 150 --batch-size 4 --base-filters 24 --mixed-precision --augmentation-policy conservative --grad-clip-norm 1.0 --ema-decay 0.999 --early-stopping-patience 35 --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_safe
```

## 8. Evaluation

Validation-threshold test evaluation:

```powershell
python -B evaluate_phase2_model_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --model-path E:\ResUpNet\runs\resupnet_torch_cuda_8gb\checkpoints\best_model.pt --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation
```

TTA plus post-processing for the raw best checkpoint:

```powershell
python -B evaluate_phase2_model_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --model-path E:\ResUpNet\runs\resupnet_torch_cuda_8gb\checkpoints\best_model.pt --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_tta_post --tta --postprocess --min-component-size 32
```

TTA plus post-processing for the EMA checkpoint, if `best_ema_model.pt` exists:

```powershell
python -B evaluate_phase2_model_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --model-path E:\ResUpNet\runs\resupnet_torch_cuda_8gb\checkpoints\best_ema_model.pt --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_ema_tta_post --tta --postprocess --min-component-size 32
```

## 9. Plots

```powershell
python -B generate_phase2_plots.py --run-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb --evaluation-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_tta_post
```

Generated plot types include:

```text
training curves
learning-rate curve
threshold curves
ROC-like and PR-like threshold plots
aggregate confusion matrix
metric distributions
violin plots
metric correlation heatmap
Bland-Altman tumor-area plot
error analysis plots
summary bar chart
radar chart
```

Trust these together:

```text
tumor-row Dice / IoU / recall / precision
all-row Dice / IoU
HD95 / ASD
empty-true false positives
```

## 10. Checked-In Artifact Validation Report

For the current checked-in trained-result artifacts:

```text
resupnet_training_curve.json
training_history_rows.json
```

regenerate the synchronized validation report and proof plots with:

```powershell
.\.venv\Scripts\python.exe generate_phase2_artifact_report.py
```

The current validated selected-slice epoch-50 result is:

```text
Dice: 0.890146
IoU:  0.802039
F1:   0.891956
HD95: 4.8877
ASD:  1.5056
Loss: 0.262043
```

Report path:

```text
reports\phase2_metrics_validation\RESUPNET_PHASE2_RESULTS_REPORT.md
```
