# Project Status

Last updated: 2026-06-19

## Current State

The repo is train-ready for Phase 2 ResUpNet on Kaggle BraTS 2021 Task 1 using
native Windows PyTorch CUDA.

```text
Task: binary whole-tumor segmentation
Input channels: T1, T1ce, T2, FLAIR
Input shape: 160x160x4
Output shape: 160x160x1
Split type: patient-wise
Patient overlap: none
Backend: native_windows_torch_cuda
```

## Current Validated Result Artifact

The current result source of truth is:

```text
resupnet_training_curve.json
```

Artifact validation report:

```text
reports\phase2_metrics_validation\RESUPNET_PHASE2_RESULTS_REPORT.md
```

Validated selected-slice validation result at epoch `50`:

```text
Dice:        0.890146
IoU:         0.802039
F1:          0.891956
Precision:   0.901572
Recall:      0.882543
Specificity: 0.993046
Accuracy:    0.990363
HD95:        4.8877
ASD:         1.5056
Loss:        0.262043
```

`training_history_rows.json` is synchronized with the full curve for its shared
rows, but it only contains epochs `1`, `2`, and `50`. Use
`resupnet_training_curve.json` for trend analysis and generated plots.

This is a validated selected-slice internal result, not an official full-volume
BraTS leaderboard result.

## Dataset Locations

Use this command path:

```text
experiments\v2_multimodal_roi\processed_splits
```

Physical storage:

```text
E:\ResUpNet\processed_splits\v2_multimodal_roi
```

Raw extracted data command path:

```text
data\kaggle_brats2021_task1\extracted
```

Raw extracted data physical storage:

```text
E:\ResUpNet\data\kaggle_brats2021_task1\extracted
```

## Prepared Split

```text
X_train: (41868, 160, 160, 4), float16
y_train: (41868, 160, 160, 1), uint8
X_val:   (9024, 160, 160, 4), float16
y_val:   (9024, 160, 160, 1), uint8
X_test:  (9034, 160, 160, 4), float16
y_test:  (9034, 160, 160, 1), uint8
```

Patient counts:

```text
train=875
val=188
test=188
```

Slice class counts:

```text
train: tumor=31247, near_tumor=3669, hard_negative=6952
val:   tumor=6738,  near_tumor=788,  hard_negative=1498
test:  tumor=6730,  near_tumor=807,  hard_negative=1497
```

## Storage Rule

Keep heavy generated outputs on `E:`.

```text
training outputs: E:\ResUpNet\runs
runtime caches:   E:\ResUpNet\cache
temporary files:  E:\ResUpNet\tmp
```

## Environment

The project venv is linked at `.venv` and physically stored at:

```text
E:\ResUpNet\venvs\resupnet_phase2
```

Activate it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\activate_resupnet.ps1
```

## GPU Training

Recommended first run:

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 150 --batch-size 8 --base-filters 32 --mixed-precision --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb
```

Memory-safe run:

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 150 --batch-size 4 --base-filters 24 --mixed-precision --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_safe
```

Evaluation:

```powershell
python -B evaluate_phase2_model_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --model-path E:\ResUpNet\runs\resupnet_torch_cuda_8gb\checkpoints\best_model.pt --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_tta_post --tta --postprocess --min-component-size 32
```

Plots:

```powershell
python -B generate_phase2_plots.py --run-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb --evaluation-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_tta_post
```
