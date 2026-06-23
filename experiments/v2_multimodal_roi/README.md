# Phase 2 Experiment: Multimodal ROI ResUpNet

Version: `v2_multimodal_roi`

## Active Dataset

Training path:

```text
experiments\v2_multimodal_roi\processed_splits
```

Physical storage:

```text
E:\ResUpNet\processed_splits\v2_multimodal_roi
```

## Split Summary

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

Slice classes:

```text
train: tumor=31247, near_tumor=3669, hard_negative=6952
val:   tumor=6738,  near_tumor=788,  hard_negative=1498
test:  tumor=6730,  near_tumor=807,  hard_negative=1497
```

## Current Validated Result

The current synchronized training artifact is:

```text
resupnet_training_curve.json
```

Validated epoch-50 selected-slice validation metrics:

```text
Dice: 0.890146
IoU:  0.802039
F1:   0.891956
HD95: 4.8877
ASD:  1.5056
Loss: 0.262043
```

The generated proof report and plots are under:

```text
reports\phase2_metrics_validation
```

## Input Strategy

- Load `T1`, `T1ce`, `T2`, and `FLAIR`.
- Normalize each modality per patient.
- Crop brain ROI using image intensity only.
- Resize ROI to `160x160`.
- Keep tumor, near-tumor, and hard-negative slices.

## Training

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 150 --batch-size 8 --base-filters 32 --mixed-precision --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb
```

If VRAM is tight:

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 150 --batch-size 4 --base-filters 24 --mixed-precision --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_safe
```

## Evaluation

```powershell
python -B evaluate_phase2_model_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --model-path E:\ResUpNet\runs\resupnet_torch_cuda_8gb\checkpoints\best_model.pt --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_tta_post --tta --postprocess --min-component-size 32
```

## Plots

```powershell
python -B generate_phase2_plots.py --run-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb --evaluation-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_tta_post
```

Report tumor-row metrics, all-row metrics, and empty-true false positives
together.
