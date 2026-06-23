# Phase 2 Strategy

Train the Phase 2 multimodal ResUpNet pipeline on Kaggle BraTS 2021 Task 1.

```text
T1 + T1ce + T2 + FLAIR -> PyTorch ResUpNet -> binary whole-tumor mask
```

## Active Dataset

Use this command path:

```text
experiments\v2_multimodal_roi\processed_splits
```

Physical storage:

```text
E:\ResUpNet\processed_splits\v2_multimodal_roi
```

Prepared split:

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

## Current Validated Artifact Result

The current trained-result artifact is `resupnet_training_curve.json`. It is
validated by:

```text
reports\phase2_metrics_validation\RESUPNET_PHASE2_RESULTS_REPORT.md
```

Epoch-50 selected-slice validation metrics:

```text
Dice: 0.890146
IoU:  0.802039
F1:   0.891956
HD95: 4.8877
ASD:  1.5056
Loss: 0.262043
```

`training_history_rows.json` is a compact synchronized cross-check for epochs
`1`, `2`, and `50`; it is not the full curve.

## Why This Setup Is Safe

- Patient-wise split prevents slice leakage.
- Four modalities preserve complementary tumor information.
- ROI cropping removes black/background area without using tumor masks for localization.
- Training uses tumor, near-tumor, and hard-negative slices.
- Validation and test patients are held out with no patient overlap. The current
  arrays still contain selected/capped slices, so they are honest for the
  selected-slice protocol but not official full-volume BraTS metrics.
- Training outputs and caches are routed to `E:` to protect `C:` space.

## Input Size Policy

The active split uses `160x160x4`. Do not reduce below `160x160` unless an
ablation confirms no drop in Dice, IoU, recall, and HD95.

Future detail ablations:

```text
160x160x4 current baseline
192x192x4 higher-detail ablation
256x256x4 full-detail ablation
```

## Training Plan

1. Train the PyTorch model on native Windows CUDA.
2. Save the best checkpoint by validation Dice.
3. Select the threshold from validation predictions only.
4. Evaluate the locked test set once.
5. Generate plots from the saved evaluation files.
6. Report all-row, tumor-row, and empty-true-row metrics together.

Training:

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 150 --batch-size 8 --base-filters 32 --mixed-precision --augmentation-policy conservative --grad-clip-norm 1.0 --ema-decay 0.999 --early-stopping-patience 35 --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb
```

Evaluation:

```powershell
python -B evaluate_phase2_model_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --model-path E:\ResUpNet\runs\resupnet_torch_cuda_8gb\checkpoints\best_model.pt --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_tta_post --tta --postprocess --min-component-size 32
```

Plots:

```powershell
python -B generate_phase2_plots.py --run-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb --evaluation-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_tta_post
```
