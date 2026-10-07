# Research Strategy

Date: 2026-10-07

## Positioning

The core architecture remains `ResUpNetTorch`: residual encoder-decoder,
attention-gated skip connections, and ASPP bottleneck. The pre-training changes
below are training, checkpointing, and evaluation upgrades. They do not change
the network topology.

Use the claim language carefully:

- Safe before training: "The pipeline is designed to improve robustness and
  selected-slice segmentation performance through stronger training controls."
- Safe for the current checked-in artifacts: "The internally consistent
  artifact curve records 0.890146 Dice / 0.802039 IoU / 4.8877 HD95 under a
  selected-slice validation setup; its aggregation cannot be reconstructed
  from the artifact alone."
- Not safe: "This is the official full-volume BraTS state of the art."

## Historical Artifact Result

`resupnet_training_curve.json` is the primary historical curve source. The synchronized
compact artifact, `training_history_rows.json`, matches the full curve for its
shared rows but only contains epochs `1`, `2`, and `50`.

Recorded epoch-50 selected-slice validation metrics:

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

Generated proof report:

```text
reports\phase2_metrics_validation\RESUPNET_PHASE2_RESULTS_REPORT.md
```

The active canonical computation is defined in
[`METRICS_PROTOCOL.md`](METRICS_PROTOCOL.md). The historical JSON files do not
contain predictions or confusion counts and therefore do not prove that these
recorded values used protocol version `1.0`.

## Implemented Pre-Training

1. Conservative MRI augmentation

   The default augmentation now uses left-right flips, mild affine transforms,
   modality-wise intensity scaling/shift, and low Gaussian noise. The old
   flip/rot90/noise policy remains available as `--augmentation-policy legacy`.
2. Gradient clipping

   `--grad-clip-norm 1.0` is enabled by default. This reduces unstable updates
   when focal Tversky, boundary loss, and mixed precision interact.
3. EMA checkpointing

   Exponential moving average weights are maintained by default with
   `--ema-decay 0.999`. The trainer saves both `best_model.pt` and
   `best_ema_model.pt`. EMA does not alter inference architecture; it selects a
   smoother set of the same model weights.
4. Research-grade resume

   Checkpoints now include model, optimizer, scheduler, scaler, best Dice,
   bad-epoch count, and history. A resumed long run no longer silently restarts
   optimizer state.
5. Stronger run manifest

   `run_config.json` now records augmentation policy, EMA settings, gradient
   clipping, early stopping, dropout, weight decay, trainable parameter count,
   reproducibility mode, and the hard-metric protocol metadata.
6. Traceable evaluation

   Evaluation output now includes patient ID, slice index, slice class, and
   original tumor pixels for each test row when `slice_metadata.json` is
   present. `primary_test_metrics` is the canonical result and is micro-averaged
   from pooled confusion counts across all selected test-slice pixels. Per-row
   summaries are retained only as diagnostics; no patient-level averaging is
   performed.

## Recommended Main Run

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 150 --batch-size 8 --base-filters 32 --mixed-precision --augmentation-policy conservative --grad-clip-norm 1.0 --ema-decay 0.999 --early-stopping-patience 35 --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb
```

If VRAM is tight:

```powershell
python -B train_phase2_resupnet_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --epochs 150 --batch-size 4 --base-filters 24 --mixed-precision --augmentation-policy conservative --grad-clip-norm 1.0 --ema-decay 0.999 --early-stopping-patience 35 --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_safe
```

## Evaluation Plan

Evaluate the raw best checkpoint:

```powershell
python -B evaluate_phase2_model_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --model-path E:\ResUpNet\runs\resupnet_torch_cuda_8gb\checkpoints\best_model.pt --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_raw_tta_post --tta --postprocess --min-component-size 32
```

Evaluate the EMA checkpoint if it exists:

```powershell
python -B evaluate_phase2_model_torch.py --data-dir experiments\v2_multimodal_roi\processed_splits --model-path E:\ResUpNet\runs\resupnet_torch_cuda_8gb\checkpoints\best_ema_model.pt --output-dir E:\ResUpNet\runs\resupnet_torch_cuda_8gb\evaluation_ema_tta_post --tta --postprocess --min-component-size 32
```

Select the final reported checkpoint using validation-selected threshold and
test evaluation output, not by looking at test threshold maxima.

## Result Criteria

Report these metrics in this order:

- primary pixel-micro test Dice, IoU, precision, recall, F1, specificity, accuracy
- secondary tumor-slice pixel-micro metrics
- diagnostic per-slice macro summaries
- per-slice HD95 and ASD distributions and valid-slice counts
- empty-true false positive rows
- selected threshold and threshold metric
- raw checkpoint versus EMA checkpoint
- exact checkpoint epoch and run configuration

## Claim Boundary

The current dataset is a selected-slice 2D protocol. A strong selected-slice
Dice is useful for this project, but it is not the same as official BraTS
full-volume Dice. For benchmark-style claims, add a full-volume/patient-level
evaluation path before comparing with published BraTS results.
