# ResUpNet Metric Computation Protocol

## Protocol status

- Protocol version: `1.0`
- Primary aggregation: `micro_over_all_pixels`
- Evaluation unit: every pixel in every selected/capped 2D test slice
- Patient-level averaging: not used
- Reference implementation: `segmentation_metrics.py`

This document is the source of truth for metric computation in the active
ResUpNet pipeline. Training, validation threshold selection, final test
evaluation, plots, and reports must use these definitions.

## Scope and prediction rule

The current task is binary whole-tumor segmentation on selected 2D slices.
Ground-truth values greater than `0.5` are foreground. A prediction is
foreground only when its probability is strictly greater than the selected
threshold:

```text
predicted foreground = probability > threshold
```

Training metrics use a fixed threshold of `0.5`. Final evaluation selects one
threshold using the validation split only, then applies that unchanged
threshold to the test split.

## Primary micro-aggregation procedure

For the primary result, the evaluator does not compute a Dice score for each
slice and then average those scores. Instead, it performs these steps:

1. Threshold every prediction.
2. Count true-positive (`TP`), false-positive (`FP`), false-negative (`FN`),
   and true-negative (`TN`) pixels for every selected slice.
3. Sum each count across the complete evaluated population.
4. Compute all primary metrics once from the pooled counts.

If there are `S` selected slices, the pooled counts are:

```text
TP = sum(TP_s for s = 1..S)
FP = sum(FP_s for s = 1..S)
FN = sum(FN_s for s = 1..S)
TN = sum(TN_s for s = 1..S)
```

The primary metrics are then:

```text
Dice       = 2TP / (2TP + FP + FN)
IoU        = TP / (TP + FP + FN)
Precision  = TP / (TP + FP)
Recall     = TP / (TP + FN)
F1         = 2TP / (2TP + FP + FN)
Specificity= TN / (TN + FP)
Accuracy   = (TP + TN) / (TP + TN + FP + FN)
```

For a single binary foreground class, micro-averaged F1 and Dice are the same
quantity. Dice and IoU also satisfy `Dice = 2IoU / (1 + IoU)`, apart from
display rounding.

## Empty-set policy

The implementation does not add an epsilon that changes reported scores.
Undefined ratios follow these explicit rules:

| Situation | Result |
| --- | --- |
| No foreground in truth or prediction | Dice, IoU, precision, recall, and F1 are `1` |
| Foreground exists in truth but no foreground is predicted | Dice, IoU, precision, recall, and F1 are `0` |
| Truth is empty but false-positive foreground is predicted | Dice, IoU, precision, and F1 are `0`; recall is `1` |
| No negative pixels exist and no false positive is possible | Specificity is `1` |
| Evaluated array contains no pixels | Accuracy is `1` |

The complete selected-slice validation and test populations normally contain
both foreground and background, so these rules mainly affect slice-level
diagnostics and small smoke tests.

## Training and checkpoint selection

During each training and validation epoch, batch-level confusion counts are
accumulated. Hard Dice, IoU, precision, recall, F1, specificity, and accuracy
are computed once from the epoch totals. Therefore, `train_dice`, `val_dice`,
and the other hard metrics are pixel-micro metrics over all pixels processed in
that epoch.

`val_dice` is the checkpoint-selection and learning-rate-scheduler metric.
`soft_dice` is different: it is computed from probabilities per image and then
averaged. It is a loss diagnostic and must not be reported as the canonical
hard micro Dice.

If `--steps-per-epoch` or `--validation-steps` limits an epoch, the metric
population contains only the pixels processed in those steps. Normal full
validation leaves `--validation-steps` unset.

## Threshold selection and test evaluation

`evaluate_phase2_model_torch.py` evaluates thresholds from `0.10` through
`0.90` in steps of `0.01`. Each candidate is scored by pooling all validation
pixels. The default selection metric is F1. The chosen validation threshold is
then applied once to the test predictions.

The canonical result is:

```text
evaluation_summary.json -> primary_test_metrics
```

`primary_test_metrics` pools all pixels in all selected/capped test slices.
`global_tumor_test_metrics` is a secondary micro result restricted to slices
whose ground-truth masks contain tumor.

## Slice-level and boundary diagnostics

`test_per_sample_metrics.csv` contains one row per selected 2D slice. The
`all_test_rows`, `tumor_test_rows`, and `empty_true_test_rows` blocks summarize
those row-level values with unweighted slice-level means, standard deviations,
medians, and quartiles. These are diagnostic macro summaries and are not the
primary result.

HD95 and ASD are calculated on each slice because they cannot be recovered from
pooled TP/FP/FN/TN counts. Their reported summary mean is therefore a macro mean
over slices with defined distances. Slices where exactly one of truth or
prediction is empty have undefined HD95/ASD and are excluded; the output records
the valid slice count.

## Patient handling

Patients are separated before slice extraction into non-overlapping training,
validation, and test splits. This prevents patient leakage. It does not mean
metrics are averaged per patient: `patient_id` is retained only for traceability
and diagnostic grouping.

The protocol is not official BraTS full-volume evaluation. The current arrays
contain selected/capped 2D slices at `160x160`, so results must be described as
selected-slice binary whole-tumor metrics.

## Output metadata contract

New training `run_config.json` files record:

- `metric_protocol_version`
- `hard_metric_aggregation`
- `hard_metric_population`
- `hard_prediction_rule`
- `hard_metric_threshold`

New `evaluation_summary.json` files record:

- `metric_protocol_version`
- `metric_protocol_document`
- `primary_metric_aggregation`
- `primary_metric_population`
- `hard_prediction_rule`
- `secondary_metric_aggregation`
- `boundary_metric_aggregation`
- `patient_level_averaging`

Plots displaying the primary result must read `primary_test_metrics`. Per-slice
plots must be labeled as diagnostic distributions.

## Historical artifact boundary

The checked-in `resupnet_training_curve.json` and `training_history_rows.json`
contain aggregate values but not the underlying predictions or confusion
counts. They therefore support training-curve consistency checks but do not let
an independent reviewer reconstruct their aggregation. In particular, their
recorded F1 and Dice are not identical, so those historical values must not be
relabeled as verified canonical pixel-micro results.

Only a new run produced by the active pipeline, with its `run_config.json`,
checkpoint, predictions, per-slice counts, and `evaluation_summary.json`, can be
reported as verified under protocol version `1.0`.
