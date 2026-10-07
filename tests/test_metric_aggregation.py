"""Regression tests for the canonical ResUpNet metric protocol."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from segmentation_metrics import (
    PRIMARY_METRIC_AGGREGATION,
    aggregate_confusion_rows,
    metrics_from_confusion_counts,
    micro_metrics_from_arrays,
)
from threshold_optimizer import compute_metrics_at_threshold


class MicroAggregationTests(unittest.TestCase):
    def test_pixels_are_pooled_before_dice_is_computed(self):
        y_true = np.zeros((2, 10), dtype=np.float32)
        y_pred = np.zeros_like(y_true)
        y_true[0, 0] = 1
        y_pred[0, 0] = 1
        y_true[1, :9] = 1

        micro = micro_metrics_from_arrays(y_true, y_pred)
        per_slice_macro = np.mean(
            [micro_metrics_from_arrays(y_true[i], y_pred[i])["dice"] for i in range(2)]
        )

        self.assertAlmostEqual(micro["dice"], 2.0 / 11.0)
        self.assertAlmostEqual(per_slice_macro, 0.5)
        self.assertNotAlmostEqual(micro["dice"], per_slice_macro)

    def test_aggregating_rows_matches_flattening_all_pixels(self):
        rows = [
            {"tp": 1, "fp": 0, "fn": 0, "tn": 9},
            {"tp": 0, "fp": 0, "fn": 9, "tn": 1},
        ]

        aggregated = aggregate_confusion_rows(rows)

        self.assertEqual(aggregated["count"], 2)
        self.assertEqual((aggregated["tp"], aggregated["fp"], aggregated["fn"], aggregated["tn"]), (1, 0, 9, 10))
        self.assertAlmostEqual(aggregated["dice"], 2.0 / 11.0)

    def test_empty_masks_have_one_consistent_policy(self):
        expected = metrics_from_confusion_counts(tp=0, fp=0, fn=0, tn=4)
        y_empty = np.zeros((1, 2, 2, 1), dtype=np.float32)
        threshold_metrics = compute_metrics_at_threshold(y_empty, y_empty, threshold=0.5)

        for key in ("dice", "iou", "precision", "recall", "f1", "specificity", "accuracy"):
            self.assertEqual(expected[key], 1.0)
            self.assertEqual(threshold_metrics[key], expected[key])

    def test_binary_micro_f1_is_identical_to_dice(self):
        metrics = metrics_from_confusion_counts(tp=17, fp=3, fn=5, tn=101)

        self.assertEqual(metrics["f1"], metrics["dice"])
        self.assertAlmostEqual(metrics["dice"], 2.0 * metrics["iou"] / (1.0 + metrics["iou"]))

    def test_protocol_name_is_stable_for_output_consumers(self):
        self.assertEqual(PRIMARY_METRIC_AGGREGATION, "micro_over_all_pixels")

    def test_training_and_evaluation_wrappers_use_canonical_counts(self):
        from evaluate_phase2_model_torch import aggregate_confusion
        from train_phase2_resupnet_torch import metrics_from_counts

        rows = [
            {"tp": 1, "fp": 0, "fn": 0, "tn": 9},
            {"tp": 0, "fp": 0, "fn": 9, "tn": 1},
        ]
        expected = metrics_from_confusion_counts(tp=1, fp=0, fn=9, tn=10)
        evaluated = aggregate_confusion(rows, "test fixture pixels")
        trained = metrics_from_counts(tp=1, fp=0, fn=9, tn=10)

        for key, value in expected.items():
            self.assertEqual(evaluated[key], value)
            self.assertEqual(trained[key], value)
        self.assertEqual(evaluated["aggregation"], PRIMARY_METRIC_AGGREGATION)


class MetricDocumentationTests(unittest.TestCase):
    def test_user_facing_docs_point_to_the_canonical_protocol(self):
        project_root = Path(__file__).resolve().parents[1]
        documents = (
            "README.md",
            "PROJECT_STATUS.md",
            "PHASE2_STRATEGY.md",
            "RESEARCH_TRAINING_STRATEGY.md",
            "TRAINING_COMMANDS_PHASE2.md",
            "RESUPNET_AUDIT.md",
            "experiments/v2_multimodal_roi/README.md",
            "reports/phase2_metrics_validation/RESUPNET_PHASE2_RESULTS_REPORT.md",
        )

        for relative_path in documents:
            with self.subTest(document=relative_path):
                content = (project_root / relative_path).read_text(encoding="utf-8")
                self.assertIn("METRICS_PROTOCOL.md", content)

    def test_protocol_document_matches_the_code_contract(self):
        project_root = Path(__file__).resolve().parents[1]
        protocol = (project_root / "METRICS_PROTOCOL.md").read_text(encoding="utf-8")

        self.assertIn(f"`{PRIMARY_METRIC_AGGREGATION}`", protocol)
        self.assertIn("Patient-level averaging: not used", protocol)
        self.assertIn("Dice       = 2TP / (2TP + FP + FN)", protocol)


if __name__ == "__main__":
    unittest.main()
