"""Validate Phase 2 result artifacts and generate proof plots/report.

This script is intentionally post-processing only. It reads the two checked-in
JSON result artifacts and produces a reproducible validation summary, figures,
and a Markdown report under reports/phase2_metrics_validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CURVE = ROOT / "resupnet_training_curve.json"
DEFAULT_HISTORY = ROOT / "training_history_rows.json"
DEFAULT_OUT = ROOT / "reports" / "phase2_metrics_validation"

SCORE_METRICS = ("dice", "iou", "precision", "recall", "f1", "specificity", "accuracy")
LOWER_IS_BETTER = ("loss", "hd95", "asd")
CONTEXT_PAPER_ROWS = [
    {
        "name": "ResUpNet current artifact",
        "metric": "Val WT Dice, selected 2D slices",
        "dice": 0.890146,
        "hd95": 4.8877,
        "protocol": "BraTS 2021 selected-slice validation, binary WT",
        "source": "resupnet_training_curve.json",
        "claim": "Primary local artifact result. Strong selected-slice evidence, not official full-volume benchmark evidence.",
    },
    {
        "name": "nnU-Net BraTS 2020",
        "metric": "WT Dice",
        "dice": 0.8895,
        "hd95": 8.498,
        "protocol": "BraTS 2020 competition, full-volume 3D ensemble",
        "source": "https://arxiv.org/abs/2011.00848",
        "claim": "Numerically similar WT Dice and lower HD95 in our artifact, but protocols differ.",
    },
    {
        "name": "Triplanar U-Net ensemble",
        "metric": "WT Dice",
        "dice": 0.89,
        "hd95": None,
        "protocol": "BraTS 2020 unseen test, triplanar ensemble",
        "source": "https://arxiv.org/abs/2105.11356",
        "claim": "Numerically comparable WT Dice, protocol differs.",
    },
    {
        "name": "Self-ensembled 3D U-Net",
        "metric": "WT Dice",
        "dice": 0.89,
        "hd95": 6.7,
        "protocol": "BraTS 2020 final test, 3D ensemble",
        "source": "https://arxiv.org/abs/2011.01045",
        "claim": "Numerically comparable WT Dice and lower HD95 in our artifact, protocol differs.",
    },
    {
        "name": "Residual Transformer ensemble",
        "metric": "Mean Dice",
        "dice": 0.876,
        "hd95": None,
        "protocol": "BraTS 2021 3D mean Dice across regions",
        "source": "https://arxiv.org/abs/2308.00128",
        "claim": "Our selected-slice Dice is numerically higher than this mean Dice, but mean-region and WT Dice are not interchangeable.",
    },
    {
        "name": "BiTr-Unet",
        "metric": "WT Dice",
        "dice": 0.9257,
        "hd95": 3.0,
        "protocol": "BraTS 2021 testing, CNN-transformer 3D method",
        "source": "https://arxiv.org/abs/2109.12271",
        "claim": "Published full-volume result is stronger; use as upper context, not as a paper we beat.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ResUpNet result artifacts and generate report.")
    parser.add_argument("--curve-json", default=str(DEFAULT_CURVE), help="Full training curve JSON.")
    parser.add_argument("--history-json", default=str(DEFAULT_HISTORY), help="Compact training history JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT), help="Output directory for report and plots.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(float(a) - float(b)) <= tol


def metric_value(row: dict[str, Any], split: str, metric: str) -> float | None:
    group = row.get(split, {})
    value = group.get(metric)
    return None if value is None else float(value)


def score_range_ok(value: float) -> bool:
    return -1e-9 <= value <= 1.0 + 1e-9


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def best_metric(block: dict[str, Any], metric: str) -> float | None:
    flat_key = f"val_{metric}"
    if flat_key in block:
        return float(block[flat_key])
    nested = block.get("validation", {})
    if metric in nested:
        return float(nested[metric])
    return None


def best_block(best: dict[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        block = best.get(name)
        if isinstance(block, dict):
            return block
    return {}


def check_epoch_subset(expected: dict[str, Any], actual: dict[str, Any], path: str, checks: list[dict[str, Any]]) -> None:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if isinstance(expected_value, dict):
            actual_value = actual.get(key, {})
            if not isinstance(actual_value, dict):
                mismatches.append(f"{path}.{key}")
                continue
            check_epoch_subset(expected_value, actual_value, f"{path}.{key}", checks)
        elif isinstance(expected_value, (int, float)):
            actual_value = actual.get(key)
            if actual_value is None or not close(float(expected_value), float(actual_value), 1e-6):
                mismatches.append(f"{path}.{key}: expected {expected_value}, computed {actual_value}")
        else:
            if actual.get(key) != expected_value:
                mismatches.append(f"{path}.{key}: expected {expected_value}, computed {actual.get(key)}")
    if path in ("summary.initial_epoch", "summary.final_epoch") and not mismatches:
        add_check(checks, f"{path} matches curve", True, "All nested values match the corresponding epoch row.")
    elif mismatches:
        add_check(checks, f"{path} matches curve", False, "; ".join(mismatches[:8]))


def compact_epoch_to_full(epoch: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {"epoch": epoch["epoch"]}
    for split in ("train", "validation"):
        for metric, value in epoch.get(split, {}).items():
            prefix = "train" if split == "train" else "val"
            flat[f"{prefix}_{metric}"] = value
    flat["lr"] = epoch.get("optimizer", {}).get("learning_rate")
    return flat


def validate_artifacts(curve: dict[str, Any], compact: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    epochs = curve.get("epochs", [])
    compact_epochs = compact.get("epochs", [])

    add_check(checks, "full curve JSON parsed", isinstance(curve, dict), "Loaded resupnet_training_curve.json.")
    add_check(checks, "compact history JSON parsed", isinstance(compact, dict), "Loaded training_history_rows.json.")
    add_check(checks, "full curve has 50 epochs", len(epochs) == 50, f"Found {len(epochs)} full-curve epochs.")

    epoch_numbers = [int(row.get("epoch", -1)) for row in epochs]
    expected_numbers = list(range(1, len(epochs) + 1))
    add_check(
        checks,
        "epoch sequence is continuous",
        epoch_numbers == expected_numbers,
        f"Epoch sequence is {epoch_numbers[:3]}...{epoch_numbers[-3:]}." if epochs else "No epochs found.",
    )

    out_of_range: list[str] = []
    identity_failures: list[str] = []
    for row in epochs:
        epoch = int(row["epoch"])
        lr = row.get("optimizer", {}).get("learning_rate")
        if lr is None or float(lr) <= 0:
            out_of_range.append(f"epoch {epoch} learning_rate={lr}")
        for split in ("train", "validation"):
            for metric in SCORE_METRICS:
                value = metric_value(row, split, metric)
                if value is not None and not score_range_ok(value):
                    out_of_range.append(f"epoch {epoch} {split}.{metric}={value}")
            for metric in LOWER_IS_BETTER:
                value = metric_value(row, split, metric)
                if value is not None and value < -1e-9:
                    out_of_range.append(f"epoch {epoch} {split}.{metric}={value}")

            p = metric_value(row, split, "precision")
            r = metric_value(row, split, "recall")
            f1 = metric_value(row, split, "f1")
            if p is not None and r is not None and f1 is not None:
                expected_f1 = 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)
                if not close(f1, expected_f1, 2e-5):
                    identity_failures.append(f"epoch {epoch} {split}.f1 expected {expected_f1:.6f}, got {f1:.6f}")

            dice = metric_value(row, split, "dice")
            iou = metric_value(row, split, "iou")
            if dice is not None and iou is not None:
                expected_dice = 0.0 if (1.0 + iou) == 0 else 2.0 * iou / (1.0 + iou)
                if not close(dice, expected_dice, 3e-5):
                    identity_failures.append(f"epoch {epoch} {split}.dice expected {expected_dice:.6f}, got {dice:.6f}")

    add_check(checks, "metric ranges are valid", not out_of_range, "All score metrics are in [0, 1], loss/distances are non-negative, and LR is positive." if not out_of_range else "; ".join(out_of_range[:8]))
    add_check(checks, "F1 and Dice/IoU identities hold", not identity_failures, "F1 equals 2PR/(P+R), and Dice equals 2IoU/(1+IoU), within rounding tolerance." if not identity_failures else "; ".join(identity_failures[:8]))

    if epochs:
        summary = curve.get("summary", {})
        check_epoch_subset(summary.get("initial_epoch", {}), epochs[0], "summary.initial_epoch", checks)
        check_epoch_subset(summary.get("final_epoch", {}), epochs[-1], "summary.final_epoch", checks)

        initial = epochs[0]["validation"]
        final = epochs[-1]["validation"]
        progress = summary.get("overall_progress", {})
        progress_expected = {
            "val_dice_improvement": final["dice"] - initial["dice"],
            "val_iou_improvement": final["iou"] - initial["iou"],
            "val_loss_reduction": initial["loss"] - final["loss"],
            "val_hd95_reduction": initial["hd95"] - final["hd95"],
            "val_asd_reduction": initial["asd"] - final["asd"],
        }
        progress_failures = []
        for key, expected in progress_expected.items():
            if key not in progress or not close(progress[key], expected, 1e-6):
                progress_failures.append(f"{key}: expected {expected:.6f}, got {progress.get(key)}")
        add_check(checks, "overall progress recomputes", not progress_failures, "All improvement/reduction values recompute from epoch 1 and epoch 50." if not progress_failures else "; ".join(progress_failures))

        gap = summary.get("final_generalization_gap", {})
        gap_expected = {
            "train_minus_val_dice": epochs[-1]["train"]["dice"] - final["dice"],
            "train_minus_val_iou": epochs[-1]["train"]["iou"] - final["iou"],
            "train_minus_val_f1": epochs[-1]["train"]["f1"] - final["f1"],
        }
        gap_failures = []
        for key, expected in gap_expected.items():
            if key not in gap or not close(gap[key], expected, 1e-6):
                gap_failures.append(f"{key}: expected {expected:.6f}, got {gap.get(key)}")
        add_check(checks, "final generalization gap recomputes", not gap_failures, "Final train-minus-validation gaps match the full curve." if not gap_failures else "; ".join(gap_failures))

        best = curve.get("best_checkpoints", {})
        best_failures = []
        best_dice = max(epochs, key=lambda row: row["validation"]["dice"])
        best_iou = max(epochs, key=lambda row: row["validation"]["iou"])
        best_hd95 = min(epochs, key=lambda row: row["validation"]["hd95"])
        best_asd = min(epochs, key=lambda row: row["validation"]["asd"])
        min_loss = min(row["validation"]["loss"] for row in epochs)
        loss_epochs = [row["epoch"] for row in epochs if close(row["validation"]["loss"], min_loss, 1e-8)]
        best_dice_block = best_block(best, "best_by_validation_dice")
        best_iou_block = best_block(best, "best_by_validation_iou")
        best_hd95_block = best_block(best, "best_by_validation_hd95", "best_by_hd95")
        best_asd_block = best_block(best, "best_by_validation_asd", "best_by_asd")
        if best_dice_block.get("epoch") != best_dice["epoch"]:
            best_failures.append("best_by_validation_dice epoch mismatch")
        if best_iou_block.get("epoch") != best_iou["epoch"]:
            best_failures.append("best_by_validation_iou epoch mismatch")
        if best_hd95_block.get("epoch") != best_hd95["epoch"]:
            best_failures.append("best_by_hd95 epoch mismatch")
        if best_asd_block.get("epoch") != best_asd["epoch"]:
            best_failures.append("best_by_asd epoch mismatch")
        loss_summary = best.get("best_by_validation_loss", {})
        if loss_summary.get("epochs") != loss_epochs or not close(loss_summary.get("val_loss", math.nan), min_loss, 1e-8):
            best_failures.append("best_by_validation_loss mismatch")
        add_check(checks, "best checkpoint claims recompute", not best_failures, f"Best Dice/IoU/HD95 at epoch {best_dice['epoch']}; best loss at epochs {loss_epochs}; best ASD at epoch {best_asd['epoch']}." if not best_failures else "; ".join(best_failures))

    compact_count = len(compact_epochs)
    if compact_count != len(epochs):
        warnings.append(
            f"training_history_rows.json contains {compact_count} selected epoch rows, not the full {len(epochs)}-epoch curve. Use resupnet_training_curve.json for plots and trend analysis."
        )
    shared_failures: list[str] = []
    epoch_by_number = {int(row["epoch"]): row for row in epochs}
    for row in compact_epochs:
        epoch_number = int(row["epoch"])
        if epoch_number not in epoch_by_number:
            shared_failures.append(f"compact epoch {epoch_number} missing in full curve")
            continue
        full_flat = compact_epoch_to_full(epoch_by_number[epoch_number])
        compact_flat = compact_epoch_to_full(row)
        for key, compact_value in compact_flat.items():
            full_value = full_flat.get(key)
            if compact_value is None and full_value is None:
                continue
            if isinstance(compact_value, (int, float)) and isinstance(full_value, (int, float)):
                if not close(compact_value, full_value, 1e-6):
                    shared_failures.append(f"epoch {epoch_number} {key}: compact {compact_value}, full {full_value}")
            elif compact_value != full_value:
                shared_failures.append(f"epoch {epoch_number} {key}: compact {compact_value}, full {full_value}")
    add_check(checks, "compact rows match full curve for shared epochs", not shared_failures, "Epochs present in both artifacts have matching metric values." if not shared_failures else "; ".join(shared_failures[:8]))

    info_full = curve.get("experiment_info", {})
    info_compact = compact.get("trained_info", {})
    common_keys = sorted(set(info_full).intersection(info_compact) - {"metric_directions", "metric_goal"})
    info_failures = [key for key in common_keys if info_full.get(key) != info_compact.get(key)]
    add_check(checks, "experiment metadata matches across artifacts", not info_failures, "Shared experiment metadata fields are consistent." if not info_failures else f"Mismatched fields: {', '.join(info_failures)}")

    final = epochs[-1]["validation"] if epochs else {}
    train_final = epochs[-1]["train"] if epochs else {}
    first = epochs[0]["validation"] if epochs else {}
    values = {
        "final_epoch": epochs[-1]["epoch"] if epochs else None,
        "initial_val_dice": first.get("dice"),
        "final_val_dice": final.get("dice"),
        "final_train_dice": train_final.get("dice"),
        "final_val_iou": final.get("iou"),
        "final_val_loss": final.get("loss"),
        "final_val_precision": final.get("precision"),
        "final_val_recall": final.get("recall"),
        "final_val_f1": final.get("f1"),
        "final_val_hd95": final.get("hd95"),
        "final_val_asd": final.get("asd"),
        "final_train_minus_val_dice": (train_final.get("dice") - final.get("dice")) if final and train_final else None,
        "val_dice_absolute_gain": (final.get("dice") - first.get("dice")) if final and first else None,
        "val_dice_relative_gain_pct": ((final.get("dice") - first.get("dice")) / first.get("dice") * 100.0) if final and first and first.get("dice") else None,
        "val_loss_reduction_pct": ((first.get("loss") - final.get("loss")) / first.get("loss") * 100.0) if final and first and first.get("loss") else None,
        "val_hd95_reduction_pct": ((first.get("hd95") - final.get("hd95")) / first.get("hd95") * 100.0) if final and first and first.get("hd95") else None,
        "val_asd_reduction_pct": ((first.get("asd") - final.get("asd")) / first.get("asd") * 100.0) if final and first and first.get("asd") else None,
    }

    if epochs:
        last10 = epochs[-10:]
        val_dice_last10 = [row["validation"]["dice"] for row in last10]
        values["last10_val_dice_mean"] = mean(val_dice_last10)
        values["last10_val_dice_population_std"] = pstdev(val_dice_last10)
        values["last10_val_dice_delta"] = val_dice_last10[-1] - val_dice_last10[0]
        values["epoch41_to_50_val_dice_gain"] = epochs[-1]["validation"]["dice"] - epochs[40]["validation"]["dice"]

    passed = all(check["passed"] for check in checks)
    return {"passed": passed, "checks": checks, "warnings": warnings, "computed": values, "paper_context": CONTEXT_PAPER_ROWS}


def setup_matplotlib(output_dir: Path):
    mpl_cache = output_dir / "_matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def series(epochs: list[dict[str, Any]], split: str, metric: str) -> list[float]:
    return [float(row[split][metric]) for row in epochs]


def plot_training(epochs: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    plt = setup_matplotlib(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    x = [row["epoch"] for row in epochs]
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(x, series(epochs, "train", "loss"), label="Train", linewidth=2)
    axes[0].plot(x, series(epochs, "validation", "loss"), label="Validation", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Combined loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, series(epochs, "train", "dice"), label="Train", linewidth=2)
    axes[1].plot(x, series(epochs, "validation", "dice"), label="Validation", linewidth=2)
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("ResUpNet Phase 2 Training Curves")
    path = plot_dir / "training_loss_dice.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ("dice", "iou", "precision", "recall", "f1"):
        ax.plot(x, series(epochs, "validation", metric), label=metric.upper(), linewidth=2)
    ax.set_title("Validation Overlap and Classification Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3)
    ax.grid(alpha=0.3)
    path = plot_dir / "validation_metrics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(x, series(epochs, "validation", "hd95"), color="#b83232", linewidth=2)
    axes[0].set_title("Validation HD95")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("HD95")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, series(epochs, "validation", "asd"), color="#355c9a", linewidth=2)
    axes[1].set_title("Validation ASD")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ASD")
    axes[1].grid(alpha=0.3)
    path = plot_dir / "boundary_metrics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ("dice", "iou", "f1"):
        gaps = [row["train"][metric] - row["validation"][metric] for row in epochs]
        ax.plot(x, gaps, label=f"{metric.upper()} gap", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Train Minus Validation Gap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap")
    ax.legend()
    ax.grid(alpha=0.3)
    path = plot_dir / "generalization_gap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(x, [row["optimizer"]["learning_rate"] for row in epochs], where="post", linewidth=2)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.grid(alpha=0.3)
    path = plot_dir / "learning_rate_schedule.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [row["name"] for row in CONTEXT_PAPER_ROWS]
    values = [row["dice"] for row in CONTEXT_PAPER_ROWS]
    colors = ["#1f7a5a" if idx == 0 else "#60758f" for idx, _ in enumerate(labels)]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0.70, 0.96)
    ax.set_ylabel("Dice")
    ax.set_title("Published Context: Mixed Protocols, Not a Leaderboard Claim")
    ax.tick_params(axis="x", rotation=35)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.004, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    path = plot_dir / "published_context_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}%"


def pass_label(value: bool) -> str:
    return "PASS" if value else "FAIL"


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def build_report(curve: dict[str, Any], compact: dict[str, Any], validation: dict[str, Any], output_dir: Path, plot_paths: list[Path]) -> str:
    epochs = curve["epochs"]
    info = curve.get("experiment_info", {})
    computed = validation["computed"]
    first = epochs[0]
    final = epochs[-1]
    best = curve.get("best_checkpoints", {})
    checks = validation["checks"]
    warnings = validation["warnings"]
    created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    plot_lookup = {path.name: rel(path, output_dir) for path in plot_paths}
    status = "validated" if validation["passed"] else "has validation failures"

    lines: list[str] = []
    lines.append("# ResUpNet Phase 2 Metrics Validation Report")
    lines.append("")
    lines.append(f"Generated: {created}")
    lines.append("")
    lines.append("## Executive Verdict")
    lines.append("")
    lines.append(f"The two result artifacts are **{status}** under artifact-level checks. The full 50-epoch curve supports a final validation Dice of **{fmt(computed['final_val_dice'], 6)}**, IoU of **{fmt(computed['final_val_iou'], 6)}**, F1 of **{fmt(computed['final_val_f1'], 6)}**, HD95 of **{fmt(computed['final_val_hd95'], 4)}**, and ASD of **{fmt(computed['final_val_asd'], 4)}** at epoch **{computed['final_epoch']}**.")
    lines.append("")
    lines.append("This is strong internal selected-slice evidence. It is not, by itself, proof of official BraTS full-volume superiority because the current protocol uses 2D selected slices at 160x160 and binary whole-tumor masks.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append("| Artifact | Role | Status |")
    lines.append("| --- | --- | --- |")
    lines.append("| `resupnet_training_curve.json` | Full 50-epoch training and validation curve | Primary source for validation and plots |")
    lines.append("| `training_history_rows.json` | Compact selected-row summary | Cross-check source for epochs 1, 2, and 50 |")
    lines.append("")
    lines.append("## Experiment Context")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    for key in ("experiment_name", "task", "model", "dataset", "input_type", "image_size", "total_epochs", "created_at"):
        lines.append(f"| {key} | {info.get(key, 'n/a')} |")
    lines.append("")
    lines.append("## Validation Checks")
    lines.append("")
    lines.append("| Check | Result | Detail |")
    lines.append("| --- | --- | --- |")
    for check in checks:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | {pass_label(check['passed'])} | {detail} |")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append("## Metric Summary")
    lines.append("")
    lines.append("| Metric | Epoch 1 validation | Epoch 50 validation | Absolute change | Relative change |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    comparison_rows = [
        ("Dice", first["validation"]["dice"], final["validation"]["dice"], final["validation"]["dice"] - first["validation"]["dice"], ((final["validation"]["dice"] - first["validation"]["dice"]) / first["validation"]["dice"] * 100.0)),
        ("IoU", first["validation"]["iou"], final["validation"]["iou"], final["validation"]["iou"] - first["validation"]["iou"], ((final["validation"]["iou"] - first["validation"]["iou"]) / first["validation"]["iou"] * 100.0)),
        ("Loss", first["validation"]["loss"], final["validation"]["loss"], final["validation"]["loss"] - first["validation"]["loss"], ((final["validation"]["loss"] - first["validation"]["loss"]) / first["validation"]["loss"] * 100.0)),
        ("HD95", first["validation"]["hd95"], final["validation"]["hd95"], final["validation"]["hd95"] - first["validation"]["hd95"], ((final["validation"]["hd95"] - first["validation"]["hd95"]) / first["validation"]["hd95"] * 100.0)),
        ("ASD", first["validation"]["asd"], final["validation"]["asd"], final["validation"]["asd"] - first["validation"]["asd"], ((final["validation"]["asd"] - first["validation"]["asd"]) / first["validation"]["asd"] * 100.0)),
    ]
    for name, start, end, delta, rel_delta in comparison_rows:
        lines.append(f"| {name} | {fmt(start, 6)} | {fmt(end, 6)} | {fmt(delta, 6)} | {pct(rel_delta)} |")
    lines.append("")
    lines.append("For lower-is-better metrics, the negative relative change means improvement. The reductions are loss **{loss}**, HD95 **{hd95}**, and ASD **{asd}**.".format(
        loss=pct(computed["val_loss_reduction_pct"]),
        hd95=pct(computed["val_hd95_reduction_pct"]),
        asd=pct(computed["val_asd_reduction_pct"]),
    ))
    lines.append("")
    lines.append("## Final Epoch Quality")
    lines.append("")
    lines.append("| Split | Loss | Dice | IoU | Precision | Recall | F1 | Specificity | Accuracy | HD95 | ASD |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    train = final["train"]
    val = final["validation"]
    lines.append(f"| Train | {fmt(train['loss'], 6)} | {fmt(train['dice'], 6)} | {fmt(train['iou'], 6)} | {fmt(train['precision'], 6)} | {fmt(train['recall'], 6)} | {fmt(train['f1'], 6)} | {fmt(train['specificity'], 6)} | {fmt(train['accuracy'], 6)} | n/a | n/a |")
    lines.append(f"| Validation | {fmt(val['loss'], 6)} | {fmt(val['dice'], 6)} | {fmt(val['iou'], 6)} | {fmt(val['precision'], 6)} | {fmt(val['recall'], 6)} | {fmt(val['f1'], 6)} | {fmt(val['specificity'], 6)} | {fmt(val['accuracy'], 6)} | {fmt(val['hd95'], 4)} | {fmt(val['asd'], 4)} |")
    lines.append("")
    lines.append(f"Final train-minus-validation Dice gap is **{fmt(computed['final_train_minus_val_dice'], 6)}**. That gap is small enough to support a stable internal-validation interpretation, while still requiring test/full-volume confirmation before publication-grade claims.")
    lines.append("")
    lines.append("## Best Checkpoints")
    lines.append("")
    lines.append("| Selection rule | Artifact result | Interpretation |")
    lines.append("| --- | --- | --- |")
    best_dice_block = best_block(best, "best_by_validation_dice")
    best_iou_block = best_block(best, "best_by_validation_iou")
    best_loss_block = best_block(best, "best_by_validation_loss")
    best_hd95_block = best_block(best, "best_by_validation_hd95", "best_by_hd95")
    best_asd_block = best_block(best, "best_by_validation_asd", "best_by_asd")
    lines.append(f"| Best validation Dice | Epoch {best_dice_block.get('epoch')}, Dice {fmt(best_metric(best_dice_block, 'dice'), 6)} | Best overlap checkpoint |")
    lines.append(f"| Best validation IoU | Epoch {best_iou_block.get('epoch')}, IoU {fmt(best_metric(best_iou_block, 'iou'), 6)} | Same checkpoint as best Dice |")
    lines.append(f"| Best validation loss | Epochs {best_loss_block.get('epochs')}, loss {fmt(best_loss_block.get('val_loss'), 6)} | Loss plateau precedes best Dice; select by Dice for segmentation overlap |")
    lines.append(f"| Best HD95 | Epoch {best_hd95_block.get('epoch')}, HD95 {fmt(best_metric(best_hd95_block, 'hd95'), 4)} | Best boundary outlier distance |")
    lines.append(f"| Best ASD | Epoch {best_asd_block.get('epoch')}, ASD {fmt(best_metric(best_asd_block, 'asd'), 4)} | Lowest average surface distance, slightly before final Dice peak |")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for name, caption in [
        ("training_loss_dice.png", "Loss and Dice curves from the full 50-epoch artifact."),
        ("validation_metrics.png", "Validation Dice, IoU, precision, recall, and F1."),
        ("boundary_metrics.png", "Validation HD95 and ASD boundary quality."),
        ("generalization_gap.png", "Train-minus-validation gaps for Dice, IoU, and F1."),
        ("learning_rate_schedule.png", "Learning-rate schedule captured in the artifact."),
        ("published_context_comparison.png", "Contextual comparison against selected papers. Mixed protocols; not a leaderboard claim."),
    ]:
        if name in plot_lookup:
            lines.append(f"![{caption}]({plot_lookup[name]})")
            lines.append("")
    lines.append("## Why The Present Structure Can Produce Better Results")
    lines.append("")
    lines.append("The current result is plausible because the current native PyTorch structure is materially stronger than the earlier project baseline and many simple 2D U-Net style setups:")
    lines.append("")
    lines.append("- Four MRI modalities are used together: T1, T1ce, T2, and FLAIR. This gives the model complementary contrast information instead of forcing it to infer tumor extent from a single channel.")
    lines.append("- The split is patient-wise with no overlap, which removes a common leakage failure mode in slice-based medical imaging experiments.")
    lines.append("- The input pipeline uses image-intensity ROI cropping, not mask-based cropping, so the crop does not leak label geometry while still reducing irrelevant background.")
    lines.append("- Tumor, near-tumor, and hard-negative slices are retained. That helps the model learn boundary ambiguity and false-positive suppression, not only obvious tumor slices.")
    lines.append("- The model is a residual encoder-decoder with attention gates and an ASPP bottleneck. Residual blocks improve gradient flow, attention gates filter skip features, and ASPP adds multi-scale context for variable tumor sizes.")
    lines.append("- The active loss combines Dice, focal Tversky, boundary loss, and BCE. That directly targets overlap, class imbalance, boundary quality, and pixel-level calibration together.")
    lines.append("- The trainer supports conservative MRI augmentation, gradient clipping, mixed precision, EMA checkpoints, and resume-safe optimizer/scheduler/scaler state. These controls improve stability on native Windows CUDA hardware.")
    lines.append("")
    lines.append("## Published Context")
    lines.append("")
    lines.append("| Work | Reported metric | Reported value | Our artifact value | What can be said |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for row in CONTEXT_PAPER_ROWS[1:]:
        lines.append(f"| [{row['name']}]({row['source']}) | {row['metric']} | {fmt(row['dice'], 4)} | {fmt(CONTEXT_PAPER_ROWS[0]['dice'], 4)} | {row['claim']} |")
    lines.append("")
    lines.append("The strongest defensible statement is: **under the current selected-slice validation protocol, ResUpNet reaches a Dice value that is numerically competitive with several published whole-tumor Dice results and stronger than the project's earlier internal baselines, while using a native PyTorch pipeline tuned for the available system.**")
    lines.append("")
    lines.append("The strongest statement that is **not** yet defensible is: **this is better than all BraTS papers or official full-volume BraTS state of the art.** BiTr-Unet, for example, reports stronger BraTS 2021 full-volume WT Dice and HD95 than this artifact.")
    lines.append("")
    lines.append("## Native-System Limitations")
    lines.append("")
    lines.append("- The current artifacts validate a 2D selected-slice protocol, not full 3D patient-volume inference.")
    lines.append("- Input size is 160x160 because of local storage and VRAM constraints; this may lose fine boundary detail compared with 192, 224, or 256 crops.")
    lines.append("- The task is binary whole-tumor segmentation, not full BraTS subregion segmentation for ET, TC, and WT.")
    lines.append("- The artifact-level proof does not include the matching checkpoint, run directory, evaluator output, or test-set summary. Those are required for publication-grade reproducibility.")
    lines.append("- The validation loss minimum occurs before the final Dice maximum, so final checkpoint selection should explicitly prioritize Dice/IoU if overlap quality is the main objective.")
    lines.append("")
    lines.append("## Claim Boundary")
    lines.append("")
    lines.append("Safe claim:")
    lines.append("")
    lines.append("> The checked-in Phase 2 result artifacts are internally consistent and show validation Dice improving from 0.120967 to 0.890146 over 50 epochs under the project's BraTS 2021 selected-slice binary whole-tumor protocol.")
    lines.append("")
    lines.append("Safe competitive-positioning claim:")
    lines.append("")
    lines.append("> The selected-slice validation Dice is numerically competitive with several published whole-tumor Dice values, but this is a contextual comparison only because published BraTS papers generally use full-volume challenge protocols.")
    lines.append("")
    lines.append("Unsafe claim until more evidence exists:")
    lines.append("")
    lines.append("> This checkpoint is the best BraTS 2021 model overall, or it beats official full-volume BraTS 2021 methods.")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("Regenerate this report and all plots with:")
    lines.append("")
    lines.append("```powershell")
    lines.append(r".\.venv\Scripts\python.exe generate_phase2_artifact_report.py")
    lines.append("```")
    lines.append("")
    lines.append("Outputs:")
    lines.append("")
    lines.append("- `reports/phase2_metrics_validation/metrics_validation_summary.json`")
    lines.append("- `reports/phase2_metrics_validation/RESUPNET_PHASE2_RESULTS_REPORT.md`")
    lines.append("- `reports/phase2_metrics_validation/plots/*.png`")
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append("- BraTS 2021 benchmark: https://arxiv.org/abs/2107.02314")
    lines.append("- nnU-Net for Brain Tumor Segmentation: https://arxiv.org/abs/2011.00848")
    lines.append("- Triplanar ensemble of U-Nets: https://arxiv.org/abs/2105.11356")
    lines.append("- Self-ensembled deeply-supervised 3D U-Net: https://arxiv.org/abs/2011.01045")
    lines.append("- BiTr-Unet: https://arxiv.org/abs/2109.12271")
    lines.append("- Residual Transformer ensemble: https://arxiv.org/abs/2308.00128")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    curve_path = Path(args.curve_json)
    history_path = Path(args.history_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curve = read_json(curve_path)
    compact = read_json(history_path)
    validation = validate_artifacts(curve, compact)
    plot_paths = plot_training(curve["epochs"], output_dir)

    summary_path = output_dir / "metrics_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)

    report = build_report(curve, compact, validation, output_dir, plot_paths)
    report_path = output_dir / "RESUPNET_PHASE2_RESULTS_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Validation: {pass_label(validation['passed'])}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    for path in plot_paths:
        print(f"Wrote {path}")
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
