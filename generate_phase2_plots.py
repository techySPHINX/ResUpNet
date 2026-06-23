"""Generate Phase 2 ResUpNet plots from PyTorch training/evaluation outputs.

This plotting script is intentionally post-processing only. It reads existing
history/evaluation files and never modifies training behavior or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from resupnet_runtime_paths import configure_runtime_paths

configure_runtime_paths()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Phase 2 ResUpNet result plots.")
    parser.add_argument("--run-dir", required=True, help="Training run directory under E:/ResUpNet/runs.")
    parser.add_argument("--evaluation-dir", default=None, help="Directory containing evaluation CSV/JSON outputs.")
    parser.add_argument("--output-dir", default=None, help="Plot output directory. Defaults to <run-dir>/plots.")
    parser.add_argument("--title", default="PyTorch ResUpNet Phase 2")
    return parser.parse_args()


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    return plt, sns


def _read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")


def plot_training_curves(history_path: Path, output_dir: Path, title: str):
    history = _read_json(history_path)
    if not history:
        return []
    plt, _ = _setup_matplotlib()
    df = pd.DataFrame(history)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Combined loss")
    axes[0].legend()
    axes[1].plot(df["epoch"], df["train_dice"], label="train")
    axes[1].plot(df["epoch"], df["val_dice"], label="validation")
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    fig.suptitle(title)
    path = output_dir / "brats_training_curves.png"
    _save(fig, path)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["lr"], marker="o")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    path = output_dir / "brats_learning_rate.png"
    _save(fig, path)
    plt.close(fig)
    paths.append(path)
    return paths


def plot_threshold_curves(eval_dir: Path, output_dir: Path):
    plt, _ = _setup_matplotlib()
    paths = []
    for name in ("validation_threshold_search", "test_threshold_search"):
        path = eval_dir / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for metric in ("dice", "iou", "f1"):
            axes[0].plot(df["threshold"], df[metric], label=metric.upper())
        axes[0].set_title(name.replace("_", " ").title())
        axes[0].set_xlabel("Threshold")
        axes[0].set_ylabel("Score")
        axes[0].legend()
        axes[1].plot(1.0 - df["specificity"], df["recall"], label="ROC-like")
        axes[1].plot(df["recall"], df["precision"], label="PR-like")
        axes[1].set_xlabel("FPR or Recall")
        axes[1].set_ylabel("Recall or Precision")
        axes[1].legend()
        out = output_dir / f"brats_{name}_roc_pr_curves.png"
        _save(fig, out)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_per_sample_metrics(metrics_path: Path, output_dir: Path):
    if not metrics_path.exists():
        return []
    plt, sns = _setup_matplotlib()
    df = pd.read_csv(metrics_path)
    numeric_metrics = ["dice", "iou", "precision", "recall", "f1", "specificity", "hd95", "asd"]
    paths = []

    fig, ax = plt.subplots(figsize=(8, 6))
    counts = np.array([[df["tn"].sum(), df["fp"].sum()], [df["fn"].sum(), df["tp"].sum()]])
    sns.heatmap(counts, annot=True, fmt=".0f", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"], ax=ax)
    ax.set_title("Aggregate Pixel Confusion Matrix")
    out = output_dir / "brats_confusion_matrix.png"
    _save(fig, out)
    plt.close(fig)
    paths.append(out)

    available = [m for m in numeric_metrics if m in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, metric in zip(axes.ravel(), available):
        sns.histplot(df[metric].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title(metric.upper())
    for ax in axes.ravel()[len(available) :]:
        ax.axis("off")
    out = output_dir / "brats_metrics_distribution.png"
    _save(fig, out)
    plt.close(fig)
    paths.append(out)

    fig, ax = plt.subplots(figsize=(12, 6))
    violin_df = df[[m for m in ["dice", "iou", "precision", "recall", "f1"] if m in df.columns]].melt(var_name="metric", value_name="value")
    sns.violinplot(data=violin_df, x="metric", y="value", ax=ax, cut=0)
    ax.set_title("Metric Violin Plots")
    out = output_dir / "brats_violin_plots.png"
    _save(fig, out)
    plt.close(fig)
    paths.append(out)

    corr_cols = [m for m in available if df[m].notna().any()]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="vlag", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Metric Correlation")
    out = output_dir / "brats_metric_correlation.png"
    _save(fig, out)
    plt.close(fig)
    paths.append(out)

    true_pixels = df["true_pixels"].astype(float)
    pred_pixels = df["pred_pixels"].astype(float)
    mean_pixels = (true_pixels + pred_pixels) / 2.0
    diff_pixels = pred_pixels - true_pixels
    mean_diff = diff_pixels.mean()
    std_diff = diff_pixels.std()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(mean_pixels, diff_pixels, s=12, alpha=0.45)
    ax.axhline(mean_diff, color="black", linestyle="-", label="mean diff")
    ax.axhline(mean_diff + 1.96 * std_diff, color="red", linestyle="--", label="+1.96 SD")
    ax.axhline(mean_diff - 1.96 * std_diff, color="red", linestyle="--", label="-1.96 SD")
    ax.set_title("Bland-Altman: Tumor Pixel Area")
    ax.set_xlabel("Mean true/pred tumor pixels")
    ax.set_ylabel("Predicted - true tumor pixels")
    ax.legend()
    out = output_dir / "brats_bland_altman_analysis.png"
    _save(fig, out)
    plt.close(fig)
    paths.append(out)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=df, x="true_pixels", y="dice", hue="empty_true", s=20, alpha=0.55, ax=axes[0])
    axes[0].set_title("Dice vs Tumor Size")
    if "hd95" in df.columns:
        sns.scatterplot(data=df, x="true_pixels", y="hd95", hue="empty_true", s=20, alpha=0.55, ax=axes[1])
        axes[1].set_title("HD95 vs Tumor Size")
    else:
        axes[1].axis("off")
    out = output_dir / "brats_error_analysis.png"
    _save(fig, out)
    plt.close(fig)
    paths.append(out)

    return paths


def plot_summary_bars(summary_path: Path, output_dir: Path, title: str):
    summary = _read_json(summary_path)
    if not summary:
        return []
    plt, _ = _setup_matplotlib()
    tumor = summary.get("tumor_test_rows", {})
    metrics = ["dice", "iou", "precision", "recall", "f1", "specificity"]
    values = [tumor.get(m, {}).get("mean", math.nan) for m in metrics]
    if all(math.isnan(v) for v in values):
        return []
    paths = []

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([m.upper() for m in metrics], values)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}: Tumor-Row Mean Metrics")
    out = output_dir / "model_comparison_bar_chart.png"
    _save(fig, out)
    plt.close(fig)
    paths.append(out)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_closed, values_closed, linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.2)
    ax.set_xticks(angles)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title("Metric Radar")
    out = output_dir / "model_comparison_radar_chart.png"
    _save(fig, out)
    plt.close(fig)
    paths.append(out)
    return paths


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    eval_dir = Path(args.evaluation_dir) if args.evaluation_dir else run_dir / "evaluation_tta_post"
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []
    created += plot_training_curves(run_dir / "logs" / "history.json", output_dir, args.title)
    created += plot_threshold_curves(eval_dir, output_dir)
    created += plot_per_sample_metrics(eval_dir / "test_per_sample_metrics.csv", output_dir)
    created += plot_summary_bars(eval_dir / "evaluation_summary.json", output_dir, args.title)

    manifest = {"run_dir": str(run_dir), "evaluation_dir": str(eval_dir), "plots": [str(path) for path in created]}
    with (output_dir / "plot_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
