"""Generate clearly labeled demo ResUpNet training outputs.

These files are for presentation layout and format demonstration only. They are
not produced by model training and must not be reported as experimental results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


DISCLAIMER = (
    "SIMULATED_DEMO_ONLY_NOT_A_REAL_TRAINING_RESULT. "
    "Use only to demonstrate expected output format/trend. "
    "Do not cite or report these values as trained model performance."
)


def smooth_curve(epoch: int, start: float, end: float, k: float = 0.085) -> float:
    x = 1.0 - math.exp(-k * epoch)
    y = start + (end - start) * x
    wiggle = 0.004 * math.sin(epoch * 0.65) + 0.002 * math.sin(epoch * 1.7)
    return max(min(y + wiggle, end + 0.003), start)


def decreasing_curve(epoch: int, start: float, end: float, k: float = 0.075) -> float:
    x = math.exp(-k * epoch)
    y = end + (start - end) * x
    wiggle = 0.08 * math.sin(epoch * 0.4)
    return max(y + wiggle, end)


def row(epoch: int) -> dict:
    val_dice = smooth_curve(epoch, 0.055, 0.907, 0.075)
    train_dice = smooth_curve(epoch, 0.075, 0.934, 0.082)
    val_iou = val_dice / (2.0 - val_dice)
    train_iou = train_dice / (2.0 - train_dice)
    val_precision = smooth_curve(epoch, 0.100, 0.922, 0.071)
    val_recall = smooth_curve(epoch, 0.160, 0.895, 0.077)
    train_precision = smooth_curve(epoch, 0.120, 0.944, 0.078)
    train_recall = smooth_curve(epoch, 0.180, 0.923, 0.081)
    val_f1 = (2 * val_precision * val_recall) / max(val_precision + val_recall, 1e-8)
    train_f1 = (2 * train_precision * train_recall) / max(train_precision + train_recall, 1e-8)
    return {
        "epoch": epoch,
        "source": "simulated_demo_not_real_training_result",
        "train_loss": round(decreasing_curve(epoch, 0.92, 0.135, 0.078), 6),
        "train_dice": round(train_dice, 6),
        "train_iou": round(train_iou, 6),
        "train_precision": round(train_precision, 6),
        "train_recall": round(train_recall, 6),
        "train_f1": round(train_f1, 6),
        "train_specificity": round(smooth_curve(epoch, 0.905, 0.997, 0.060), 6),
        "train_accuracy": round(smooth_curve(epoch, 0.890, 0.994, 0.068), 6),
        "val_loss": round(decreasing_curve(epoch, 0.96, 0.165, 0.070), 6),
        "val_dice": round(val_dice, 6),
        "val_iou": round(val_iou, 6),
        "val_precision": round(val_precision, 6),
        "val_recall": round(val_recall, 6),
        "val_f1": round(val_f1, 6),
        "val_specificity": round(smooth_curve(epoch, 0.900, 0.996, 0.055), 6),
        "val_accuracy": round(smooth_curve(epoch, 0.885, 0.992, 0.062), 6),
        "val_hd95": round(decreasing_curve(epoch, 17.5, 4.15, 0.060), 4),
        "val_asd": round(decreasing_curve(epoch, 6.2, 1.25, 0.066), 4),
        "lr": 0.0001 if epoch < 35 else 0.00005,
    }


def make_notebook(history_json: Path) -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ResUpNet Demo Training Output\n",
                    "\n",
                    f"**{DISCLAIMER}**\n",
                    "\n",
                    "This notebook demonstrates how training metrics may be displayed. Replace this demo file with a real `history.json` before research reporting."
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    f"history_path = Path(r\"{history_json}\")\n",
                    "payload = json.loads(history_path.read_text())\n",
                    "print(payload['disclaimer'])\n",
                    "history = pd.DataFrame(payload['history'])\n",
                    "display(history.tail())"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(2, 4, figsize=(20, 9))\n",
                    "for ax, metric in zip(axes.ravel(), ['loss', 'dice', 'iou', 'precision', 'recall', 'f1', 'specificity', 'accuracy']):\n",
                    "    ax.plot(history['epoch'], history[f'train_{metric}'], label='train')\n",
                    "    ax.plot(history['epoch'], history[f'val_{metric}'], label='validation')\n",
                    "    ax.set_title(metric.upper())\n",
                    "    ax.set_xlabel('Epoch')\n",
                    "    ax.grid(True, alpha=0.3)\n",
                    "    ax.legend()\n",
                    "plt.tight_layout()"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "best = history.loc[history['val_dice'].idxmax()]\n",
                    "display(best.to_frame('best_epoch_value'))"
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    history = [row(epoch) for epoch in range(1, 51)]
    payload = {
        "disclaimer": DISCLAIMER,
        "description": "Expected-looking ResUpNet metric trend for UI/report-format demonstration only.",
        "target_final_epoch": {
            "epoch": 50,
            "val_dice": history[-1]["val_dice"],
            "val_iou": history[-1]["val_iou"],
            "val_precision": history[-1]["val_precision"],
            "val_recall": history[-1]["val_recall"],
            "val_f1": history[-1]["val_f1"],
            "val_specificity": history[-1]["val_specificity"],
            "val_accuracy": history[-1]["val_accuracy"],
            "val_hd95": history[-1]["val_hd95"],
            "val_asd": history[-1]["val_asd"],
        },
        "history": history,
    }
    history_json = output_dir / "expected_training_curve_demo.json"
    history_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    list_json = output_dir / "expected_training_history_rows_demo.json"
    list_json.write_text(json.dumps(history, indent=2), encoding="utf-8")
    notebook = output_dir / "expected_training_curve_demo.ipynb"
    notebook.write_text(json.dumps(make_notebook(history_json.resolve()), indent=2), encoding="utf-8")
    print(f"Wrote {history_json}")
    print(f"Wrote {list_json}")
    print(f"Wrote {notebook}")


if __name__ == "__main__":
    main()
