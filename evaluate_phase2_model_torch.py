"""Evaluate native PyTorch ResUpNet checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt, label

from resupnet_runtime_paths import configure_runtime_paths, default_runs_root
from resupnet_torch_model import ResUpNetTorch
from threshold_optimizer import compute_metrics_at_threshold

RUNTIME_PATHS = configure_runtime_paths()


def _mask_array(mask) -> np.ndarray:
    return np.squeeze(np.asarray(mask) > 0.5)


def dice_np(y_true, y_pred, smooth=1e-6):
    yt = _mask_array(y_true)
    yp = _mask_array(y_pred)
    intersection = np.logical_and(yt, yp).sum()
    return (2.0 * intersection + smooth) / (yt.sum() + yp.sum() + smooth)


def iou_np(y_true, y_pred, smooth=1e-6):
    yt = _mask_array(y_true)
    yp = _mask_array(y_pred)
    intersection = np.logical_and(yt, yp).sum()
    union = np.logical_or(yt, yp).sum()
    return (intersection + smooth) / (union + smooth)


def confusion_counts(y_true, y_pred):
    yt = _mask_array(y_true)
    yp = _mask_array(y_pred)
    tp = int(np.logical_and(yt, yp).sum())
    fp = int(np.logical_and(~yt, yp).sum())
    fn = int(np.logical_and(yt, ~yp).sum())
    tn = int(np.logical_and(~yt, ~yp).sum())
    return tp, fp, fn, tn


def confusion_metrics(y_true, y_pred, smooth=1e-6):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2.0 * precision * recall + smooth) / (precision + recall + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    return float(precision), float(recall), float(f1), float(specificity)


def _surface_distances(y_true, y_pred):
    yt = _mask_array(y_true)
    yp = _mask_array(y_pred)
    if not yt.any() and not yp.any():
        return np.array([0.0], dtype=np.float32)
    if not yt.any() or not yp.any():
        return np.array([math.nan], dtype=np.float32)
    true_surface = np.logical_xor(yt, binary_erosion(yt))
    pred_surface = np.logical_xor(yp, binary_erosion(yp))
    dist_to_pred = distance_transform_edt(~pred_surface)
    dist_to_true = distance_transform_edt(~true_surface)
    distances = np.concatenate([dist_to_pred[true_surface], dist_to_true[pred_surface]])
    return distances.astype(np.float32)


def hd95_np(y_true, y_pred):
    distances = _surface_distances(y_true, y_pred)
    if np.isnan(distances).any():
        return None
    return float(np.percentile(distances, 95))


def asd_np(y_true, y_pred):
    distances = _surface_distances(y_true, y_pred)
    if np.isnan(distances).any():
        return None
    return float(np.mean(distances))


def postprocess_mask(mask, min_component_size=32):
    binary = _mask_array(mask)
    labeled, count = label(binary)
    cleaned = np.zeros_like(binary, dtype=bool)
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if component.sum() >= min_component_size:
            cleaned |= component
    cleaned = binary_fill_holes(cleaned)
    return cleaned.astype(np.float32)[..., None]


def summarize(rows):
    if not rows:
        return {}
    summary = {"count": len(rows)}
    metric_keys = ["dice", "iou", "precision", "recall", "f1", "specificity", "hd95", "asd"]
    for key in metric_keys:
        values = [row[key] for row in rows if row.get(key) is not None and not math.isnan(float(row[key]))]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch ResUpNet model.")
    parser.add_argument("--data-dir", default="experiments/v2_multimodal_roi/processed_splits")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--min-component-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional evaluation smoke-test limit.")
    parser.add_argument("--threshold-metric", choices=("f1", "dice", "iou"), default="f1")
    return parser.parse_args()


def load_arrays(data_dir: Path):
    arrays = {}
    for name in ("X_val", "y_val", "X_test", "y_test"):
        path = data_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        arrays[name] = np.load(path, mmap_mode="r")
    return arrays


def limit_arrays(arrays: dict[str, np.ndarray], max_samples: int | None):
    if max_samples is None:
        return arrays
    limited = dict(arrays)
    for name in ("X_val", "y_val", "X_test", "y_test"):
        limited[name] = arrays[name][:max_samples]
    return limited


def load_split_metadata(data_dir: Path) -> dict[str, list[dict]]:
    path = data_dir / "slice_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    split_rows = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row.get("split")
        if split in split_rows:
            split_rows[split].append(row)
    return split_rows


def limit_metadata(metadata: dict[str, list[dict]], max_samples: int | None) -> dict[str, list[dict]]:
    if max_samples is None:
        return metadata
    return {key: value[:max_samples] for key, value in metadata.items()}


class EvalDataset:
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        import torch

        x = np.asarray(self.X[index], dtype=np.float32)
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x), index


def predict(model, X, batch_size, device, tta=False):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(EvalDataset(X), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    preds = np.zeros((len(X), X.shape[1], X.shape[2], 1), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for xb, indices in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            prob = torch.sigmoid(logits)
            if tta:
                lr = torch.flip(torch.sigmoid(model(torch.flip(xb, dims=[3]))), dims=[3])
                ud = torch.flip(torch.sigmoid(model(torch.flip(xb, dims=[2]))), dims=[2])
                prob = (prob + lr + ud) / 3.0
            prob = prob.detach().cpu().numpy()
            prob = np.transpose(prob, (0, 2, 3, 1))
            preds[np.asarray(indices)] = prob
    return preds


def find_threshold(y_true, y_prob, optimize_for="f1"):
    thresholds = np.round(np.linspace(0.10, 0.90, 81), 3)
    rows = []
    best = None
    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_prob, threshold)
        row = {
            "threshold": float(threshold),
            "dice": float(metrics["dice"]),
            "iou": float(metrics["iou"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "specificity": float(metrics["specificity"]),
        }
        rows.append(row)
        if best is None or row[optimize_for] > best[optimize_for]:
            best = row
    best = dict(best)
    best["optimized_for"] = optimize_for
    return best, rows


def evaluate_per_sample(y_true, y_prob, threshold, postprocess=False, min_component_size=32, metadata=None):
    rows = []
    for i in range(len(y_true)):
        true_mask = np.asarray(y_true[i], dtype=np.float32)
        pred_mask = (y_prob[i] > threshold).astype(np.float32)
        if postprocess:
            pred_mask = postprocess_mask(pred_mask, min_component_size)
        tp, fp, fn, tn = confusion_counts(true_mask, pred_mask)
        precision, recall, f1, specificity = confusion_metrics(true_mask, pred_mask)
        true_pixels = float(true_mask.sum())
        pred_pixels = float(pred_mask.sum())
        row = {
            "index": i,
            "true_pixels": true_pixels,
            "pred_pixels": pred_pixels,
            "empty_true": true_pixels == 0,
            "empty_pred": pred_pixels == 0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice": float(dice_np(true_mask, pred_mask)),
            "iou": float(iou_np(true_mask, pred_mask)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "hd95": hd95_np(true_mask, pred_mask),
            "asd": asd_np(true_mask, pred_mask),
        }
        if metadata is not None and i < len(metadata):
            item = metadata[i]
            row.update(
                {
                    "patient_id": item.get("patient_id"),
                    "slice_index": item.get("slice_index"),
                    "slice_class": item.get("slice_class"),
                    "tumor_pixels_original": item.get("tumor_pixels_original"),
                }
            )
        rows.append(row)
    return rows


def aggregate_confusion(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    tp = int(sum(row["tp"] for row in rows))
    fp = int(sum(row["fp"] for row in rows))
    fn = int(sum(row["fn"] for row in rows))
    tn = int(sum(row["tn"] for row in rows))
    smooth = 1e-6
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    f1 = (2.0 * precision * recall + smooth) / (precision + recall + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    return {
        "count": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    import torch

    checkpoint = torch.load(args.model_path, map_location="cpu")
    config = checkpoint.get("config", {})
    input_shape = config.get("input_shape", [160, 160, 4])
    model = ResUpNetTorch(
        in_channels=int(input_shape[-1]),
        base_filters=int(config.get("base_filters", 32)),
        dropout=float(config.get("dropout", 0.20)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir) if args.output_dir else default_runs_root() / "evaluations" / model_path.parent.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = limit_arrays(load_arrays(data_dir), args.max_samples)
    metadata = limit_metadata(load_split_metadata(data_dir), args.max_samples)
    val_prob = predict(model, arrays["X_val"], args.batch_size, device, tta=args.tta)
    best_threshold, threshold_rows = find_threshold(arrays["y_val"], val_prob, optimize_for=args.threshold_metric)
    threshold = best_threshold["threshold"]
    write_csv(output_dir / "validation_threshold_search.csv", threshold_rows)

    test_prob = predict(model, arrays["X_test"], args.batch_size, device, tta=args.tta)
    _, test_threshold_rows = find_threshold(arrays["y_test"], test_prob, optimize_for=args.threshold_metric)
    write_csv(output_dir / "test_threshold_search.csv", test_threshold_rows)
    rows = evaluate_per_sample(
        arrays["y_test"],
        test_prob,
        threshold,
        postprocess=args.postprocess,
        min_component_size=args.min_component_size,
        metadata=metadata.get("test"),
    )
    write_csv(output_dir / "test_per_sample_metrics.csv", rows)

    tumor_rows = [row for row in rows if not row["empty_true"]]
    empty_rows = [row for row in rows if row["empty_true"]]
    summary = {
        "backend": "native_windows_torch_cuda" if device.type == "cuda" else "native_windows_torch_cpu",
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "evaluation_protocol": "selected_slice_2d",
        "protocol_note": "Current arrays are selected/capped 2D slices, not full-volume official BraTS evaluation.",
        "threshold_source": "validation",
        "threshold_metric": args.threshold_metric,
        "selected_threshold": best_threshold,
        "tta": args.tta,
        "postprocess": args.postprocess,
        "max_samples": args.max_samples,
        "runtime_paths": RUNTIME_PATHS,
        "all_test_rows": summarize(rows),
        "tumor_test_rows": summarize(tumor_rows),
        "empty_true_test_rows": summarize(empty_rows),
        "global_test_metrics": aggregate_confusion(rows),
        "global_tumor_test_metrics": aggregate_confusion(tumor_rows),
        "empty_true_false_positive_rows": int(sum(row["empty_true"] and row["pred_pixels"] > 0 for row in rows)),
        "counts": {
            "test_rows": len(rows),
            "tumor_rows": len(tumor_rows),
            "empty_true_rows": len(empty_rows),
            "metadata_rows": len(metadata.get("test", [])),
        },
    }
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["counts"], indent=2))
    print(f"Selected threshold: {threshold}")
    print(f"Summary: {output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
