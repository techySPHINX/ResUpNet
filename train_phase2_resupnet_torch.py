"""Native Windows CUDA PyTorch trainer for Phase 2 ResUpNet."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from resupnet_runtime_paths import configure_runtime_paths, default_runs_root
from resupnet_torch_model import ResUpNetTorch, combined_loss, dice_score_from_logits

RUNTIME_PATHS = configure_runtime_paths()


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResUpNet with native Windows PyTorch CUDA.")
    parser.add_argument("--data-dir", default="experiments/v2_multimodal_roi/processed_splits")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-filters", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", default=None, help="Optional .pt checkpoint to continue model weights from.")
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true", default=True)
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentation.")
    parser.add_argument(
        "--augmentation-policy",
        choices=("conservative", "legacy", "none"),
        default="conservative",
        help="Conservative uses mild affine/intensity MRI augmentation; legacy keeps the older flip/rot90/noise policy.",
    )
    parser.add_argument("--augment-noise-std", type=float, default=0.025)
    parser.add_argument("--no-balanced-batches", action="store_true")
    parser.add_argument("--sampler-positive-fraction", type=float, default=0.70)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=35)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-val-start-epoch", type=int, default=5)
    parser.add_argument("--ema-val-interval", type=int, default=1)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="Favor reproducibility over peak CuDNN speed.")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=1.0,
        help="Print live phase metrics every N seconds. Use 1 for terminal updates every second.",
    )
    parser.add_argument(
        "--progress-format",
        choices=("line", "json"),
        default="line",
        help="Use compact dotted terminal lines or raw JSON progress records.",
    )
    return parser.parse_args()


def load_splits(data_dir: Path):
    arrays = {}
    for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        path = data_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        arrays[name] = np.load(path, mmap_mode="r")
    return arrays


def mask_has_tumor(y, chunk_size=1024):
    flags = np.zeros(len(y), dtype=bool)
    for start in range(0, len(y), chunk_size):
        end = min(start + chunk_size, len(y))
        chunk = np.asarray(y[start:end])
        flags[start:end] = chunk.reshape(len(chunk), -1).sum(axis=1) > 0
    return flags


def _mild_affine_pair(x, y, rng):
    import cv2

    height, width = x.shape[:2]
    center = (width * 0.5, height * 0.5)
    angle = float(rng.uniform(-7.0, 7.0))
    scale = float(rng.uniform(0.95, 1.05))
    tx = float(rng.uniform(-4.0, 4.0))
    ty = float(rng.uniform(-4.0, 4.0))
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    warped_channels = [
        cv2.warpAffine(
            x[..., channel],
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        for channel in range(x.shape[-1])
    ]
    x_warped = np.stack(warped_channels, axis=-1)
    y_warped = cv2.warpAffine(
        y[..., 0],
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )[..., None]
    return x_warped.astype(np.float32), (y_warped > 0.5).astype(np.float32)


def _conservative_augment(x, y, rng, noise_std):
    if rng.random() > 0.5:
        x = np.flip(x, axis=1).copy()
        y = np.flip(y, axis=1).copy()
    if rng.random() > 0.65:
        x, y = _mild_affine_pair(x, y, rng)
    scale = rng.uniform(0.90, 1.10, size=(1, 1, x.shape[-1])).astype(np.float32)
    shift = rng.normal(0.0, 0.025, size=(1, 1, x.shape[-1])).astype(np.float32)
    x = x * scale + shift
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return np.clip(x, -5.0, 5.0), y


def _legacy_augment(x, y, rng, noise_std):
    if rng.random() > 0.5:
        x = np.flip(x, axis=1).copy()
        y = np.flip(y, axis=1).copy()
    if rng.random() > 0.5:
        x = np.flip(x, axis=0).copy()
        y = np.flip(y, axis=0).copy()
    k = int(rng.integers(0, 4))
    if k:
        x = np.rot90(x, k, axes=(0, 1)).copy()
        y = np.rot90(y, k, axes=(0, 1)).copy()
    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return np.clip(x, -5.0, 5.0), y


class NpySegmentationDataset:
    def __init__(self, X, y, augment=False, augmentation_policy="conservative", noise_std=0.025, seed=42):
        self.X = X
        self.y = y
        self.augment = augment
        self.augmentation_policy = "none" if not augment else augmentation_policy
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = np.asarray(self.X[index], dtype=np.float32).copy()
        y = np.asarray(self.y[index], dtype=np.float32).copy()
        if self.augmentation_policy == "conservative":
            x, y = _conservative_augment(x, y, self.rng, self.noise_std)
        elif self.augmentation_policy == "legacy":
            x, y = _legacy_augment(x, y, self.rng, self.noise_std)
        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))
        import torch

        return torch.from_numpy(x), torch.from_numpy(y)


def make_loader(
    X,
    y,
    batch_size,
    training,
    augment,
    augmentation_policy,
    noise_std,
    balanced,
    sampler_positive_fraction,
    seed,
    num_workers,
):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = NpySegmentationDataset(
        X,
        y,
        augment=augment and training,
        augmentation_policy=augmentation_policy,
        noise_std=noise_std,
        seed=seed,
    )
    sampler = None
    shuffle = training
    generator = torch.Generator()
    generator.manual_seed(seed)
    if training and balanced:
        pos_fraction = min(max(float(sampler_positive_fraction), 0.05), 0.95)
        tumor_flags = mask_has_tumor(y)
        weights = np.where(
            tumor_flags,
            pos_fraction / max(tumor_flags.sum(), 1),
            (1.0 - pos_fraction) / max((~tumor_flags).sum(), 1),
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
    )


def cuda_memory_status(device):
    if device.type != "cuda":
        return {}
    import torch

    return {
        "cuda_allocated_mb": round(torch.cuda.memory_allocated(device) / 1024**2, 1),
        "cuda_reserved_mb": round(torch.cuda.memory_reserved(device) / 1024**2, 1),
        "cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
    }


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def create_ema_model(model):
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


def update_ema_model(ema_model, model, decay: float):
    if ema_model is None:
        return
    with __import__("torch").no_grad():
        ema_state = ema_model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


def batch_confusion_metrics(logits, targets, threshold=0.5, smooth=1e-6):
    import torch

    preds = torch.sigmoid(logits).detach() > threshold
    truth = targets.detach() > 0.5
    tp = torch.logical_and(preds, truth).sum().item()
    fp = torch.logical_and(preds, ~truth).sum().item()
    fn = torch.logical_and(~preds, truth).sum().item()
    tn = torch.logical_and(~preds, ~truth).sum().item()
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2.0 * precision * recall + smooth) / (precision + recall + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }


def metrics_from_counts(tp, fp, fn, tn, smooth=1e-6):
    if tp + fp + fn == 0:
        foreground_metrics = {
            "dice": 1.0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    elif tp + fn == 0:
        foreground_metrics = {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0 if fp > 0 else 1.0,
            "recall": 1.0,
            "f1": 0.0,
        }
    elif tp + fp == 0:
        foreground_metrics = {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    else:
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        foreground_metrics = {
            "dice": (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth),
            "iou": (tp + smooth) / (tp + fp + fn + smooth),
            "precision": precision,
            "recall": recall,
            "f1": (2.0 * precision * recall + smooth) / (precision + recall + smooth),
        }
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2.0 * precision * recall + smooth) / (precision + recall + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    return {
        "dice": float(foreground_metrics["dice"]),
        "iou": float(foreground_metrics["iou"]),
        "precision": float(foreground_metrics["precision"]),
        "recall": float(foreground_metrics["recall"]),
        "f1": float(foreground_metrics["f1"]),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }


def format_progress_line(progress):
    pct = 100.0 * progress["step"] / max(progress["total_steps"], 1)
    dots = "." * ((progress["step"] % 20) + 1)
    return (
        f"[{progress['phase']}] epoch {progress['epoch']} "
        f"step {progress['step']}/{progress['total_steps']} {pct:5.1f}% {dots:<20} "
        f"loss={progress['avg_loss']:.5f} "
        f"dice={progress['dice']:.5f} "
        f"iou={progress['iou']:.5f} "
        f"prec={progress['precision']:.5f} "
        f"rec={progress['recall']:.5f} "
        f"f1={progress['f1']:.5f} "
        f"spec={progress['specificity']:.5f} "
        f"acc={progress['accuracy']:.5f} "
        f"gpu={progress.get('cuda_reserved_mb', 0):.0f}MB "
        f"pid={progress['pid']}"
    )


def format_epoch_line(row):
    status = "best-updated" if row.get("best_checkpoint_updated") else "checkpoint"
    base = (
        f"[epoch_end] epoch {row['epoch']} {status} "
        f"train_loss={row['train_loss']:.5f} train_dice={row['train_dice']:.5f} "
        f"train_iou={row['train_iou']:.5f} train_prec={row['train_precision']:.5f} "
        f"train_rec={row['train_recall']:.5f} train_f1={row['train_f1']:.5f} "
        f"val_loss={row['val_loss']:.5f} val_dice={row['val_dice']:.5f} "
        f"val_iou={row['val_iou']:.5f} val_prec={row['val_precision']:.5f} "
        f"val_rec={row['val_recall']:.5f} val_f1={row['val_f1']:.5f} "
        f"best_val_dice={row['best_val_dice']:.5f} lr={row['lr']:.2e}"
    )
    if row.get("ema_val_dice") is not None:
        base += f" ema_val_dice={row['ema_val_dice']:.5f} best_ema_val_dice={row['best_ema_val_dice']:.5f}"
    return base


def save_checkpoint(path, model, config, epoch, optimizer, scheduler, scaler, best_dice, bad_epochs, history):
    import torch

    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_dice": best_dice,
            "bad_epochs": bad_epochs,
            "history": history,
        },
        path,
    )


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    training,
    mixed_precision,
    max_steps=None,
    epoch=None,
    phase="train",
    progress_interval=0.0,
    progress_format="line",
    grad_clip_norm=None,
    ema_model=None,
    ema_decay=0.999,
):
    import torch

    model.train(training)
    total_loss = 0.0
    total_soft_dice = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    steps = 0
    total_steps = min(len(loader), max_steps) if max_steps is not None else len(loader)
    last_progress = time.monotonic()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
                logits = model(x)
                loss = combined_loss(logits, y)
            if training:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                update_ema_model(ema_model, model, ema_decay)
        with torch.no_grad():
            soft_dice = dice_score_from_logits(logits.float(), y.float())
            batch_metrics = batch_confusion_metrics(logits.float(), y.float())
        total_loss += float(loss.detach().cpu())
        total_soft_dice += float(soft_dice.detach().cpu())
        total_tp += batch_metrics["tp"]
        total_fp += batch_metrics["fp"]
        total_fn += batch_metrics["fn"]
        total_tn += batch_metrics["tn"]
        steps += 1
        now = time.monotonic()
        if progress_interval > 0 and (now - last_progress >= progress_interval or steps == 1):
            running_metrics = metrics_from_counts(total_tp, total_fp, total_fn, total_tn)
            progress = {
                "event": "progress",
                "pid": os.getpid(),
                "phase": phase,
                "epoch": epoch,
                "step": steps,
                "total_steps": total_steps,
                "avg_loss": total_loss / max(steps, 1),
                "soft_dice": total_soft_dice / max(steps, 1),
                "dice": running_metrics["dice"],
                "iou": running_metrics["iou"],
                "precision": running_metrics["precision"],
                "recall": running_metrics["recall"],
                "f1": running_metrics["f1"],
                "specificity": running_metrics["specificity"],
                "accuracy": running_metrics["accuracy"],
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "tn": total_tn,
            }
            progress.update(cuda_memory_status(device))
            if progress_format == "json":
                print(json.dumps(progress), flush=True)
            else:
                print(format_progress_line(progress), flush=True)
            last_progress = now
        if max_steps is not None and steps >= max_steps:
            break
    final_metrics = metrics_from_counts(total_tp, total_fp, total_fn, total_tn)
    final_metrics.update(
        {
            "loss": total_loss / max(steps, 1),
            "soft_dice": total_soft_dice / max(steps, 1),
            "steps": steps,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
        }
    )
    return final_metrics


def main():
    args = parse_args()

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        if not args.deterministic:
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = datetime.now().strftime("torch_run_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else default_runs_root() / "torch_native_windows" / run_name
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    arrays = load_splits(Path(args.data_dir))
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]

    model = ResUpNetTorch(in_channels=X_train.shape[-1], base_filters=args.base_filters, dropout=args.dropout).to(device)
    resume_epoch = 0
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(resume_checkpoint["model"])
        resume_epoch = int(resume_checkpoint.get("epoch", 0))
        print(f"[resume] loaded={args.resume_from} previous_epoch={resume_epoch}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")
    if resume_checkpoint is not None:
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        if "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])

    ema_model = None if args.no_ema else create_ema_model(model).to(device)

    train_loader = make_loader(
        X_train,
        y_train,
        args.batch_size,
        training=True,
        augment=not args.no_augment,
        augmentation_policy="none" if args.no_augment else args.augmentation_policy,
        noise_std=args.augment_noise_std,
        balanced=not args.no_balanced_batches,
        sampler_positive_fraction=args.sampler_positive_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        X_val,
        y_val,
        args.batch_size,
        training=False,
        augment=False,
        augmentation_policy="none",
        noise_std=0.0,
        balanced=False,
        sampler_positive_fraction=args.sampler_positive_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    config = {
        "backend": "native_windows_torch_cuda" if device.type == "cuda" else "native_windows_torch_cpu",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_total_vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None,
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "input_shape": list(X_train.shape[1:]),
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "base_filters": args.base_filters,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "mixed_precision": args.mixed_precision,
        "augmentation_policy": "none" if args.no_augment else args.augmentation_policy,
        "augment_noise_std": args.augment_noise_std,
        "balanced_batches": not args.no_balanced_batches,
        "sampler_positive_fraction": args.sampler_positive_fraction,
        "grad_clip_norm": args.grad_clip_norm,
        "ema_enabled": ema_model is not None,
        "ema_decay": args.ema_decay,
        "ema_val_start_epoch": args.ema_val_start_epoch,
        "ema_val_interval": args.ema_val_interval,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "deterministic": args.deterministic,
        "trainable_parameters": count_trainable_parameters(model),
        "pid": os.getpid(),
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "best_checkpoint": str(output_dir / "checkpoints" / "best_model.pt"),
        "best_ema_checkpoint": str(output_dir / "checkpoints" / "best_ema_model.pt"),
        "last_checkpoint": str(output_dir / "checkpoints" / "last_model.pt"),
        "history_path": str(output_dir / "logs" / "history.json"),
        "progress_interval": args.progress_interval,
        "progress_format": args.progress_format,
        "resume_from": args.resume_from,
        "resume_epoch": resume_epoch,
        "runtime_paths": RUNTIME_PATHS,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(json.dumps({"event": "run_start", **config}, indent=2), flush=True)

    history = list(resume_checkpoint.get("history", [])) if resume_checkpoint is not None else []
    best_dice = float(resume_checkpoint.get("best_val_dice", -1.0)) if resume_checkpoint is not None else -1.0
    if best_dice < 0 and history:
        best_dice = max(float(row.get("val_dice", -1.0)) for row in history)
    best_ema_dice = max((float(row.get("ema_val_dice", -1.0)) for row in history), default=-1.0)
    bad_epochs = int(resume_checkpoint.get("bad_epochs", 0)) if resume_checkpoint is not None else 0
    for epoch in range(resume_epoch + 1, resume_epoch + args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            True,
            args.mixed_precision,
            args.steps_per_epoch,
            epoch=epoch,
            phase="train",
            progress_interval=args.progress_interval,
            progress_format=args.progress_format,
            grad_clip_norm=args.grad_clip_norm,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            device,
            False,
            args.mixed_precision,
            args.validation_steps,
            epoch=epoch,
            phase="validation",
            progress_interval=args.progress_interval,
            progress_format=args.progress_format,
        )
        ema_val_metrics = None
        if (
            ema_model is not None
            and epoch >= args.ema_val_start_epoch
            and args.ema_val_interval > 0
            and (epoch - args.ema_val_start_epoch) % args.ema_val_interval == 0
        ):
            ema_val_metrics = run_epoch(
                ema_model,
                val_loader,
                optimizer,
                scaler,
                device,
                False,
                args.mixed_precision,
                args.validation_steps,
                epoch=epoch,
                phase="validation_ema",
                progress_interval=args.progress_interval,
                progress_format=args.progress_format,
            )
        scheduler.step(val_metrics["dice"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_soft_dice": train_metrics["soft_dice"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_specificity": train_metrics["specificity"],
            "train_accuracy": train_metrics["accuracy"],
            "train_tp": train_metrics["tp"],
            "train_fp": train_metrics["fp"],
            "train_fn": train_metrics["fn"],
            "train_tn": train_metrics["tn"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_soft_dice": val_metrics["soft_dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_specificity": val_metrics["specificity"],
            "val_accuracy": val_metrics["accuracy"],
            "val_tp": val_metrics["tp"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_tn": val_metrics["tn"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if ema_val_metrics is not None:
            row.update(
                {
                    "ema_val_loss": ema_val_metrics["loss"],
                    "ema_val_dice": ema_val_metrics["dice"],
                    "ema_val_soft_dice": ema_val_metrics["soft_dice"],
                    "ema_val_iou": ema_val_metrics["iou"],
                    "ema_val_precision": ema_val_metrics["precision"],
                    "ema_val_recall": ema_val_metrics["recall"],
                    "ema_val_f1": ema_val_metrics["f1"],
                    "ema_val_specificity": ema_val_metrics["specificity"],
                    "ema_val_accuracy": ema_val_metrics["accuracy"],
                    "ema_val_tp": ema_val_metrics["tp"],
                    "ema_val_fp": ema_val_metrics["fp"],
                    "ema_val_fn": ema_val_metrics["fn"],
                    "ema_val_tn": ema_val_metrics["tn"],
                }
            )
        raw_best_updated = False
        ema_best_updated = False
        if val_metrics["dice"] > best_dice + args.early_stopping_min_delta:
            best_dice = val_metrics["dice"]
            bad_epochs = 0
            raw_best_updated = True
        else:
            bad_epochs += 1
        if ema_val_metrics is not None and ema_val_metrics["dice"] > best_ema_dice + args.early_stopping_min_delta:
            best_ema_dice = ema_val_metrics["dice"]
            ema_best_updated = True
        row["best_checkpoint_updated"] = raw_best_updated
        if ema_val_metrics is not None:
            row["best_ema_checkpoint_updated"] = ema_best_updated
        row["best_val_dice"] = best_dice
        row["best_ema_val_dice"] = best_ema_dice
        row["last_checkpoint"] = str(output_dir / "checkpoints" / "last_model.pt")
        row["best_checkpoint"] = str(output_dir / "checkpoints" / "best_model.pt")
        row["best_ema_checkpoint"] = str(output_dir / "checkpoints" / "best_ema_model.pt")
        history.append(row)
        if raw_best_updated:
            save_checkpoint(
                output_dir / "checkpoints" / "best_model.pt",
                model,
                config,
                epoch,
                optimizer,
                scheduler,
                scaler,
                best_dice,
                bad_epochs,
                history,
            )
        if ema_best_updated:
            save_checkpoint(
                output_dir / "checkpoints" / "best_ema_model.pt",
                ema_model,
                config,
                epoch,
                optimizer,
                scheduler,
                scaler,
                best_ema_dice,
                bad_epochs,
                history,
            )
        save_checkpoint(
            output_dir / "checkpoints" / "last_model.pt",
            model,
            config,
            epoch,
            optimizer,
            scheduler,
            scaler,
            best_dice,
            bad_epochs,
            history,
        )
        with (output_dir / "logs" / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if args.progress_format == "json":
            print(json.dumps({"event": "epoch_end", **row}), flush=True)
        else:
            print(format_epoch_line(row), flush=True)
            print(f"[checkpoint] last={row['last_checkpoint']}", flush=True)
            print(f"[checkpoint] best={row['best_checkpoint']}", flush=True)
        if bad_epochs >= args.early_stopping_patience:
            print(
                f"Early stopping: validation Dice did not improve for {args.early_stopping_patience} epochs.",
                flush=True,
            )
            break

    print(f"Training complete: {output_dir}", flush=True)
    print(f"Best model: {output_dir / 'checkpoints' / 'best_model.pt'}", flush=True)
    if ema_model is not None:
        print(f"Best EMA model: {output_dir / 'checkpoints' / 'best_ema_model.pt'}", flush=True)


if __name__ == "__main__":
    main()
