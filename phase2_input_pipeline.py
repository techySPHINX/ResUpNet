"""
Phase 2 input pipeline: multimodal MRI + image-based ROI + balanced slices.

This module is intentionally separate from the original single-modality loader so
the v1 baseline remains reproducible while v2 experiments evolve.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class Phase2InputConfig:
    modalities: tuple[str, ...] = ("t1", "t1ce", "t2", "flair")
    target_size: tuple[int, int] = (256, 256)
    binary_segmentation: bool = True
    clip_percentile: float = 99.5
    roi_crop: bool = True
    roi_margin: int = 8
    min_tumor_pixels: int = 1
    near_tumor_slices: int = 2
    tumor_fraction: float = 0.65
    near_fraction: float = 0.20
    hard_negative_fraction: float = 0.15
    max_tumor_slices_per_patient: int | None = 48
    max_near_slices_per_patient: int | None = 16
    max_hard_negative_slices_per_patient: int | None = 16
    max_empty_slices_no_tumor_patient: int = 8
    random_state: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    version: str = "v2_multimodal_roi"


def _strip_nifti_suffix(filename: str) -> str:
    name = filename.lower()
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _find_token_file(files: Sequence[str], token: str) -> str | None:
    token = token.lower()
    for filename in files:
        stem = _strip_nifti_suffix(filename).replace("-", "_")
        tokens = stem.split("_")
        if token in tokens:
            return filename
    return None


def _load_nifti(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.float32)


def _normalize_volume(volume: np.ndarray, clip_percentile: float) -> np.ndarray:
    volume = volume.astype(np.float32, copy=True)
    brain_mask = volume > 0
    if not np.any(brain_mask):
        return volume

    if clip_percentile < 100:
        upper = np.percentile(volume[brain_mask], clip_percentile)
        volume = np.clip(volume, 0, upper)
        brain_mask = volume > 0

    mean = float(volume[brain_mask].mean())
    std = float(volume[brain_mask].std())
    if std > 0:
        volume = (volume - mean) / std
        volume = np.clip(volume, -5, 5)
    return volume.astype(np.float32)


def _brain_bbox_2d(image: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    brain_mask = np.any(image != 0, axis=-1)
    coords = np.argwhere(brain_mask)
    height, width = brain_mask.shape
    if coords.size == 0:
        return 0, height, 0, width

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    y_min = max(int(y_min) - margin, 0)
    x_min = max(int(x_min) - margin, 0)
    y_max = min(int(y_max) + margin, height)
    x_max = min(int(x_max) + margin, width)
    return y_min, y_max, x_min, x_max


def _resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    width, height = target_size[1], target_size[0]
    image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if image_resized.ndim == 2:
        image_resized = np.expand_dims(image_resized, axis=-1)
    if mask_resized.ndim == 2:
        mask_resized = np.expand_dims(mask_resized, axis=-1)
    return image_resized.astype(np.float32), (mask_resized > 0.5).astype(np.float32)


def _sample_indices(indices: list[int], count: int, rng: np.random.Generator) -> list[int]:
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return list(indices)
    return sorted(rng.choice(indices, size=count, replace=False).astype(int).tolist())


def _stable_patient_seed(patient_id: str) -> int:
    return sum((idx + 1) * byte for idx, byte in enumerate(patient_id.encode("utf-8")))


def _balanced_slice_indices(
    tumor_pixels: np.ndarray,
    config: Phase2InputConfig,
    rng: np.random.Generator,
) -> tuple[list[int], dict[str, list[int]]]:
    tumor = [int(i) for i, px in enumerate(tumor_pixels) if px >= config.min_tumor_pixels]
    tumor_set = set(tumor)

    near = []
    hard_negative = []
    if tumor:
        for i, px in enumerate(tumor_pixels):
            if i in tumor_set or px > 0:
                continue
            nearest = min(abs(i - z) for z in tumor)
            if nearest <= config.near_tumor_slices:
                near.append(int(i))
            else:
                hard_negative.append(int(i))
    else:
        hard_negative = [int(i) for i in range(len(tumor_pixels))]

    if not tumor:
        selected_hard = _sample_indices(
            hard_negative,
            config.max_empty_slices_no_tumor_patient,
            rng,
        )
        categories = {"tumor": [], "near_tumor": [], "hard_negative": selected_hard}
        return selected_hard, categories

    selected_tumor = _sample_indices(
        tumor,
        config.max_tumor_slices_per_patient or len(tumor),
        rng,
    )
    target_total = max(
        len(selected_tumor),
        int(round(len(selected_tumor) / config.tumor_fraction)),
    )
    near_count = int(round(target_total * config.near_fraction))
    hard_count = int(round(target_total * config.hard_negative_fraction))
    if config.max_near_slices_per_patient is not None:
        near_count = min(near_count, config.max_near_slices_per_patient)
    if config.max_hard_negative_slices_per_patient is not None:
        hard_count = min(hard_count, config.max_hard_negative_slices_per_patient)

    selected_near = _sample_indices(near, near_count, rng)
    selected_hard = _sample_indices(hard_negative, hard_count, rng)
    selected = sorted(set(selected_tumor + selected_near + selected_hard))
    categories = {
        "tumor": selected_tumor,
        "near_tumor": selected_near,
        "hard_negative": selected_hard,
    }
    return selected, categories


def find_patient_folders(dataset_root: str | Path, modalities: Sequence[str]) -> list[Path]:
    root = Path(dataset_root)
    candidate_dirs = [root] + [path for path in root.rglob("*") if path.is_dir()]
    patient_folders = []
    for entry in sorted(candidate_dirs):
        files = os.listdir(entry)
        has_all_modalities = all(_find_token_file(files, modality) for modality in modalities)
        has_seg = _find_token_file(files, "seg") is not None
        if has_all_modalities and has_seg:
            patient_folders.append(entry)
    return patient_folders


def process_patient_folder(
    patient_folder: str | Path,
    config: Phase2InputConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    patient_folder = Path(patient_folder)
    files = os.listdir(patient_folder)

    modality_paths = {}
    for modality in config.modalities:
        filename = _find_token_file(files, modality)
        if filename is None:
            raise FileNotFoundError(f"Missing modality '{modality}' in {patient_folder}")
        modality_paths[modality] = patient_folder / filename

    seg_filename = _find_token_file(files, "seg")
    if seg_filename is None:
        raise FileNotFoundError(f"Missing segmentation mask in {patient_folder}")

    modality_volumes = [
        _normalize_volume(_load_nifti(modality_paths[modality]), config.clip_percentile)
        for modality in config.modalities
    ]
    shapes = {volume.shape for volume in modality_volumes}
    if len(shapes) != 1:
        raise ValueError(f"Modality shape mismatch in {patient_folder}: {sorted(shapes)}")

    seg_volume = _load_nifti(patient_folder / seg_filename)
    if seg_volume.shape != modality_volumes[0].shape:
        raise ValueError(
            f"Segmentation shape {seg_volume.shape} does not match image shape "
            f"{modality_volumes[0].shape} in {patient_folder}"
        )

    if config.binary_segmentation:
        seg_volume = (seg_volume > 0).astype(np.float32)

    volume = np.stack(modality_volumes, axis=-1)
    tumor_pixels = np.array([seg_volume[:, :, z].sum() for z in range(seg_volume.shape[2])])
    rng = np.random.default_rng(config.random_state + _stable_patient_seed(patient_folder.name))
    selected_indices, categories = _balanced_slice_indices(tumor_pixels, config, rng)

    images = []
    masks = []
    metadata = []
    category_by_index = {
        z: category for category, values in categories.items() for z in values
    }

    for z in selected_indices:
        image = volume[:, :, z, :]
        mask = seg_volume[:, :, z]
        if config.roi_crop:
            y_min, y_max, x_min, x_max = _brain_bbox_2d(image, config.roi_margin)
            image = image[y_min:y_max, x_min:x_max, :]
            mask = mask[y_min:y_max, x_min:x_max]
        else:
            y_min, y_max, x_min, x_max = 0, image.shape[0], 0, image.shape[1]

        image_resized, mask_resized = _resize_pair(image, mask, config.target_size)
        images.append(image_resized)
        masks.append(mask_resized)
        metadata.append(
            {
                "patient_id": patient_folder.name,
                "slice_index": int(z),
                "slice_class": category_by_index.get(z, "tumor"),
                "tumor_pixels_original": float(tumor_pixels[z]),
                "roi_bbox_ymin_ymax_xmin_xmax": [int(y_min), int(y_max), int(x_min), int(x_max)],
            }
        )

    return images, masks, metadata


def load_phase2_dataset(
    dataset_root: str | Path,
    config: Phase2InputConfig | None = None,
    max_patients: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    config = config or Phase2InputConfig()
    patient_folders = find_patient_folders(dataset_root, config.modalities)
    if max_patients is not None:
        patient_folders = patient_folders[:max_patients]
    if verbose:
        print(f"Found {len(patient_folders)} patients with modalities {config.modalities}")

    all_images: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_metadata: list[dict] = []
    for patient_folder in patient_folders:
        images, masks, metadata = process_patient_folder(patient_folder, config)
        all_images.extend(images)
        all_masks.extend(masks)
        all_metadata.extend(metadata)
        if verbose:
            counts = {}
            for item in metadata:
                counts[item["slice_class"]] = counts.get(item["slice_class"], 0) + 1
            print(f"{patient_folder.name}: {len(images)} slices {counts}")

    images_arr = np.array(all_images, dtype=np.float32)
    masks_arr = np.array(all_masks, dtype=np.float32)
    if verbose:
        print(f"Loaded X={images_arr.shape}, y={masks_arr.shape}")
        if len(masks_arr):
            print(f"Tumor pixel ratio: {masks_arr.mean():.6f}")
    return images_arr, masks_arr, all_metadata


def split_phase2_dataset(
    images: np.ndarray,
    masks: np.ndarray,
    metadata: list[dict],
    config: Phase2InputConfig | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], dict]:
    config = config or Phase2InputConfig()
    if len(images) != len(masks) or len(images) != len(metadata):
        raise ValueError("images, masks, and metadata must have equal length")
    if abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1")

    patient_ids = [item["patient_id"] for item in metadata]
    unique_patients = sorted(set(patient_ids))
    train_patients, temp_patients = train_test_split(
        unique_patients,
        test_size=(config.val_ratio + config.test_ratio),
        random_state=config.random_state,
    )
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=config.test_ratio / (config.val_ratio + config.test_ratio),
        random_state=config.random_state,
    )

    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_patients]

    split_info = {
        "version": config.version,
        "config": asdict(config),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_patients": sorted(train_patients),
        "val_patients": sorted(val_patients),
        "test_patients": sorted(test_patients),
        "patient_overlap": {
            "train_val": sorted(set(train_patients) & set(val_patients)),
            "train_test": sorted(set(train_patients) & set(test_patients)),
            "val_test": sorted(set(val_patients) & set(test_patients)),
        },
        "slice_class_counts": _slice_class_counts(metadata),
    }

    return (
        (images[train_indices], masks[train_indices]),
        (images[val_indices], masks[val_indices]),
        (images[test_indices], masks[test_indices]),
        split_info,
    )


def _slice_class_counts(metadata: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in metadata:
        label = item.get("slice_class", "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def save_phase2_splits(
    splits: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    metadata: list[dict],
    split_info: dict,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits
    np.save(output / "X_train.npy", X_train)
    np.save(output / "y_train.npy", y_train)
    np.save(output / "X_val.npy", X_val)
    np.save(output / "y_val.npy", y_val)
    np.save(output / "X_test.npy", X_test)
    np.save(output / "y_test.npy", y_test)

    with (output / "slice_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (output / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    summary = {
        "version": split_info.get("version", "v2_multimodal_roi"),
        "X_train": list(X_train.shape),
        "X_val": list(X_val.shape),
        "X_test": list(X_test.shape),
        "y_train": list(y_train.shape),
        "y_val": list(y_val.shape),
        "y_test": list(y_test.shape),
        "slice_class_counts_all": split_info.get("slice_class_counts", {}),
        "slice_class_counts_by_split": _slice_class_counts_by_split(metadata, split_info),
        "patient_counts": {
            "train": len(split_info.get("train_patients", [])),
            "val": len(split_info.get("val_patients", [])),
            "test": len(split_info.get("test_patients", [])),
        },
        "patient_overlap": split_info.get("patient_overlap", {}),
    }
    with (output / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    overlaps = summary["patient_overlap"]
    if any(overlaps.values()):
        raise RuntimeError(f"Patient overlap detected after saving splits: {overlaps}")


def _slice_class_counts_by_split(metadata: list[dict], split_info: dict) -> dict[str, dict[str, int]]:
    result = {}
    for split_name, key in (
        ("train", "train_indices"),
        ("val", "val_indices"),
        ("test", "test_indices"),
    ):
        rows = [metadata[i] for i in split_info.get(key, [])]
        result[split_name] = _slice_class_counts(rows)
    return result
