"""
Prepare versioned v2 multimodal ROI BraTS splits.

Example:
python prepare_phase2_dataset.py --dataset-root C:/Datasets/BraTS2021_Training_Data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from resupnet_runtime_paths import configure_runtime_paths

configure_runtime_paths()

from phase2_input_pipeline import (
    Phase2InputConfig,
    find_patient_folders,
    process_patient_folder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 2 multimodal ROI BraTS data.")
    parser.add_argument("--dataset-root", required=True, help="Path to raw BraTS patient folders.")
    parser.add_argument(
        "--output-dir",
        default="experiments/v2_multimodal_roi/processed_splits",
        help="Versioned output directory.",
    )
    parser.add_argument("--max-patients", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--target-size", type=int, default=256, help="Output H/W size. Prefer 256.")
    parser.add_argument("--roi-margin", type=int, default=8, help="Brain ROI margin in pixels.")
    parser.add_argument("--no-roi", action="store_true", help="Disable ROI crop ablation.")
    parser.add_argument("--min-tumor-pixels", type=int, default=1, help="Positive slice threshold.")
    parser.add_argument("--near-slices", type=int, default=2, help="Distance for near-tumor empty slices.")
    parser.add_argument("--max-tumor-slices-per-patient", type=int, default=48)
    parser.add_argument("--max-near-slices-per-patient", type=int, default=16)
    parser.add_argument("--max-hard-negative-slices-per-patient", type=int, default=16)
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Use float16 to keep full BraTS Phase 2 splits trainable on normal workstations.",
    )
    parser.add_argument(
        "--patients-per-chunk",
        type=int,
        default=12,
        help="Temporary chunk size used during streaming preprocessing.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _slice_class_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = row.get("slice_class", "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def _save_chunk(
    split_name: str,
    chunk_index: int,
    images: list[np.ndarray],
    masks: list[np.ndarray],
    metadata: list[dict],
    chunks_dir: Path,
) -> dict:
    x = np.asarray(images, dtype=np.float32)
    y = np.asarray(masks, dtype=np.float32)
    chunk_prefix = chunks_dir / f"{split_name}_{chunk_index:04d}"
    np.save(f"{chunk_prefix}_X.npy", x)
    np.save(f"{chunk_prefix}_y.npy", y)
    with Path(f"{chunk_prefix}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return {
        "x_path": f"{chunk_prefix}_X.npy",
        "y_path": f"{chunk_prefix}_y.npy",
        "metadata_path": f"{chunk_prefix}_metadata.json",
        "shape": list(x.shape),
        "mask_shape": list(y.shape),
        "count": int(len(x)),
    }


def _process_split_to_chunks(
    split_name: str,
    patient_folders: list[Path],
    config: Phase2InputConfig,
    chunks_dir: Path,
    patients_per_chunk: int,
) -> tuple[list[dict], list[dict]]:
    chunks = []
    split_metadata = []
    chunk_images: list[np.ndarray] = []
    chunk_masks: list[np.ndarray] = []
    chunk_metadata: list[dict] = []
    chunk_index = 0

    for patient_number, patient_folder in enumerate(patient_folders, start=1):
        images, masks, metadata = process_patient_folder(patient_folder, config)
        chunk_images.extend(images)
        chunk_masks.extend(masks)
        chunk_metadata.extend(metadata)
        split_metadata.extend(metadata)
        counts = _slice_class_counts(metadata)
        print(
            f"{split_name} {patient_number:04d}/{len(patient_folders):04d} "
            f"{patient_folder.name}: {len(images)} slices {counts}",
            flush=True,
        )
        if patient_number % patients_per_chunk == 0 and chunk_images:
            chunks.append(
                _save_chunk(
                    split_name,
                    chunk_index,
                    chunk_images,
                    chunk_masks,
                    chunk_metadata,
                    chunks_dir,
                )
            )
            chunk_images, chunk_masks, chunk_metadata = [], [], []
            chunk_index += 1

    if chunk_images:
        chunks.append(
            _save_chunk(
                split_name,
                chunk_index,
                chunk_images,
                chunk_masks,
                chunk_metadata,
                chunks_dir,
            )
        )

    return chunks, split_metadata


def _combine_chunks(
    split_name: str,
    chunks: list[dict],
    output_dir: Path,
    storage_dtype: str,
) -> tuple[list[int], list[int]]:
    if not chunks:
        raise ValueError(f"No chunks were produced for split '{split_name}'")

    first_x = np.load(chunks[0]["x_path"], mmap_mode="r")
    first_y = np.load(chunks[0]["y_path"], mmap_mode="r")
    total = sum(int(chunk["count"]) for chunk in chunks)
    x_shape = (total, *first_x.shape[1:])
    y_shape = (total, *first_y.shape[1:])
    x_out = np.lib.format.open_memmap(
        output_dir / f"X_{split_name}.npy",
        mode="w+",
        dtype=np.dtype(storage_dtype),
        shape=x_shape,
    )
    y_out = np.lib.format.open_memmap(
        output_dir / f"y_{split_name}.npy",
        mode="w+",
        dtype=np.uint8,
        shape=y_shape,
    )

    offset = 0
    for chunk in chunks:
        x_chunk = np.load(chunk["x_path"], mmap_mode="r")
        y_chunk = np.load(chunk["y_path"], mmap_mode="r")
        next_offset = offset + len(x_chunk)
        x_out[offset:next_offset] = x_chunk.astype(storage_dtype, copy=False)
        y_out[offset:next_offset] = (y_chunk > 0.5).astype(np.uint8, copy=False)
        offset = next_offset

    x_out.flush()
    y_out.flush()
    return list(x_shape), list(y_shape)


def _process_split_to_memmaps(
    split_name: str,
    patient_folders: list[Path],
    config: Phase2InputConfig,
    output_dir: Path,
    storage_dtype: str,
) -> tuple[list[int], list[int], list[dict]]:
    split_metadata: list[dict] = []
    total = 0
    image_shape = None
    mask_shape = None

    print(f"{split_name}: counting selected slices", flush=True)
    for patient_number, patient_folder in enumerate(patient_folders, start=1):
        images, masks, metadata = process_patient_folder(patient_folder, config)
        if images:
            image_shape = images[0].shape
            mask_shape = masks[0].shape
        total += len(images)
        split_metadata.extend(metadata)
        counts = _slice_class_counts(metadata)
        print(
            f"{split_name} count {patient_number:04d}/{len(patient_folders):04d} "
            f"{patient_folder.name}: {len(images)} slices {counts}",
            flush=True,
        )

    if total == 0 or image_shape is None or mask_shape is None:
        raise RuntimeError(f"No slices selected for split '{split_name}'")

    x_shape = (total, *image_shape)
    y_shape = (total, *mask_shape)
    x_out = np.lib.format.open_memmap(
        output_dir / f"X_{split_name}.npy",
        mode="w+",
        dtype=np.dtype(storage_dtype),
        shape=x_shape,
    )
    y_out = np.lib.format.open_memmap(
        output_dir / f"y_{split_name}.npy",
        mode="w+",
        dtype=np.uint8,
        shape=y_shape,
    )

    print(f"{split_name}: writing {total} slices to final memmaps", flush=True)
    offset = 0
    for patient_number, patient_folder in enumerate(patient_folders, start=1):
        images, masks, _ = process_patient_folder(patient_folder, config)
        if not images:
            continue
        x = np.asarray(images, dtype=np.float32)
        y = np.asarray(masks, dtype=np.float32)
        next_offset = offset + len(x)
        x_out[offset:next_offset] = x.astype(storage_dtype, copy=False)
        y_out[offset:next_offset] = (y > 0.5).astype(np.uint8, copy=False)
        offset = next_offset
        print(
            f"{split_name} write {patient_number:04d}/{len(patient_folders):04d} "
            f"{patient_folder.name}: offset={offset}",
            flush=True,
        )

    x_out.flush()
    y_out.flush()
    return list(x_shape), list(y_shape), split_metadata


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    config = Phase2InputConfig(
        target_size=(args.target_size, args.target_size),
        roi_crop=not args.no_roi,
        roi_margin=args.roi_margin,
        min_tumor_pixels=args.min_tumor_pixels,
        near_tumor_slices=args.near_slices,
        max_tumor_slices_per_patient=args.max_tumor_slices_per_patient,
        max_near_slices_per_patient=args.max_near_slices_per_patient,
        max_hard_negative_slices_per_patient=args.max_hard_negative_slices_per_patient,
        random_state=args.random_state,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_dir / "_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    patient_folders = find_patient_folders(args.dataset_root, config.modalities)
    if args.max_patients is not None:
        patient_folders = patient_folders[: args.max_patients]
    if not patient_folders:
        raise RuntimeError(f"No BraTS patient folders found under {args.dataset_root}")
    if abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1")

    train_folders, temp_folders = train_test_split(
        patient_folders,
        test_size=(config.val_ratio + config.test_ratio),
        random_state=config.random_state,
    )
    val_folders, test_folders = train_test_split(
        temp_folders,
        test_size=config.test_ratio / (config.val_ratio + config.test_ratio),
        random_state=config.random_state,
    )

    split_folders = {
        "train": sorted(train_folders),
        "val": sorted(val_folders),
        "test": sorted(test_folders),
    }
    split_patients = {
        split: [folder.name for folder in folders]
        for split, folders in split_folders.items()
    }
    patient_overlap = {
        "train_val": sorted(set(split_patients["train"]) & set(split_patients["val"])),
        "train_test": sorted(set(split_patients["train"]) & set(split_patients["test"])),
        "val_test": sorted(set(split_patients["val"]) & set(split_patients["test"])),
    }
    if any(patient_overlap.values()):
        raise RuntimeError(f"Patient overlap detected before preprocessing: {patient_overlap}")

    print(
        "Patient-wise split: "
        f"train={len(split_folders['train'])}, "
        f"val={len(split_folders['val'])}, "
        f"test={len(split_folders['test'])}",
        flush=True,
    )

    all_metadata = []
    split_shapes = {}
    split_class_counts = {}
    for split_name, folders in split_folders.items():
        x_shape, y_shape, metadata = _process_split_to_memmaps(
            split_name,
            folders,
            config,
            output_dir,
            args.storage_dtype,
        )
        split_shapes[f"X_{split_name}"] = x_shape
        split_shapes[f"y_{split_name}"] = y_shape
        split_class_counts[split_name] = _slice_class_counts(metadata)
        all_metadata.extend({**row, "split": split_name} for row in metadata)

    split_metadata = {
        "version": config.version,
        "config": config.__dict__,
        "dataset_root": str(args.dataset_root),
        "storage_dtype": args.storage_dtype,
        "train_patients": split_patients["train"],
        "val_patients": split_patients["val"],
        "test_patients": split_patients["test"],
        "patient_overlap": patient_overlap,
    }
    summary = {
        "version": config.version,
        "dataset_root": str(args.dataset_root),
        "storage_dtype": args.storage_dtype,
        **split_shapes,
        "patient_counts": {
            "train": len(split_patients["train"]),
            "val": len(split_patients["val"]),
            "test": len(split_patients["test"]),
        },
        "slice_class_counts_by_split": split_class_counts,
        "slice_class_counts_all": _slice_class_counts(all_metadata),
        "patient_overlap": patient_overlap,
    }

    _write_json(output_dir / "slice_metadata.json", all_metadata)
    _write_json(output_dir / "split_metadata.json", split_metadata)
    _write_json(output_dir / "dataset_summary.json", summary)
    _write_json(output_dir / "prepare_config.json", {"args": vars(args), "config": config.__dict__})

    print(f"Saved Phase 2 data to: {output_dir}")


if __name__ == "__main__":
    main()
