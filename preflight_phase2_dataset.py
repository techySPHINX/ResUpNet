"""
Preflight checks for Phase 2 BraTS training.

This script validates the dataset root before preprocessing/training:
- expected patient folders exist
- each patient has T1, T1ce, T2, FLAIR, and segmentation mask
- optional deep check loads one patient and confirms compatible volume shapes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2_input_pipeline import (
    Phase2InputConfig,
    find_patient_folders,
    process_patient_folder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BraTS dataset for Phase 2.")
    parser.add_argument("--dataset-root", required=True, help="Path to BraTS patient folders.")
    parser.add_argument("--deep-check", action="store_true", help="Load one patient and process slices.")
    parser.add_argument("--max-report", type=int, default=5, help="Number of patient folders to show.")
    parser.add_argument(
        "--output-json",
        default="experiments/v2_multimodal_roi/dataset_preflight.json",
        help="Path to write preflight summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    config = Phase2InputConfig()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    patient_folders = find_patient_folders(dataset_root, config.modalities)
    if not patient_folders:
        raise RuntimeError(
            "No valid BraTS patient folders found. Expected each patient folder to contain "
            "T1, T1ce, T2, FLAIR, and seg NIfTI files."
        )

    summary = {
        "dataset_root": str(dataset_root),
        "dataset_expected": "BraTS-style raw patient folders, e.g. BraTS2021_00000",
        "modalities_required": list(config.modalities),
        "segmentation_required": "seg",
        "valid_patient_count": len(patient_folders),
        "sample_patients": [folder.name for folder in patient_folders[: args.max_report]],
        "deep_check": None,
    }

    print(f"Dataset root: {dataset_root}")
    print(f"Valid patient folders: {len(patient_folders)}")
    print("Sample patients:")
    for folder in patient_folders[: args.max_report]:
        print(f"  - {folder.name}")

    if args.deep_check:
        first = patient_folders[0]
        images, masks, metadata = process_patient_folder(first, config)
        summary["deep_check"] = {
            "patient_id": first.name,
            "processed_slices": len(images),
            "image_shape": list(images[0].shape) if images else None,
            "mask_shape": list(masks[0].shape) if masks else None,
            "slice_class_counts": {},
        }
        for item in metadata:
            label = item["slice_class"]
            summary["deep_check"]["slice_class_counts"][label] = (
                summary["deep_check"]["slice_class_counts"].get(label, 0) + 1
            )
        print("Deep check:")
        print(json.dumps(summary["deep_check"], indent=2))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {output_json}")


if __name__ == "__main__":
    main()
