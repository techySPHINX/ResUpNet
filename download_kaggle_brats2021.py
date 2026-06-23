"""
Download the target Kaggle BraTS 2021 Task 1 dataset.

Dataset slug:
    dschettler8845/brats-2021-task1

Kaggle requires authentication. Put kaggle.json at:
    data/.kaggle/kaggle.json

or set:
    KAGGLE_USERNAME
    KAGGLE_KEY
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from resupnet_runtime_paths import configure_runtime_paths

configure_runtime_paths()


DATASET_SLUG = "dschettler8845/brats-2021-task1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kaggle BraTS 2021 Task 1 dataset.")
    parser.add_argument("--output-dir", default="data/kaggle_brats2021_task1")
    parser.add_argument("--kaggle-config-dir", default="data/.kaggle")
    parser.add_argument("--no-unzip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config_dir = Path(args.kaggle_config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    kaggle_json = config_dir / "kaggle.json"
    has_env_auth = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not kaggle_json.exists() and not has_env_auth:
        raise RuntimeError(
            "Kaggle credentials missing. Download kaggle.json from Kaggle account settings "
            f"and place it at {kaggle_json}, or set KAGGLE_USERNAME and KAGGLE_KEY."
        )

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(config_dir.resolve())

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(output_dir),
    ]
    if not args.no_unzip:
        command.append("--unzip")

    print("Running:", " ".join(command))
    subprocess.check_call(command, env=env)
    print(f"Dataset downloaded to: {output_dir}")
    print("Next:")
    print(f"python preflight_phase2_dataset.py --dataset-root {output_dir} --deep-check")


if __name__ == "__main__":
    main()
