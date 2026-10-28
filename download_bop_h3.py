"""
BOP-H3 Dataset Downloader and Utilities

This script provides command-line tools for downloading, extracting, and
preparing BOP-H3 datasets from the Hugging Face Hub.

Features:
- Download BOP-H3 datasets (handal, hope, hot3d) and their onboarding types
  (static, dynamic) using the Hugging Face Hub.
- Extract and organize dataset files and folders, including onboarding and
  validation data.
- Post-process datasets by renaming folders for consistency.
- Generate validation target files and COCO-format ground truth annotations
  for the validation split.

Usage:
    python download_bop_h3.py -d <data_dir> [--onboarding_types static dynamic] [--datasets handal hope hot3d]

Dependencies:
    - argparse
    - tqdm
    - huggingface_hub
    - subprocess
    - pathlib
    - logging
    - json

Intended for use in preparing BOP-H3 datasets for benchmarking and evaluation
workflows.
"""

import argparse
import json
import logging
import subprocess
from tqdm import tqdm

from pathlib import Path

from huggingface_hub import hf_hub_download  # type: ignore

logger = logging.getLogger(__name__)

BOP_H3_DATASETS = [
    "handal",
    "hope",
    "hot3d",
]

BOP_H3_ONBOARDING_TYPES = [
    "static",
    "dynamic",
]


def parse_args():
    """
    Parse command line arguments for the BOP-H3 downloader.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Download BOP-H3 datasets")
    parser.add_argument(
        "-d",
        "--data_dir",
        help="Path to the data directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-t",
        "--onboarding_types",
        nargs="+",
        default=BOP_H3_ONBOARDING_TYPES,
        help="Types of onboarding data to download",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=BOP_H3_DATASETS,
        help="Datasets to download",
    )
    return parser.parse_args()


def download_dataset(
    dataset_name: str,
    data_dir: Path,
    onboarding_types: list,
) -> None:
    """Download a BOP-H3 dataset.

    Args:
        dataset_name: The name of the dataset to download.
        data_dir: The path to the data directory.
    """
    required_files = ["models"]
    required_folders = []
    if dataset_name in ["handal", "hope"]:
        required_files.extend([f"onboarding_{type}" for type in onboarding_types])
        required_files.append("test_bop24")
        if dataset_name == "hope":
            required_files.append("val_realsense")
        else:
            required_files.append("val")
    else:
        required_folders.extend(
            [f"object_ref_aria_{type}" for type in onboarding_types]
        )

    logger.info("Downloading dataset %s", dataset_name)
    dataset_dirname = "hopev2" if dataset_name == "hope" else dataset_name
    dataset_dir = data_dir / dataset_dirname
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for filename in required_files:
        filepath = hf_hub_download(
            repo_id=f"bop-benchmark/{dataset_name}",
            filename=f"{dataset_name}_{filename}.zip",
            repo_type="dataset",
            local_dir=f"{dataset_dir}",
        )
        logger.info("Downloaded file %s", filepath)

        zip_path = Path(filepath)
        try:
            subprocess.run(
                ["unzip", "-o", str(zip_path), "-d", str(dataset_dir)], check=True
            )
            try:
                zip_path.unlink()
                logger.info("Removed archive %s", zip_path)
            except Exception as e:
                logger.warning("Could not remove zip %s: %s", zip_path, e)
        except Exception as e:
            logger.warning("Failed to extract %s: %s", zip_path, e)

    for folder in required_folders:
        local_folder_path = dataset_dir / folder
        download_cmd = f"hf download bop-benchmark/{dataset_name} --include {folder}/*.tar --local-dir {dataset_dir} --repo-type=dataset"
        try:
            subprocess.run(download_cmd, shell=True, check=True)
            logger.info("Downloaded %s", folder)
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to run download command for %s: %s", folder, e)

        files = Path(local_folder_path).glob("*.tar")
        files = sorted(files, key=lambda x: x.name)
        for file in tqdm(files, desc=f"Extracting {folder}"):
            sub_folder = local_folder_path / file.stem
            sub_folder.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(
                    ["tar", "-xf", str(file), "-C", str(sub_folder)], check=True
                )
                try:
                    file.unlink()
                    logger.info("Removed archive %s", file)
                except Exception as e:
                    logger.warning("Could not remove tar %s: %s", file, e)
            except Exception as e:
                logger.warning("Failed to extract %s: %s", file, e)

    if dataset_name == "handal":
        for onboarding_type in onboarding_types:
            (dataset_dir / onboarding_type).rename(
                dataset_dir / f"onboarding_{onboarding_type}"
            )
    if dataset_name == "hot3d":
        for onboarding_type in onboarding_types:
            (dataset_dir / f"object_ref_aria_{onboarding_type}").rename(
                dataset_dir / f"onboarding_{onboarding_type}"
            )


def create_val_targets(data_dir: Path, dataset_name: str) -> None:
    """Create validation target files for a BOP-H3 dataset.

    Args:
        data_dir: The path to the data directory.
        dataset_name: The name of the dataset.
    """
    dataset_dirname = "hopev2" if dataset_name == "hope" else dataset_name
    dataset_dir = data_dir / dataset_dirname
    val_dir = dataset_dir / "val"

    targets_file = dataset_dir / "val_targets_bop24.json"
    if targets_file.exists():
        logger.info("Validation targets file already exists for %s", dataset_name)
        return

    logger.info("Creating validation targets file for %s", dataset_name)
    val_targets = []
    scene_folders = sorted(val_dir.iterdir())
    for scene_folder in scene_folders:
        if not scene_folder.is_dir():
            continue
        scene_id = int(scene_folder.name)
        rgb_folder = scene_folder / "rgb"
        image_files = sorted(rgb_folder.iterdir())
        for image_file in image_files:
            if image_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            im_id = int(image_file.stem)
            val_targets.append(
                {
                    "im_id": im_id,
                    "scene_id": scene_id,
                }
            )
    with open(targets_file, "w") as f:
        json.dump(val_targets, f, indent=2)


def calc_gt_coco_for_val_split(data_dir: Path, dataset_name: str) -> None:
    """Generate COCO-format ground truth annotations for the validation split.

    This function calls the BOP Toolkit `calc_gt_coco.py` script to create
    `scene_gt_coco.json` files for the validation split of a given dataset,
    using the previously generated `val_targets_bop24.json` targets file.

    Args:
        data_dir: Root directory where BOP-H3 datasets are stored.
        dataset_name: Name of the dataset for which to generate validation
            COCO ground truth annotations.
    """
    dataset_dirname = "hopev2" if dataset_name == "hope" else dataset_name
    logger.info("Creating scene_gt_coco.json files for %s", dataset_name)
    cmd = "python thirdparty/bop_toolkit/scripts/calc_gt_coco.py" + (
        f" --dataset={dataset_dirname} --dataset_split=val "
        f"--targets_filename=val_targets_bop24.json "
        f"--datasets_path={data_dir}"
    )
    logger.info("Executing %s", cmd)
    subprocess.run(cmd, check=True)
    logger.info("Created scene_gt_coco.json files for %s", dataset_name)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    root_dir = Path(args.data_dir) / "bop_h3"
    root_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        download_dataset(dataset, root_dir, args.onboarding_types)
        create_val_targets(root_dir, dataset)
        calc_gt_coco_for_val_split(root_dir, dataset)


if __name__ == "__main__":
    main()
