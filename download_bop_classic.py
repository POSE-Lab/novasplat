import argparse
import logging
import subprocess

from pathlib import Path

from huggingface_hub import hf_hub_download  # type: ignore

logger = logging.getLogger(__name__)

BOP_CLASSIC_DATASETS = [
    "lm",
]


def parse_args():
    """
    Parse command line arguments for the BOP-Classic downloader.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Download BOP-Classic datasets")
    parser.add_argument(
        "-d",
        "--data_dir",
        help="Path to the data directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=BOP_CLASSIC_DATASETS,
        help="Datasets to download",
    )
    return parser.parse_args()


def download_dataset(
    dataset_name: str,
    data_dir: Path,
) -> None:
    """Download a BOP-Classic dataset.

    Args:
        dataset_name: The name of the dataset to download.
        data_dir: The path to the data directory.
    """
    logger.info("Downloading dataset %s", dataset_name)
    dataset_dir = data_dir / dataset_name

    filepath = hf_hub_download(
        repo_id=f"bop-benchmark/{dataset_name}",
        filename=f"{dataset_name}_base.zip",
        repo_type="dataset",
        local_dir=f"{dataset_dir}",
    )
    logger.info("Downloaded file %s", filepath)
    zip_path = Path(filepath)
    try:
        subprocess.run(
            ["unzip", "-o", str(zip_path), "-d", str(data_dir)],
            check=True,
        )
        try:
            zip_path.unlink()
            logger.info("Removed archive %s", zip_path)
        except OSError as e:
            logger.warning("Could not remove zip %s: %s", zip_path, e)
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to extract %s: %s", zip_path, e)

    required_files = ["train", "test_bop19"]
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
                [
                    "unzip",
                    "-o",
                    str(zip_path),
                    "-d",
                    str(dataset_dir),
                ],
                check=True,
            )
            try:
                zip_path.unlink()
                logger.info("Removed archive %s", zip_path)
            except OSError as e:
                logger.warning("Could not remove zip %s: %s", zip_path, e)
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to extract %s: %s", zip_path, e)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    root_dir = Path(args.data_dir) / "bop_classic"
    root_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        download_dataset(dataset, root_dir)


if __name__ == "__main__":
    main()
