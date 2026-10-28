"""Novasplat runner for 3D scene reconstruction."""

import argparse
import logging
import subprocess
import time
from pathlib import Path

from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy

from novasplat.colmap import (
    create_colmap_reconstruction,
    create_colmap_reconstruction_from_two_videos,
    create_sparse_reconstruction,
    write_colmap_reconstruction,
)
from novasplat.features.dense_features import extract_and_fuse_features
from novasplat.io import copy_images_and_masks, load_onboarding_data
from novasplat.masking import apply_masks_to_images
from novasplat.openvms import create_dense_reconstruction
from novasplat.templates import render_object_templates
from novasplat.utils.pcl import poisson_disk_subsample
from novasplat.utils.splat_to_colmap import (
    export_colmap_with_splat_visibility_from_disk,
)
from thirdparty.gsplat.simple_trainer import Config
from thirdparty.gsplat.simple_trainer import main as splat_main

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments for the Novasplat runner.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Run Novasplat")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to the input directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--object",
        help="Object ID",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--icosphere_level",
        help="ICOSPHERE level",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "--scale",
        help="Scaling factor for template rendering",
        required=False,
        type=float,
        default=None,
    )
    return parser.parse_args()


def run_splat_training(data_dir: Path, result_dir: Path):
    """Run the splat training for a given data and result directory.

    Args:
        data_dir: The path to the data directory.
        result_dir: The path to the result directory.
    """
    logger.info("Running splat training")
    cfg = Config(
        strategy=DefaultStrategy(
            verbose=False,
            revised_opacity=True,
            refine_start_iter=0,
            refine_stop_iter=7000,
        ),
        data_dir=str(data_dir),
        result_dir=str(result_dir),
        save_ply=True,
        data_factor=1,
        test_every=2500,
        max_steps=5000,
        normalize_world_space=False,
        use_fused_bilagrid=True,
        depth_loss=True,
    )
    cli(splat_main, cfg, verbose=False)


def run(
    input_dir: str,
    output_dir: str,
    object_id: str,
    icosphere_level: int,
    scale: float,
):
    """Run the Novasplat reconstruction pipeline."""
    logger.info("Running Novasplat reconstruction pipeline for object %s", object_id)
    _t_total = time.perf_counter()

    _t0 = time.perf_counter()
    onboarding_data = load_onboarding_data(Path(input_dir), object_id)
    _t_data_load = time.perf_counter() - _t0

    should_use_two_videos = "down" in onboarding_data and "up" in onboarding_data

    _t0 = time.perf_counter()
    if should_use_two_videos:
        reconstruction = create_colmap_reconstruction_from_two_videos(onboarding_data)
    else:
        reconstruction = create_colmap_reconstruction(onboarding_data)
    _t_colmap = time.perf_counter() - _t0

    onboarding_dir = Path(output_dir) / "onboarding"
    onboarding_dir.mkdir(parents=True, exist_ok=True)

    _t0 = time.perf_counter()
    write_colmap_reconstruction(
        reconstruction,
        onboarding_dir,
        object_id,
    )
    if should_use_two_videos:
        if (Path(input_dir) / f"{object_id}_down").exists():
            copy_images_and_masks(
                Path(input_dir),
                onboarding_dir,
                object_id,
                onboarding_data["down"]["scene_camera"],
                "down",
            )
        if (Path(input_dir) / f"{object_id}_up").exists():
            copy_images_and_masks(
                Path(input_dir),
                onboarding_dir,
                object_id,
                onboarding_data["up"]["scene_camera"],
                "up",
            )
    else:
        copy_images_and_masks(
            Path(input_dir),
            onboarding_dir,
            object_id,
            onboarding_data["scene_camera"],
            image_extension="png",
        )
    _t_write_copy = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    create_sparse_reconstruction(
        onboarding_dir / object_id,
    )
    _t_sparse = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    apply_masks_to_images(
        onboarding_dir / object_id / "images",
        onboarding_dir / object_id / "masks",
        onboarding_dir / object_id / "images",
        image_suffix=".jpg" if should_use_two_videos else ".png",
    )
    _t_masks = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    run_splat_training(
        onboarding_dir / object_id,
        onboarding_dir / object_id / "splat",
    )
    _t_splat = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    export_colmap_with_splat_visibility_from_disk(
        onboarding_dir / object_id / "splat" / "ply" / "point_cloud_4999.ply",
        onboarding_dir / object_id / "sparse" / "1",
        onboarding_dir / object_id / "splat" / "colmap",
    )
    _t_export_colmap = time.perf_counter() - _t0

    templates_dir = Path(output_dir) / "templates" / object_id
    templates_dir.mkdir(parents=True, exist_ok=True)
    try:
        _t0 = time.perf_counter()
        create_dense_reconstruction(
            onboarding_dir / object_id,
            "jpg" if should_use_two_videos else "png",
        )
        _t_dense = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        poisson_disk_subsample(
            str(onboarding_dir / object_id / "scene_mesh.ply"),
            str(onboarding_dir / object_id / "sparse" / "1" / "points3D.ply"),
            n_points=100000,
        )
        poisson_disk_subsample(
            str(onboarding_dir / object_id / "scene_mesh.ply"),
            str(templates_dir / "model.ply"),
        )
        _t_subsample = time.perf_counter() - _t0
    except (RuntimeError, subprocess.CalledProcessError) as e:
        # Template depth maps and GeDi features need the mesh, so stop here.
        logger.error("Dense reconstruction failed for object %s: %s", object_id, e)
        raise

    _t0 = time.perf_counter()
    render_object_templates(
        onboarding_dir / object_id / "splat" / "ckpts" / "ckpt_4999_rank0.pt",
        onboarding_dir / object_id / "scene_mesh.ply",
        templates_dir,
        icosphere_level=icosphere_level,
        scale=scale,
        size=480,
    )
    _t_templates = time.perf_counter() - _t0
    logger.info("Object templates rendered")

    logger.info("Starting template feature extraction (DINOv3 + GeDi + fusion)...")
    try:
        _t0 = time.perf_counter()
        extract_and_fuse_features(templates_dir, templates_dir / "features")
        _t_features = time.perf_counter() - _t0
        logger.info("Template feature extraction completed.")
    except Exception as e:
        logger.exception("Feature extraction pipeline failed: %s", e)
        return

    logger.info(
        (
            "Onboarding timing summary for %s:\n"
            "  Data loading:          %6.1f s\n"
            "  COLMAP reconstruction: %6.1f s\n"
            "  Write + copy images:   %6.1f s\n"
            "  Sparse reconstruction: %6.1f s\n"
            "  Apply masks:           %6.1f s\n"
            "  Splat training:        %6.1f s\n"
            "  Export COLMAP:         %6.1f s\n"
            "  Dense reconstruction:  %6.1f s\n"
            "  Poisson subsampling:   %6.1f s\n"
            "  Template rendering:    %6.1f s\n"
            "  Feature extraction:    %6.1f s\n"
            "  TOTAL:                 %6.1f s"
        ),
        object_id,
        _t_data_load,
        _t_colmap,
        _t_write_copy,
        _t_sparse,
        _t_masks,
        _t_splat,
        _t_export_colmap,
        _t_dense,
        _t_subsample,
        _t_templates,
        _t_features,
        time.perf_counter() - _t_total,
    )


def main():
    """Main entry point for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    args = parse_args()
    run(args.input, args.output, args.object, args.icosphere_level, args.scale)


if __name__ == "__main__":
    main()
