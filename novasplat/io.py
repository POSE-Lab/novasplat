"""IO utilities for loading and processing BOP dataset scenes."""

import logging
import shutil

from pathlib import Path

from bop_toolkit_lib.dataset import bop_scenewise
from bop_toolkit_lib.inout import load_cam_params

logger = logging.getLogger(__name__)


def load_onboarding_data(input_dir: Path, object_id: str) -> dict:
    """Load onboarding data for both up and down directions."""
    logger.info("Loading onboarding data for %s", object_id)
    scene_data = dict()
    input_path_down = input_dir / f"{object_id}_down"
    if input_path_down.exists():
        logger.info("Loading down scene data from %s", input_path_down)
        scene_data["down"] = bop_scenewise.load_scene_data(
            input_path_down,
            load_scene_gt_info=False,
        )
    input_path_up = input_dir / f"{object_id}_up"
    if input_path_up.exists():
        logger.info("Loading up scene data from %s", input_path_up)
        scene_data["up"] = bop_scenewise.load_scene_data(
            input_path_up,
            load_scene_gt_info=False,
        )
    input_path = input_dir / f"{object_id}"
    if input_path.exists():
        logger.info("Loading scene data from %s", input_path)
        scene_data = bop_scenewise.load_scene_data(
            input_path,
            load_scene_gt_info=False,
        )
        scene_data["cam_params"] = load_cam_params(input_dir.parent / "camera.json")
    logger.info("Loaded %d scene sequences", len(scene_data))
    return scene_data


def copy_images_and_masks(
    input_dir: Path,
    output_dir: Path,
    object_id: str,
    scene_cameras: dict,
    direction: str = "",
    image_extension: str = "jpg",
) -> None:
    """Copy images and masks for the specified direction and camera sequence."""
    logger.info("Copying images and masks for %s", direction)
    input_path = (
        input_dir / f"{object_id}_{direction}" if direction else input_dir / object_id
    )
    if not input_path.exists():
        logger.warning("Input directory not found: %s", input_path)
        return
    images_path = input_path / "rgb"
    masks_path = input_path / "mask_visib"

    output_path = output_dir / object_id
    output_path.mkdir(parents=True, exist_ok=True)
    output_images_path = output_path / "images"
    output_images_path.mkdir(parents=True, exist_ok=True)
    output_masks_path = output_path / "masks"
    output_masks_path.mkdir(parents=True, exist_ok=True)

    for frame_id in scene_cameras.keys():
        image_stem = f"{int(frame_id):06d}"

        # Source paths
        image_src_path = images_path / f"{image_stem}.{image_extension}"
        mask_src_path = masks_path / f"{image_stem}_000000.png"

        output_image_path = (
            output_images_path / f"{image_stem}_{direction}.{image_extension}"
            if direction
            else output_images_path / f"{image_stem}.{image_extension}"
        )
        mask_dst_path = (
            output_masks_path / f"{image_stem}_{direction}.{image_extension}.png"
            if direction
            else output_masks_path / f"{image_stem}.{image_extension}.png"
        )

        if image_src_path.exists():
            shutil.copy(image_src_path, output_image_path)
        else:
            logger.warning("Image not found: %s", image_src_path)
        if mask_src_path.exists():
            shutil.copy(mask_src_path, mask_dst_path)
        else:
            logger.warning("Mask not found: %s", mask_src_path)
