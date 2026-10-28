import logging
import os
import subprocess
from pathlib import Path

OPENMVS_BIN_DIR = os.getenv("HOME") + "/.local/bin/OpenMVS"

logger = logging.getLogger(__name__)


def create_dense_reconstruction(object_dir: Path, image_extension: str = "jpg") -> None:
    """Create a dense reconstruction using OpenMVS."""
    object_dir = object_dir.resolve()
    logger.info("Starting dense reconstruction with OpenMVS")
    object_dir = Path(object_dir).resolve()
    images_dir = object_dir / "images"
    sparse_dir = object_dir / "splat" / "colmap"
    dense_dir = object_dir / "dense"
    dense_dir.mkdir(exist_ok=True)

    colmap_undistorter_cmd = [
        "colmap",
        "image_undistorter",
        "--input_path",
        str(sparse_dir),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(dense_dir),
        "--output_type",
        "COLMAP",
    ]
    subprocess.run(colmap_undistorter_cmd, check=True)

    interface_colmap_cmd = [
        f"{OPENMVS_BIN_DIR}/InterfaceCOLMAP",
        "-w",
        str(object_dir),
        "-i",
        "dense",
        "-o",
        "scene.mvs",
        "--image-folder",
        "images",
    ]
    subprocess.run(interface_colmap_cmd, check=True)

    masks_src = object_dir / "masks"
    masks_dst = dense_dir / "masks"
    masks_dst.mkdir(exist_ok=True)
    for mask_file in masks_src.iterdir():
        if mask_file.is_file():
            new_name = mask_file.stem.replace(f".{image_extension}", "") + ".mask.png"
            subprocess.run(
                ["cp", str(mask_file), str(masks_dst / new_name)], check=True
            )

    densify_pcl_command = [
        f"{OPENMVS_BIN_DIR}/DensifyPointCloud",
        "scene.mvs",
        "-w",
        str(object_dir),
        "--mask-path",
        str(object_dir / "dense" / "masks"),
        "--ignore-mask-label",
        "0",
        "--filter-point-cloud",
        "1",
        "--remove-dmaps",
        "1",
        "--tower-mode",
        "0",
    ]
    subprocess.run(densify_pcl_command, check=True)

    reconstruct_mesh_command = [
        f"{OPENMVS_BIN_DIR}/ReconstructMesh",
        "scene_dense.mvs",
        "-w",
        str(object_dir),
        "-p",
        "scene_dense.ply",
        "-o",
        "scene_mesh.ply",
    ]
    subprocess.run(reconstruct_mesh_command, check=True)

    logger.info("Dense reconstruction completed")
