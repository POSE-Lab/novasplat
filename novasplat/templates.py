"""Render object templates."""

from pathlib import Path
from typing import List

from icosphere import icosphere
import numpy as np
import torch

from thirdparty.gsplat.render_templates import load_checkpoint
from thirdparty.gsplat.render_templates import render_templates
from novasplat.utils.depth_renderer import render_depth_maps


def normalize(vec):
    """
    Normalize a vector.
    """
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return vec / norm


def look_at(cam_location: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Create a world-to-camera look-at matrix.

    Args:
        cam_location: The camera location.
        point: The point to look at.

    Returns:
        The world-to-camera transformation matrix.
    """
    direction = point - cam_location
    if np.linalg.norm(direction) < 1e-6:
        cam_location = cam_location + np.array([0.1, 0.1, 0.1])
        direction = point - cam_location

    forward = normalize(direction)
    world_up = np.array([0, 0, 1])

    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([1, 0, 0])

    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))

    mat = np.eye(4)
    mat[0, :3] = -right
    mat[1, :3] = up
    mat[2, :3] = forward
    mat[3, :3] = 0

    mat[:3, 3] = -mat[:3, :3] @ cam_location

    det = np.linalg.det(mat[:3, :3])
    if abs(det) < 1e-6:
        mat[:3, :3] = np.eye(3)
        mat[:3, 3] = -cam_location

    return mat


def convert_location_to_rotation(locations: List[np.ndarray]) -> np.ndarray:
    """
    Convert locations to rotation matrices.

    Args:
        locations: The locations to convert.

    Returns:
        The rotation matrices.
    """
    obj_poses = np.zeros((len(locations), 4, 4))
    for idx, pt in enumerate(locations):
        obj_poses[idx] = look_at(pt, np.array([0, 0, 0]))
    return obj_poses


def calculate_optimal_scale(
    intrinsics: np.ndarray,
    width: int,
    height: int,
    scene_radius: float = 1.0,
):
    """
    Calculate optimal scale to fit the scene within 50% of the image bounds.

    Args:
        intrinsics: 3x3 camera intrinsic matrix
        width, height: Image dimensions
        scene_radius: Radius of the scene (default 1.0 for unit sphere)

    Returns:
        float: Optimal scale factor
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    fov_x = 2 * np.arctan(width / (2 * fx))
    fov_y = 2 * np.arctan(height / (2 * fy))
    fov = min(fov_x, fov_y)
    desired_fraction = np.sqrt(0.5)  # 50% of the image area
    distance = scene_radius / np.tan(fov / 2) / desired_fraction
    distance = max(distance, 1.0)  # minimum scale is 1.0
    return distance


def estimate_scene_radius(splats: dict[str, torch.Tensor]) -> float:
    """
    Estimate the radius of the scene from gaussian means.

    Args:
        splats: The splats parameters.

    Returns:
        The radius of the scene.
    """
    if splats is None or "means" not in splats:
        return 1.0  # Default fallback

    means = splats["means"].cpu().numpy()
    center = np.mean(means, axis=0)
    distances = np.linalg.norm(means - center, axis=1)
    radius = np.max(distances)
    return radius


def sample_poses_on_icosphere(
    level: int,
    scale: float = None,
    intrinsics: np.ndarray = None,
    width: int = None,
    height: int = None,
    splats: dict[str, torch.Tensor] = None,
):
    """
    Sample poses on an icosphere with automatic scale calculation.

    Args:
        level: The level of the icosphere.
        scale: The scale of the icosphere.
        intrinsics: The intrinsic matrix.
        width: The width of the image.
        height: The height of the image.
        splats: The splats parameters.

    Returns:
        The camera poses.
    """
    positions, _ = icosphere(level)
    cam_poses = convert_location_to_rotation(positions)

    if (
        scale is None
        and intrinsics is not None
        and width is not None
        and height is not None
    ):
        scene_radius = estimate_scene_radius(splats)
        scale = calculate_optimal_scale(intrinsics, width, height, scene_radius)
    elif scale is None:
        scale = 1000.0  # Default fallback

    cam_poses[:, :3, 3] *= scale
    all_poses = list(cam_poses)

    for angle_rad in [
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
        np.pi,
        5 * np.pi / 4,
        3 * np.pi / 2,
        7 * np.pi / 4,
    ]:
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        y_rotation = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        rotated_poses = []
        for pose in cam_poses:
            rotated_pose = pose.copy()
            rotated_pose[:3, :3] = pose[:3, :3] @ y_rotation.T
            rotated_poses.append(rotated_pose)
        all_poses.extend(rotated_poses)

    return all_poses


def filter_transparent_gaussians(
    splats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    opacity_threshold = 0.3
    opacities_sigmoid = torch.sigmoid(splats["opacities"])
    valid_gaussians = opacities_sigmoid > opacity_threshold

    splats["means"] = splats["means"][valid_gaussians]
    splats["quats"] = splats["quats"][valid_gaussians]
    splats["scales"] = splats["scales"][valid_gaussians]
    splats["opacities"] = splats["opacities"][valid_gaussians]
    splats["sh0"] = splats["sh0"][valid_gaussians]
    splats["shN"] = splats["shN"][valid_gaussians]

    return splats


def render_object_templates(
    ckpt_path: Path,
    ply_path: Path,
    output_dir: Path,
    icosphere_level: int = 0,
    scale: float = None,
    size: int = 480,
):
    """
    Render object templates.

    Args:
        ckpt_path: The path to the checkpoint.
        ply_path: The path to the input .ply mesh used for rendering depth maps.
        output_dir: The path to the output directory.
        icosphere_level: The level of the icosphere.
        scale: The scale of the icosphere.
        size: The image width and height (default 480).
    """
    width = size
    height = size
    f = float(size)
    intrinsics = np.array(
        [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float32
    )

    assert Path(ckpt_path).exists()
    splats = load_checkpoint(ckpt_path)

    cam_poses = sample_poses_on_icosphere(
        level=icosphere_level,
        scale=scale,
        intrinsics=intrinsics,
        width=width,
        height=height,
        splats=splats,
    )

    render_templates(
        output_dir,
        splats,
        cam_poses,
        intrinsics,
        width,
        height,
    )

    render_depth_maps(ply_path, output_dir, intrinsics, cam_poses, width, height)

    np.savez(
        Path(output_dir) / "rendered_camera_params.npz",
        intrinsics=intrinsics,
        width=width,
        height=height,
        poses=np.array(cam_poses),
    )
