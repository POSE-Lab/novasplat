import numpy as np
import torch
import open3d as o3d
from novasplat.utils.transforms import inv_resize_to_patches


PATCH_SIZE = 16


def compute_pca_diameter(points: torch.Tensor):
    """Compute a diameter of the point cloud using PCA."""
    centered = points - points.mean(dim=0)
    U, S, V = torch.pca_lowrank(centered, q=1)
    proj = centered @ V[:, 0]
    return (proj.max() - proj.min()).item()


def backproject_rgbd(u, v, depth, intrinsics, campose=None):
    """
    Backproject RGB-D pixels to 3D world coordinates (COLMAP convention).

    Args:
        u, v: [H, W] pixel coordinate grids
        depth: [H, W] depth values (meters)
        intrinsics: [3, 3] camera intrinsic matrix
        campose: [4, 4] world->camera matrix (COLMAP cam_from_world)

    Returns:
        points_world: [N, 3] 3D world points
    """
    if campose is None:
        campose = np.eye(4)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # TODO: Check the following copilot comment:
    # The internal valid = depth > 0 filtering has been commented out, but the
    # docstring still describes u, v, depth as [H, W] grids and the
    # implementation will produce a wrongly-shaped output if 2D inputs are
    # passed. Either restore/support 2D-grid inputs (flatten internally) or
    # update the docstring and enforce 1D [N] inputs explicitly.
    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u)
        v = torch.from_numpy(v)
        depth = torch.from_numpy(depth)

    z = depth
    # --- Backproject to camera coordinates ---
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_cam = torch.stack((x, y, z), dim=1)  # [N, 3]

    # --- Extract R and t (world -> camera) ---
    R = campose[:3, :3]
    t = campose[:3, 3]

    # Convert to torch if needed
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R).to(points_cam.dtype)
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).to(points_cam.dtype)

    # --- Invert pose (camera -> world) ---
    R_inv = R.t()
    t_inv = -R_inv @ t

    # --- Transform to world coordinates ---
    points_world = (R_inv @ points_cam.t()).t() + t_inv

    return points_world


def points_to_pcd(points, device="cuda"):
    """
    Backproject depth map to Open3D point cloud.
    """
    points = points.to(device)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu())

    return pcd


def filter_pcd(pcd, eps_frac=0.08, min_points=20):
    """
    Filter pcd and keep largest connected component.
    eps is relative to the point-cloud diameter.
    """
    pcd.remove_non_finite_points()

    if len(pcd.points) < 2:
        return pcd

    points = torch.from_numpy(np.asarray(pcd.points)).float()
    diameter = compute_pca_diameter(points)

    eps = float(diameter) * eps_frac

    labels = np.asarray(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    valid = labels >= 0

    if not np.any(valid):
        return pcd

    largest_label = np.bincount(labels[valid]).argmax()
    keep_idx = np.where(labels == largest_label)[0]

    return pcd.select_by_index(keep_idx)


def compute_xyz(dino, idx, patch=False):
    data = dino[idx]

    bbox = data["bbox"]
    x0, y0 = bbox[:2]

    resized_shape = (
        data["resized_image"].shape[1:]
        if "resized_image" in data
        else data["resized_image_shape"]
    )

    pu, pv = inv_resize_to_patches(
        (bbox[2] - bbox[0], bbox[3] - bbox[1]),
        resized_shape,
        PATCH_SIZE,
    )

    pu += y0
    pv += x0

    depth = data["depth"]

    if patch:
        pu = pu[PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE]
        pv = pv[PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE]
        depth = depth[PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE]

    XYZ = backproject_rgbd(
        pv.flatten(),
        pu.flatten(),
        depth.flatten(),
        data["intrinsics"],
        data["pose"],
    )

    return XYZ
