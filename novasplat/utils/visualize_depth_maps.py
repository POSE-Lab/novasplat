from pathlib import Path

import numpy as np
import torch


def se3_inverse(pose):
    """
    Efficiently invert a 4x4 SE(3) transformation matrix (rotation + translation).
    Args:
        pose: (4,4) torch tensor, world-to-camera or camera-to-world
    Returns:
        (4,4) torch tensor, the inverse transformation
    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    R_inv = R.t()
    t_inv = -R_inv @ t
    pose_inv = torch.eye(4, device=pose.device, dtype=pose.dtype)
    pose_inv[:3, :3] = R_inv
    pose_inv[:3, 3] = t_inv
    return pose_inv


def backproject_depth_to_points(depth, K, pose, device="cuda"):
    """
    Backprojects a depth map to 3D points in world coordinates.

    Args:
        depth: (H, W) depth map, where each value is the positive Z coordinate in camera space (points in front have positive Z)
        K: (3,3) camera intrinsics
        pose: (4,4) camera-to-world transformation (camera -> world)
            If passing a world-to-camera pose (as produced by novasplat.templates look_at),
            you must invert it before calling this function.
        device: Torch device
    Returns:
        (N, 3) 3D points in world coordinates
    Note:
        This assumes the camera coordinate system has +Z forward (OpenCV/colmap convention),
        and depth is the positive Z value in camera space.
    """
    H, W = depth.shape
    depth = depth.to(device)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = torch.meshgrid(
        torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy"
    )

    Z = depth.reshape(-1)
    valid = Z > 0
    Z = Z[valid]

    u = u.reshape(-1)[valid]
    v = v.reshape(-1)[valid]

    x = (u - cx) / fx
    y = (v - cy) / fy

    # NOTE: +Z forward (OpenCV/colmap convention)
    Xc = torch.stack([x * Z, y * Z, Z], dim=1)

    R = pose[:3, :3]
    t = pose[:3, 3]
    Xw = (R @ Xc.T).T + t

    return Xw


def save_depth_npz_to_ply(
    depth_npz_path, ply_path, K, pose, device="cuda", pose_is_world_to_cam=False
):
    """
    Loads a depth .npz file, backprojects to 3D points, and saves as a .ply file.

    Args:
        depth_npz_path: Path to the .npz file (must contain 'depth' array).
        ply_path: Output path for the .ply file.
        K: (3,3) camera intrinsics (numpy array or torch tensor).
        pose: (4,4) pose matrix (numpy array or torch tensor).
            If pose_is_world_to_cam is True, pose is world-to-camera and will be inverted.
            If False, pose is assumed to be camera-to-world.
        device: Device for torch tensors.
        pose_is_world_to_cam: If True, invert pose to get camera-to-world.
    """
    # Load depth
    depth = np.load(depth_npz_path)["depth"]
    # Remove singleton channel dimension if present
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    depth = torch.from_numpy(depth).float().to(device)

    # Convert K and pose to torch if needed
    if not torch.is_tensor(K):
        K = torch.from_numpy(K).float().to(device)
    if not torch.is_tensor(pose):
        pose = torch.from_numpy(pose).float().to(device)

    # Invert pose if explicitly requested
    if pose_is_world_to_cam:
        pose = se3_inverse(pose)

    # Backproject to 3D points
    points = backproject_depth_to_points(depth, K, pose, device=device)  # (N, 3)

    # Save as .ply
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points.cpu().numpy():
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def main(depth_dir, cam_params_path, output_dir, device="cuda"):
    """
    Converts all depth_*.npz files in depth_dir to PLY files using camera parameters.

    Args:
        depth_dir: Directory containing depth_*.npz files.
        cam_params_path: Path to rendered_camera_params.npz (poses are world-to-camera, will be inverted).
        output_dir: Directory to save PLY files.
        device: Torch device.
    """
    depth_dir = Path(depth_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load camera parameters
    cam_params = np.load(cam_params_path)
    intrinsics = cam_params["intrinsics"]
    poses = cam_params["poses"]

    # Find all depth files
    depth_files = sorted(depth_dir.glob("depth_*.npz"))

    for i, depth_file in enumerate(depth_files):
        ply_path = output_dir / f"cloud_{i:04d}.ply"
        pose = poses[i]
        # rendered_camera_params.npz stores world-to-camera poses, so set flag True
        save_depth_npz_to_ply(
            depth_npz_path=depth_file,
            ply_path=ply_path,
            K=intrinsics,
            pose=pose,
            device=device,
            pose_is_world_to_cam=True,
        )
        print(f"Saved {ply_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert depth maps to PLY point clouds using camera parameters."
    )
    parser.add_argument(
        "--depth_dir",
        type=str,
        required=True,
        help="Directory containing depth_*.npz files.",
    )
    parser.add_argument(
        "--cam_params",
        type=str,
        required=True,
        help="Path to rendered_camera_params.npz.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save PLY files."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Torch device (default: cuda)"
    )
    args = parser.parse_args()
    main(args.depth_dir, args.cam_params, args.output_dir, args.device)
