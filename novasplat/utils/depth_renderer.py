from pathlib import Path
from tqdm import tqdm
from typing import List

import cv2
import numpy as np
import open3d as o3d
from numba import njit, prange, get_num_threads, get_thread_id


@njit
def rasterize_triangle(depth: np.ndarray, pts: np.ndarray):
    """
    Rasterizes a single triangle into the provided depth buffer using
    barycentric interpolation.

    This function updates the depth buffer in-place by projecting a triangle
    defined by three points (in image coordinates with depth) and filling in
    the pixels inside the triangle with the minimum depth value.
    For each pixel covered by the triangle, the function computes the
    barycentric coordinates and interpolates the depth (z-value). If the
    interpolated depth is less than the current value in the depth buffer at
    that pixel, the buffer is updated.

    Args:
        depth (np.ndarray): 2D array of shape (H, W) representing the depth
            buffer. Modified in-place.
        pts (np.ndarray): 3x3 array where each row is (u, v, z) for a triangle
            vertex in image coordinates (u, v) and depth (z).

    Returns:
        None. The function modifies the depth buffer in-place.

    Notes:
        - Pixels outside the triangle are not modified.
        - The function assumes the triangle is defined in screen space
            (pixel coordinates and depth).
        - If the triangle is degenerate (zero area), the function returns
            immediately.
    """
    p0, p1, p2 = pts
    min_x = max(int(np.floor(np.min(pts[:, 0]))), 0)
    max_x = min(int(np.ceil(np.max(pts[:, 0]))), depth.shape[1] - 1)
    min_y = max(int(np.floor(np.min(pts[:, 1]))), 0)
    max_y = min(int(np.ceil(np.max(pts[:, 1]))), depth.shape[0] - 1)

    area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
    eps = 1e-8
    if abs(area) < eps:
        return

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            w0 = (p1[0] - p0[0]) * (y - p0[1]) - (x - p0[0]) * (p1[1] - p0[1])
            w1 = (p2[0] - p1[0]) * (y - p1[1]) - (x - p1[0]) * (p2[1] - p1[1])
            w2 = (p0[0] - p2[0]) * (y - p2[1]) - (x - p2[0]) * (p0[1] - p2[1])
            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                alpha = (
                    (p1[0] - p2[0]) * (y - p2[1]) - (x - p2[0]) * (p1[1] - p2[1])
                ) / area
                beta = (
                    (p2[0] - p0[0]) * (y - p0[1]) - (x - p0[0]) * (p2[1] - p0[1])
                ) / area
                gamma = 1.0 - alpha - beta

                # Perspective-correct depth interpolation: interpolate reciprocal depth (1/z)
                inv_z0 = 1.0 / (p0[2] + eps)
                inv_z1 = 1.0 / (p1[2] + eps)
                inv_z2 = 1.0 / (p2[2] + eps)
                inv_z = alpha * inv_z0 + beta * inv_z1 + gamma * inv_z2

                # Skip if invalid interpolated reciprocal depth
                if inv_z <= 0.0:
                    continue

                z = 1.0 / inv_z
                if z < depth[y, x]:
                    depth[y, x] = z


@njit(parallel=True)
def rasterize_mesh(
    depth: np.ndarray,
    faces: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    Z: np.ndarray,
    thread_buffers=None,
):
    """
    Parallel and safe mesh rasterization using per-thread depth buffers.
    Each thread rasterizes a subset of faces into its own buffer, then the global
    depth buffer is computed as the minimum across all per-thread buffers.

    Args:
        depth (np.ndarray): 2D array of shape (H, W) representing the depth buffer. Modified in-place.
        faces (np.ndarray): (N, 3) triangle indices.
        u, v, Z (np.ndarray): vertex projections and depths.
        thread_buffers (np.ndarray, optional): Preallocated per-thread buffers of
            shape (n_threads, H, W) and same dtype as ``depth``. If provided, it
            will be reused (and reset to ``np.inf``) instead of allocating a new
            buffer on each call.

    Returns:
        None. The function modifies the depth buffer in-place.
    """
    n_threads = get_num_threads()
    H, W = depth.shape
    # Allocate or reuse per-thread depth buffers
    if thread_buffers is None:
        thread_buffers = np.full((n_threads, H, W), np.inf, dtype=depth.dtype)
    else:
        # Reset contents to inf
        for t in range(n_threads):
            for y in range(H):
                for x in range(W):
                    thread_buffers[t, y, x] = np.inf
    n_faces = faces.shape[0]
    # Each thread writes only to its own buffer
    for i in prange(n_faces):
        thread_id = get_thread_id()
        f = faces[i]
        pts = np.array(
            [
                [u[f[0]], v[f[0]], Z[f[0]]],
                [u[f[1]], v[f[1]], Z[f[1]]],
                [u[f[2]], v[f[2]], Z[f[2]]],
            ],
            dtype=np.float32,
        )
        rasterize_triangle(thread_buffers[thread_id], pts)
    # Reduce: take minimum across all thread buffers (Numba does not support np.min with axis)
    for y in range(H):
        for x in range(W):
            min_z = np.inf
            for t in range(n_threads):
                if thread_buffers[t, y, x] < min_z:
                    min_z = thread_buffers[t, y, x]
            depth[y, x] = min_z


def render_depth_maps(
    mesh_path: Path,
    output_dir: Path,
    intrinsics: np.ndarray,
    poses: List[np.ndarray],
    width: int,
    height: int,
):
    """
    Renders and saves depth maps for a mesh from multiple camera poses.

    For each camera pose, this function projects the mesh vertices into camera
    space, computes the depth map by rasterizing the mesh triangles, and saves
    both the raw and normalized depth maps to the output directory.
    The function creates a 'depth' subdirectory in the output directory if it
    does not exist.

    Args:
        mesh_path (Path): Path to the mesh file to be rendered
            (supported by Open3D).
        output_dir (Path): Output directory where depth maps will be saved.
        intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        poses (List[np.ndarray]): List of 4x4 world-to-camera (view)
            transformation matrices that map world coordinates to
            camera coordinates.
        width (int): Width of the output depth maps (in pixels).
        height (int): Height of the output depth maps (in pixels).

    Returns:
        None. The function saves .npz and .png files for each pose in the
            output directory.

    Notes:
        - The function uses Open3D to load the mesh and assumes the mesh is
            watertight and manifold.
        - Depth maps are saved as compressed .npz files (raw depth) and as
            8-bit .png images (normalized depth).
        - Pixels with no intersection are set to 0 in the output depth maps.
        - The function uses tqdm to display progress.
    """
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    verts_h = np.concatenate(
        [verts, np.ones((verts.shape[0], 1), dtype=np.float32)], axis=1
    )

    # Preallocate per-thread buffers for rasterization (reuse for all poses)
    n_threads = get_num_threads()
    thread_buffers = np.full((n_threads, height, width), np.inf, dtype=np.float32)

    for i, pose in enumerate(tqdm(poses, desc="Rendering depth maps")):
        verts_cam = (pose @ verts_h.T).T[:, :3]

        X, Y, Z = verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2]

        # Define a small positive near plane to avoid division by zero and
        # to cull geometry that is at or behind the camera.
        near = 1e-6
        valid_verts = Z > near

        # Cull faces that reference any vertex that is not in front of the camera.
        face_mask = np.all(valid_verts[faces], axis=1)
        faces_valid = faces[face_mask]

        depth = np.full((height, width), np.inf, dtype=np.float32)

        # If no faces are valid for this pose, skip rasterization.
        if faces_valid.size == 0:
            depth[depth == np.inf] = 0.0

            np.savez_compressed(depth_dir / f"depth_{i:04d}.npz", depth=depth)

            depth_range = depth.max() - depth.min() + 1e-8
            depth_norm = (depth - depth.min()) / depth_range
            cv2.imwrite(
                str(depth_dir / f"depth_{i:04d}.png"),
                (depth_norm * 255).astype(np.uint8),
            )
            continue

        # Use a safe depth for projection to avoid division by zero.
        Z_safe = np.where(valid_verts, Z, near)
        u = intrinsics[0, 0] * X / Z_safe + intrinsics[0, 2]
        v = intrinsics[1, 1] * Y / Z_safe + intrinsics[1, 2]

        rasterize_mesh(depth, faces_valid, u, v, Z_safe, thread_buffers=thread_buffers)
        # Compute normalization range using only valid (finite) depth values
        valid_mask = np.isfinite(depth)
        if np.any(valid_mask):
            valid_depth = depth[valid_mask]
            dmin = valid_depth.min()
            dmax = valid_depth.max()
            depth_range = dmax - dmin + 1e-8
            depth_norm = np.zeros_like(depth, dtype=np.float32)
            depth_norm[valid_mask] = (depth[valid_mask] - dmin) / depth_range
        else:
            # No valid depth values; produce an all-zero normalized map
            depth_norm = np.zeros_like(depth, dtype=np.float32)

        # Save depth map with invalid pixels set to 0.0 to match existing behavior
        depth_to_save = depth.copy()
        depth_to_save[~valid_mask] = 0.0
        np.savez_compressed(depth_dir / f"depth_{i:04d}.npz", depth=depth_to_save)
        cv2.imwrite(
            str(depth_dir / f"depth_{i:04d}.png"), (depth_norm * 255).astype(np.uint8)
        )
