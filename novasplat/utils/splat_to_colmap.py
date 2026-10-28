"""Splat → COLMAP-with-visibility exporter.

Converts a trained 3D Gaussian Splat (inria-format .ply) plus its source COLMAP
sparse reconstruction into a new COLMAP reconstruction whose points3D are
derived from splat centers and whose per-image points2D reflect splat
visibility computed via splat-depth occlusion testing.

The output is consumed by the photogrammetry ImportColmapSFM stage to seed
DensifyPointCloud in OpenMVS.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import pycolmap
import torch
from plyfile import PlyData

logger = logging.getLogger(__name__)


# Spherical-harmonic DC coefficient used by the inria/gsplat convention.
_SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# T1 — Splat .ply loader + opacity/scale filter
# ---------------------------------------------------------------------------


class Splats(TypedDict):
    means: torch.Tensor  # [N, 3] float32, COLMAP world coordinates
    quats: torch.Tensor  # [N, 4] float32, normalized (w, x, y, z)
    scales: torch.Tensor  # [N, 3] float32, post-exp (real sigma)
    opacities: torch.Tensor  # [N]    float32, post-sigmoid in [0, 1]
    colors: torch.Tensor  # [N, 3] float32, in [0, 1] (post-SH-DC)


_REQUIRED_PLY_FIELDS = (
    "x",
    "y",
    "z",
    "scale_0",
    "scale_1",
    "scale_2",
    "opacity",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
)


def load_and_filter_splats_from_ply(
    ply_path: Path,
    *,
    min_opacity: float = 0.1,
    max_scale_frac_of_scene: float = 0.05,
    scene_scale: float,
    device: str | torch.device = "cpu",
) -> Splats:
    """Load an inria-format Gaussian-splat .ply and apply opacity + scale filters.

    Keep splat i iff
        sigmoid(opacity[i]) > min_opacity
        AND max(exp(scale[i, :])) < max_scale_frac_of_scene * scene_scale.

    Raises:
        FileNotFoundError: if ply_path doesn't exist.
        ValueError: if required fields are missing.
        RuntimeError: if 0 splats pass the filter.
    """
    ply_path = Path(ply_path)
    if not ply_path.is_file():
        raise FileNotFoundError(f"Splat .ply not found: {ply_path}")

    plydata = PlyData.read(ply_path.as_posix())
    if "vertex" not in {el.name for el in plydata.elements}:
        raise ValueError(f"Splat .ply is missing the 'vertex' element: {ply_path}")

    vert = plydata["vertex"]
    available = set(vert.data.dtype.names or ())
    missing = [name for name in _REQUIRED_PLY_FIELDS if name not in available]
    if missing:
        raise ValueError(
            f"Splat .ply is missing required field(s): {missing}. Path: {ply_path}"
        )

    n_in = int(vert.count)
    logger.info("Loaded %d splats from %s", n_in, ply_path)

    xyz = np.stack(
        [
            np.asarray(vert["x"], dtype=np.float32),
            np.asarray(vert["y"], dtype=np.float32),
            np.asarray(vert["z"], dtype=np.float32),
        ],
        axis=1,
    )
    raw_scales = np.stack(
        [
            np.asarray(vert["scale_0"], dtype=np.float32),
            np.asarray(vert["scale_1"], dtype=np.float32),
            np.asarray(vert["scale_2"], dtype=np.float32),
        ],
        axis=1,
    )
    raw_opacity = np.asarray(vert["opacity"], dtype=np.float32)
    quats = np.stack(
        [
            np.asarray(vert["rot_0"], dtype=np.float32),
            np.asarray(vert["rot_1"], dtype=np.float32),
            np.asarray(vert["rot_2"], dtype=np.float32),
            np.asarray(vert["rot_3"], dtype=np.float32),
        ],
        axis=1,
    )
    f_dc = np.stack(
        [
            np.asarray(vert["f_dc_0"], dtype=np.float32),
            np.asarray(vert["f_dc_1"], dtype=np.float32),
            np.asarray(vert["f_dc_2"], dtype=np.float32),
        ],
        axis=1,
    )

    real_scales = np.exp(raw_scales)
    sigmoid_opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
    max_scale_cap = float(max_scale_frac_of_scene) * float(scene_scale)

    keep = (sigmoid_opacity > float(min_opacity)) & (
        real_scales.max(axis=1) < max_scale_cap
    )
    n_keep = int(keep.sum())
    logger.info(
        "Filtered splats: kept %d / %d (opacity > %g, max_scale < %g)",
        n_keep,
        n_in,
        min_opacity,
        max_scale_cap,
    )
    if n_keep == 0:
        raise RuntimeError(
            f"0 splats passed the opacity/scale filter "
            f"(min_opacity={min_opacity}, max_scale_cap={max_scale_cap}, scene_scale={scene_scale})."
        )

    xyz = xyz[keep]
    real_scales = real_scales[keep]
    sigmoid_opacity = sigmoid_opacity[keep]
    quats = quats[keep]
    f_dc = f_dc[keep]

    # Normalize quaternions. The inria pipeline does not always normalize.
    q_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    quats = quats / q_norms

    colors = np.clip(0.5 + _SH_C0 * f_dc, 0.0, 1.0).astype(np.float32)

    dev = torch.device(device)
    return Splats(
        means=torch.from_numpy(xyz.astype(np.float32)).to(dev),
        quats=torch.from_numpy(quats.astype(np.float32)).to(dev),
        scales=torch.from_numpy(real_scales.astype(np.float32)).to(dev),
        opacities=torch.from_numpy(sigmoid_opacity.astype(np.float32)).to(dev),
        colors=torch.from_numpy(colors).to(dev),
    )


# ---------------------------------------------------------------------------
# T2 — Camera helpers: undistorted intrinsics + distorted projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraUndistortion:
    K_undist: np.ndarray  # [3, 3] float64, undistorted PINHOLE intrinsic
    width: int  # post-undistortion image width
    height: int  # post-undistortion image height


def _camera_K_and_params(
    cam: pycolmap.Camera,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Decompose a pycolmap.Camera into (K, distortion_params, camtype).

    Mirrors the if/elif tree in radiance/ddgsplat/colmap.py:Parser.
    """
    model = cam.model
    if model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
        f, cx, cy = cam.params
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K, np.empty(0, dtype=np.float32), "pinhole"
    if model == pycolmap.CameraModelId.PINHOLE:
        fx, fy, cx, cy = cam.params
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K, np.empty(0, dtype=np.float32), "pinhole"
    if model == pycolmap.CameraModelId.SIMPLE_RADIAL:
        f, cx, cy, k1 = cam.params
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        params = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float32)
        return K, params, "perspective"
    if model == pycolmap.CameraModelId.RADIAL:
        f, cx, cy, k1, k2 = cam.params
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        params = np.array([k1, k2, 0.0, 0.0], dtype=np.float32)
        return K, params, "perspective"
    if model == pycolmap.CameraModelId.OPENCV:
        fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        params = np.array([k1, k2, p1, p2], dtype=np.float32)
        return K, params, "perspective"
    if model == pycolmap.CameraModelId.OPENCV_FISHEYE:
        fx, fy, cx, cy, k1, k2, k3, k4 = cam.params
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        params = np.array([k1, k2, k3, k4], dtype=np.float32)
        return K, params, "fisheye"
    raise ValueError(
        f"Unsupported COLMAP camera model: {model}. "
        "Supported: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE."
    )


def _fisheye_undistortion(
    K: np.ndarray, params: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, int, int]:
    """Compute (K_undist, width_undist, height_undist) for OPENCV_FISHEYE.

    Reuses the meshgrid + theta-poly trick from ddgsplat/colmap.py Parser
    (fisheye branch). We only need the cropped region's K and dimensions —
    no remap data.
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    x1 = (grid_x - cx) / fx
    y1 = (grid_y - cy) / fy
    theta = np.sqrt(x1**2 + y1**2)
    r = (
        1.0
        + params[0] * theta**2
        + params[1] * theta**4
        + params[2] * theta**6
        + params[3] * theta**8
    )
    mapx = fx * x1 * r + width // 2
    mapy = fy * y1 * r + height // 2
    valid = (mapx > 0) & (mapy > 0) & (mapx < width - 1) & (mapy < height - 1)
    if not valid.any():
        raise ValueError("Fisheye undistortion produced no valid pixels.")
    y_idx, x_idx = np.nonzero(valid)
    y_min, y_max = int(y_idx.min()), int(y_idx.max()) + 1
    x_min, x_max = int(x_idx.min()), int(x_idx.max()) + 1
    K_undist = K.copy()
    K_undist[0, 2] -= x_min
    K_undist[1, 2] -= y_min
    return K_undist, x_max - x_min, y_max - y_min


def build_camera_undistortion(
    reconstruction: pycolmap.Reconstruction,
) -> dict[int, CameraUndistortion]:
    """For each camera_id, compute the PINHOLE intrinsic + post-undistortion size."""
    out: dict[int, CameraUndistortion] = {}
    for camera_id, cam in reconstruction.cameras.items():
        K, params, camtype = _camera_K_and_params(cam)
        width = int(cam.width)
        height = int(cam.height)
        if camtype == "pinhole" or params.size == 0:
            out[camera_id] = CameraUndistortion(
                K_undist=K.astype(np.float64),
                width=width,
                height=height,
            )
            continue
        if camtype == "perspective":
            K_undist, roi = cv2.getOptimalNewCameraMatrix(
                K.astype(np.float64), params, (width, height), 0
            )
            _, _, w_und, h_und = roi
            if w_und <= 0 or h_und <= 0:
                # Degenerate ROI — fall back to original size.
                w_und, h_und = width, height
            out[camera_id] = CameraUndistortion(
                K_undist=np.asarray(K_undist, dtype=np.float64),
                width=int(w_und),
                height=int(h_und),
            )
        elif camtype == "fisheye":
            K_undist, w_und, h_und = _fisheye_undistortion(K, params, width, height)
            out[camera_id] = CameraUndistortion(
                K_undist=K_undist.astype(np.float64),
                width=int(w_und),
                height=int(h_und),
            )
        else:
            raise ValueError(f"Unsupported camtype: {camtype}")
    return out


def _image_camera_center(image: pycolmap.Image) -> np.ndarray:
    """Return the world-space camera center for a pycolmap.Image."""
    cam_from_world = image.cam_from_world()
    R = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
    t = np.asarray(cam_from_world.translation, dtype=np.float64).reshape(3)
    return -R.T @ t


def compute_scene_scale(reconstruction: pycolmap.Reconstruction) -> float:
    """Return max distance between any camera center and the camera-center centroid."""
    centers = np.stack(
        [_image_camera_center(img) for img in reconstruction.images.values()],
        axis=0,
    )
    centroid = centers.mean(axis=0)
    dists = np.linalg.norm(centers - centroid, axis=1)
    return float(dists.max())


def build_world_to_undist_viewmats(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[list[int], np.ndarray]:
    """Return (ordered_image_ids, w2c_matrices) sorted by ascending image_id.

    Each row of w2c_matrices is a 4x4 float32 cam_from_world matrix.
    """
    image_ids = sorted(reconstruction.images.keys())
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    mats = []
    for image_id in image_ids:
        image = reconstruction.images[image_id]
        m34 = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        m44 = np.concatenate([m34, bottom], axis=0)
        mats.append(m44)
    return image_ids, np.stack(mats, axis=0).astype(np.float32)


def project_point_distorted(
    point_world: np.ndarray,
    image: pycolmap.Image,
    camera: pycolmap.Camera,
) -> tuple[float, float] | None:
    """Project a world point through the original distorted camera model.

    Returns (u, v) in original-image pixel coords, or None if the point is
    behind the camera (z <= 0).
    """
    cam_from_world = image.cam_from_world()
    R = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
    t = np.asarray(cam_from_world.translation, dtype=np.float64).reshape(3)
    pt = R @ np.asarray(point_world, dtype=np.float64).reshape(3) + t
    if pt[2] <= 0.0:
        return None
    # pycolmap 3.12: img_from_cam expects a 3D camera-frame point and handles
    # the distortion model internally. Passing 2D normalized coords is
    # deprecated.
    uv = np.asarray(camera.img_from_cam(pt), dtype=np.float64).reshape(2)
    return float(uv[0]), float(uv[1])


# ---------------------------------------------------------------------------
# T3 — Visibility kernel: depth-render + occlusion test
# ---------------------------------------------------------------------------


def _splats_device(splats: Splats) -> torch.device:
    return splats["means"].device


def _canonical_device(d: torch.device) -> torch.device:
    """Resolve `cuda` (no index) to the current cuda device so that
    `torch.device("cuda")` compares equal to `torch.device("cuda:0")` on a
    single-GPU host.
    """
    if d.type == "cuda" and d.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return d


def compute_visibility(
    *,
    splats: Splats,
    viewmats: np.ndarray | torch.Tensor,
    image_ids: list[int],
    K_undist_per_image: list[np.ndarray],
    width_per_image: list[int],
    height_per_image: list[int],
    depth_tol_relative: float = 0.01,
    depth_downsample: int = 2,
    device: str | torch.device = "cuda",
    debug_callback=None,
) -> dict[int, torch.Tensor]:
    """For each image_id, return a 1-D bool tensor of length N (number of splats)
    marking which splats are visible from that image.

    Per image: render a depth map with gsplat (RGB+ED, downsampled by
    depth_downsample), then for every splat: project to the downsampled
    pixel grid, sample the rendered depth, and check
        |z_point - depth_render| / z_point < depth_tol_relative.

    The returned tensors are on CPU. Only one depth map lives on GPU at a time.

    If `debug_callback` is provided, it is invoked per image as
        debug_callback(image_id, depth_map_torch, status_array_np, uv_full_np)
    where:
        depth_map_torch: [H_ds, W_ds] float32 tensor (still on render device).
        status_array_np: [N] int8 with 0=out-of-frustum, 1=visible, 2=occluded.
        uv_full_np: [N, 2] float32 in original-resolution undistorted pixel
            coords (NaN for points behind the camera).
    """
    from gsplat.rendering import rasterization

    dev = torch.device(device)
    means = splats["means"]
    if _canonical_device(means.device) != _canonical_device(dev):
        raise ValueError(
            f"splats.means.device ({means.device}) != requested device ({dev}). "
            "Move splats to `device` before calling compute_visibility."
        )

    M = len(image_ids)
    if not (
        len(K_undist_per_image) == M
        and len(width_per_image) == M
        and len(height_per_image) == M
    ):
        raise ValueError("image_ids and per-image lists must have the same length.")

    if isinstance(viewmats, np.ndarray):
        viewmats_t = torch.from_numpy(viewmats.astype(np.float32)).to(dev)
    else:
        viewmats_t = viewmats.to(dev).float()
    if viewmats_t.shape[0] != M or viewmats_t.shape[-2:] != (4, 4):
        raise ValueError(
            f"viewmats must be shape [{M}, 4, 4]; got {tuple(viewmats_t.shape)}"
        )

    quats = splats["quats"]
    scales = splats["scales"]
    opacities = splats["opacities"]
    colors = splats["colors"]

    N = means.shape[0]
    out: dict[int, torch.Tensor] = {}

    for i, image_id in enumerate(image_ids):
        W_full = int(width_per_image[i])
        H_full = int(height_per_image[i])
        W_i = max(1, W_full // depth_downsample)
        H_i = max(1, H_full // depth_downsample)
        K_full = np.asarray(K_undist_per_image[i], dtype=np.float64)
        K_ds = K_full.copy()
        K_ds[:2, :] /= float(depth_downsample)

        K_ds_t = torch.from_numpy(K_ds.astype(np.float32)).to(dev).unsqueeze(0)
        viewmat_i = viewmats_t[i : i + 1]

        renders, _alphas, _info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat_i,
            Ks=K_ds_t,
            width=W_i,
            height=H_i,
            render_mode="RGB+ED",
        )
        depth = renders[0, ..., -1]  # [H_i, W_i] float32

        R = viewmat_i[0, :3, :3]  # [3, 3]
        t = viewmat_i[0, :3, 3]  # [3]
        pts_cam = means @ R.T + t  # [N, 3]
        z = pts_cam[:, 2]

        # Project at full undistorted resolution using K_full.
        K_full_t = torch.from_numpy(K_full.astype(np.float32)).to(dev)
        # Avoid div-by-zero — clamp z minimally; we'll mask by z > 0 anyway.
        z_safe = torch.where(z > 0, z, torch.ones_like(z))
        xn = pts_cam[:, 0] / z_safe
        yn = pts_cam[:, 1] / z_safe
        u_full = K_full_t[0, 0] * xn + K_full_t[0, 2]
        v_full = K_full_t[1, 1] * yn + K_full_t[1, 2]

        u_ds = u_full / float(depth_downsample)
        v_ds = v_full / float(depth_downsample)
        u_idx = torch.floor(u_ds).long()
        v_idx = torch.floor(v_ds).long()

        in_frustum = (
            (z > 0) & (u_idx >= 0) & (u_idx < W_i) & (v_idx >= 0) & (v_idx < H_i)
        )

        # Safe-clamp indices for gather; we mask out-of-frustum afterwards.
        u_clamped = u_idx.clamp(0, W_i - 1)
        v_clamped = v_idx.clamp(0, H_i - 1)
        depth_at = depth[v_clamped, u_clamped]

        nonzero_depth = depth_at > 0
        rel_err = torch.abs(z - depth_at) / z_safe
        visible = in_frustum & nonzero_depth & (rel_err < float(depth_tol_relative))

        if debug_callback is not None:
            # 1 = visible, 2 = in-frustum but occluded, 0 = out-of-frustum
            status = torch.zeros(N, dtype=torch.int8, device=dev)
            status[in_frustum] = 2
            status[visible] = 1
            uv_full = torch.stack([u_full, v_full], dim=1)
            uv_full = torch.where(
                (z > 0).unsqueeze(1),
                uv_full,
                torch.full_like(uv_full, float("nan")),
            )
            debug_callback(
                image_id,
                depth.detach().cpu(),
                status.detach().cpu().numpy(),
                uv_full.detach().cpu().numpy(),
            )

        out[image_id] = visible.detach().cpu()
        # Free the depth map before the next iteration.
        del renders, depth, depth_at, rel_err

    logger.debug("compute_visibility: %d images, %d candidate splats", M, N)
    return out


# ---------------------------------------------------------------------------
# T4 — Export orchestrator: assemble points/observations and write COLMAP
# ---------------------------------------------------------------------------


def _make_track(observations: list[tuple[int, int]]) -> pycolmap.Track:
    """Build a pycolmap.Track from a list of (image_id, point2D_idx).

    Different pycolmap versions expose either ``add_element`` or ``append`` on
    Track. Try ``add_element`` first, fall back to ``append``.
    """
    track = pycolmap.Track()
    for image_id, p2d_idx in observations:
        elem = pycolmap.TrackElement(image_id, p2d_idx)
        if hasattr(track, "add_element"):
            try:
                track.add_element(image_id, p2d_idx)
            except TypeError:
                track.add_element(elem)
        else:
            track.append(elem)
    return track


def _make_point2d_no_pid(xy: np.ndarray) -> pycolmap.Point2D:
    """Construct a pycolmap.Point2D with only the pixel coords set.

    The point3D_id field is left at its invalid default. Reconstruction.add_point3D
    will assign it via SetPoint3DForPoint2D once the corresponding track is added.
    """
    p2d = pycolmap.Point2D()
    p2d.xy = xy
    return p2d


def _set_image_points2D(
    image: pycolmap.Image, p2d_list: list[pycolmap.Point2D]
) -> None:
    """Replace image.points2D with a new list.

    pycolmap 3.12's ``SetPoints2D`` only accepts an assignment when the
    underlying points2D vector is empty (it has a Check that fails otherwise).
    To replace a non-empty list, first clear it via the pybind-vector
    ``.clear()`` proxy, then assign the new list.
    """
    image.points2D.clear()
    try:
        image.points2D = p2d_list
        return
    except TypeError:
        pass
    container_cls = getattr(pycolmap, "Point2DList", None) or getattr(
        pycolmap, "ListPoint2D", None
    )
    if container_cls is not None:
        image.points2D = container_cls(p2d_list)
        return
    raise RuntimeError(
        "Could not assign Image.points2D — pycolmap API not as expected."
    )


def _write_overlay_png(
    out_path: Path,
    depth_map: torch.Tensor,
    status: np.ndarray,
    uv_full: np.ndarray,
    width_full: int,
    height_full: int,
) -> None:
    """Write a per-image diagnostic overlay PNG."""
    depth_np = depth_map.numpy() if isinstance(depth_map, torch.Tensor) else depth_map
    depth_np = np.asarray(depth_np, dtype=np.float32)
    if depth_np.size == 0:
        return
    dmax = float(depth_np.max())
    if not np.isfinite(dmax) or dmax <= 0:
        norm = np.zeros_like(depth_np, dtype=np.uint8)
    else:
        norm = np.clip(depth_np / dmax, 0.0, 1.0)
        norm = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    # Mask out zero-depth pixels (no splat coverage).
    colored[depth_np <= 0] = 0
    overlay = cv2.resize(
        colored, (width_full, height_full), interpolation=cv2.INTER_NEAREST
    )
    for j in range(uv_full.shape[0]):
        if status[j] == 0:
            continue
        u, v = uv_full[j, 0], uv_full[j, 1]
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < width_full and 0 <= vi < height_full):
            continue
        if status[j] == 1:
            color = (0, 200, 0)
        else:
            color = (0, 0, 200)
        cv2.circle(overlay, (ui, vi), 2, color, -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path.as_posix(), overlay)


def _assemble_and_write(
    *,
    splats: Splats,
    reconstruction: pycolmap.Reconstruction,
    visibility: dict[int, torch.Tensor],
    out_dir: Path,
    min_track_len: int,
    point_error_init: float,
    distorted_uvs: dict[int, dict[int, tuple[float, float]]] | None = None,
    debug_dir: Path | None = None,
) -> dict[str, int | float]:
    """Build points3D + per-image points2D from a precomputed visibility dict
    and write the COLMAP binary files.

    `distorted_uvs[image_id][splat_idx] = (u, v)` is an optional precomputed
    mapping. When None, we compute uvs on the fly via project_point_distorted.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    means_np = splats["means"].detach().cpu().numpy().astype(np.float64)
    colors_np = (
        (splats["colors"].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    )
    N = means_np.shape[0]

    # Pass 1: per-image, collect (splat_idx, xy_distorted) for visible splats.
    per_image_pairs: dict[int, list[tuple[int, np.ndarray]]] = {}
    splat_track_count = np.zeros(N, dtype=np.int64)
    for image_id, mask_t in visibility.items():
        if image_id not in reconstruction.images:
            continue
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        mask = mask_t.numpy().astype(bool)
        visible_idx = np.nonzero(mask)[0]
        pairs: list[tuple[int, np.ndarray]] = []
        for j in visible_idx:
            j_int = int(j)
            if distorted_uvs is not None and j_int in distorted_uvs.get(image_id, {}):
                u, v = distorted_uvs[image_id][j_int]
            else:
                uv = project_point_distorted(means_np[j_int], image, camera)
                if uv is None:
                    continue
                u, v = uv
            pairs.append((j_int, np.array([u, v], dtype=np.float64)))
            splat_track_count[j_int] += 1
        per_image_pairs[image_id] = pairs

    # Pass 2: decide which splats survive min_track_len.
    survives = splat_track_count >= int(min_track_len)
    n_survivors = int(survives.sum())
    if n_survivors == 0:
        raise RuntimeError(
            f"0 splats survived the min_track_len={min_track_len} filter; "
            "cannot export a splat-seeded COLMAP reconstruction."
        )

    # Pass 3: filter per-image pairs to keep only surviving splats. Build the
    # per-image local point2D index AND the per-splat observation list in one
    # walk so they stay consistent.
    final_pairs: dict[int, list[tuple[int, np.ndarray]]] = {}
    splat_observations: dict[int, list[tuple[int, int]]] = {
        j: [] for j in range(N) if survives[j]
    }
    for image_id, pairs in per_image_pairs.items():
        kept = [(j, xy) for (j, xy) in pairs if survives[j]]
        final_pairs[image_id] = kept
        for local_idx, (j, _xy) in enumerate(kept):
            splat_observations[j].append((image_id, local_idx))

    # Pass 4: clear old points3D. This also resets each old Point2D's
    # point3D_id to "invalid", but we'll overwrite all points2D below anyway.
    for pid in list(reconstruction.points3D.keys()):
        reconstruction.delete_point3D(pid)

    # Pass 5: install the new points2D lists on each image, with point3D_id
    # left at its invalid default. We MUST do this before adding point3Ds,
    # because Reconstruction.add_point3D walks the track and calls
    # image.SetPoint3DForPoint2D(point2D_idx, new_pid) for each element —
    # which requires that point2D_idx already exist on the image AND that
    # the point is not yet associated with a point3D.
    total_obs = 0
    for image_id, image in reconstruction.images.items():
        kept = final_pairs.get(image_id, [])
        p2d_list = [_make_point2d_no_pid(xy) for _j, xy in kept]
        _set_image_points2D(image, p2d_list)
        total_obs += len(p2d_list)

    # Pass 6: add point3Ds with their tracks. add_point3D will propagate the
    # new pid back into each referenced image's Point2D.point3D_id field.
    track_lens: list[int] = []
    for j in range(N):
        if not survives[j]:
            continue
        observations = splat_observations[j]
        track = _make_track(observations)
        xyz = means_np[j]
        rgb = colors_np[j]
        pid = int(reconstruction.add_point3D(xyz, track, rgb))
        track_lens.append(len(observations))
        # Set the initial error if the underlying object exposes it.
        try:
            reconstruction.points3D[pid].error = float(point_error_init)
        except (AttributeError, KeyError):
            pass

    reconstruction.write(out_dir.as_posix())

    mean_track_len = float(np.mean(track_lens)) if track_lens else 0.0
    metrics: dict[str, int | float] = {
        "num_input_splats": int(N),
        "num_points_after_track_filter": int(n_survivors),
        "num_observations": int(total_obs),
        "mean_track_length": mean_track_len,
        "num_images": int(len(reconstruction.images)),
    }
    logger.info("Splat→COLMAP export metrics: %s", metrics)

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        _write_track_length_csv(debug_dir / "track_length_histogram.csv", track_lens)
        _write_per_image_stats_csv(
            debug_dir / "per_image_stats.csv",
            reconstruction=reconstruction,
            visibility=visibility,
            per_image_pairs=per_image_pairs,
            num_splats=N,
        )

    return metrics


def _write_track_length_csv(path: Path, track_lens: list[int]) -> None:
    counts: dict[int, int] = {}
    for length in track_lens:
        counts[length] = counts.get(length, 0) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_len", "count"])
        for length in sorted(counts.keys()):
            writer.writerow([length, counts[length]])


def _write_per_image_stats_csv(
    path: Path,
    *,
    reconstruction: pycolmap.Reconstruction,
    visibility: dict[int, torch.Tensor],
    per_image_pairs: dict[int, list[tuple[int, np.ndarray]]],
    num_splats: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_id", "image_name", "num_visible", "num_in_frustum", "num_total"]
        )
        for image_id in sorted(reconstruction.images.keys()):
            image = reconstruction.images[image_id]
            num_visible = (
                int(visibility[image_id].sum().item()) if image_id in visibility else 0
            )
            # We don't carry an in-frustum count separately here; approximate
            # by len(per_image_pairs) which counts visible after distorted
            # projection. (Better than nothing for the diagnostic CSV.)
            num_in_frustum = len(per_image_pairs.get(image_id, []))
            writer.writerow(
                [image_id, image.name, num_visible, num_in_frustum, num_splats]
            )


def export_colmap_with_splat_visibility(
    *,
    splats: Splats,
    reconstruction: pycolmap.Reconstruction,
    scene_scale: float,
    out_dir: Path,
    depth_tol_relative: float = 0.01,
    depth_downsample: int = 2,
    min_track_len: int = 2,
    point_error_init: float = 1.0,
    device: str | torch.device = "cuda",
    debug_dir: Path | None = None,
) -> dict[str, int | float]:
    """Core exporter. Assumes splats are already loaded and filtered.

    Writes out_dir/{cameras.bin, images.bin, points3D.bin}. cameras and images
    metadata (id, camera_id, qvec, tvec, name) are preserved byte-identically
    from the source reconstruction; only points2D is replaced. points3D is
    replaced with one entry per surviving splat.
    """
    out_dir = Path(out_dir)

    cam_undist = build_camera_undistortion(reconstruction)
    image_ids, viewmats = build_world_to_undist_viewmats(reconstruction)

    K_undist_per_image: list[np.ndarray] = []
    width_per_image: list[int] = []
    height_per_image: list[int] = []
    for image_id in image_ids:
        camera_id = reconstruction.images[image_id].camera_id
        u = cam_undist[camera_id]
        K_undist_per_image.append(u.K_undist)
        width_per_image.append(u.width)
        height_per_image.append(u.height)

    # Move splats to render device.
    dev = torch.device(device)
    splats_dev: Splats = Splats(
        means=splats["means"].to(dev),
        quats=splats["quats"].to(dev),
        scales=splats["scales"].to(dev),
        opacities=splats["opacities"].to(dev),
        colors=splats["colors"].to(dev),
    )

    debug_callback = None
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        # Build a parallel lookup from image_id → (name, W, H) for overlay
        # writing inside the callback.
        meta_by_id: dict[int, tuple[str, int, int]] = {}
        for image_id, w, h in zip(image_ids, width_per_image, height_per_image):
            meta_by_id[image_id] = (reconstruction.images[image_id].name, w, h)

        def _cb(image_id, depth_map, status, uv_full):
            name, w_full, h_full = meta_by_id[image_id]
            stem = Path(name).stem
            out_png = debug_dir / f"{stem}.overlay.png"
            _write_overlay_png(out_png, depth_map, status, uv_full, w_full, h_full)

        debug_callback = _cb

    visibility = compute_visibility(
        splats=splats_dev,
        viewmats=viewmats,
        image_ids=image_ids,
        K_undist_per_image=K_undist_per_image,
        width_per_image=width_per_image,
        height_per_image=height_per_image,
        depth_tol_relative=depth_tol_relative,
        depth_downsample=depth_downsample,
        device=dev,
        debug_callback=debug_callback,
    )

    return _assemble_and_write(
        splats=splats_dev,
        reconstruction=reconstruction,
        visibility=visibility,
        out_dir=out_dir,
        min_track_len=min_track_len,
        point_error_init=point_error_init,
        debug_dir=debug_dir,
    )


def export_colmap_with_splat_visibility_from_disk(
    splat_ply_path: Path,
    source_colmap_dir: Path,
    out_dir: Path,
    *,
    min_opacity: float = 0.1,
    max_scale_frac_of_scene: float = 0.05,
    depth_tol_relative: float = 0.01,
    depth_downsample: int = 2,
    min_track_len: int = 2,
    point_error_init: float = 1.0,
    device: str | torch.device = "cuda",
    debug_dir: Path | None = None,
) -> dict[str, int | float]:
    """Disk wrapper: load COLMAP + PLY from disk, compute scene_scale, filter
    splats, run the core exporter, return metrics.
    """
    splat_ply_path = Path(splat_ply_path)
    source_colmap_dir = Path(source_colmap_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading source COLMAP reconstruction from %s", source_colmap_dir)
    reconstruction = pycolmap.Reconstruction(source_colmap_dir.as_posix())
    logger.info(
        "Source reconstruction: %d images, %d cameras, %d points3D",
        len(reconstruction.images),
        len(reconstruction.cameras),
        len(reconstruction.points3D),
    )

    scene_scale = compute_scene_scale(reconstruction)
    logger.info("Scene scale: %.6f", scene_scale)

    splats = load_and_filter_splats_from_ply(
        splat_ply_path,
        min_opacity=min_opacity,
        max_scale_frac_of_scene=max_scale_frac_of_scene,
        scene_scale=scene_scale,
        device="cpu",  # move to render device inside the core function
    )

    return export_colmap_with_splat_visibility(
        splats=splats,
        reconstruction=reconstruction,
        scene_scale=scene_scale,
        out_dir=out_dir,
        depth_tol_relative=depth_tol_relative,
        depth_downsample=depth_downsample,
        min_track_len=min_track_len,
        point_error_init=point_error_init,
        device=device,
        debug_dir=debug_dir,
    )
