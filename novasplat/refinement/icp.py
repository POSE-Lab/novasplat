"""Point-cloud ICP refinement using Open3D."""

import logging
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import open3d as o3d
import torch

from novasplat.datasets.image_mask_dataset import rle_decode

logger = logging.getLogger(__name__)


def run_icp(
    model_pts_obj: "torch.Tensor | np.ndarray",
    query_pts_cam: "torch.Tensor | np.ndarray",
    init_rotation: "torch.Tensor | np.ndarray",
    init_translation: "torch.Tensor | np.ndarray",
    *,
    max_iterations: int = 50,
    max_correspondence_dist_mm: float = 50.0,
    voxel_size_mm: float = 3.0,
    method: Literal["point_to_point", "point_to_plane"] = "point_to_point",
) -> tuple[np.ndarray, np.ndarray, bool]:
    """ICP: source (model.ply in object frame) → target (scene depth in camera frame).

    Args:
        model_pts_obj: [N, 3] model points in object frame (mm).
        query_pts_cam: [M, 3] scene depth points in camera frame (mm).
        init_rotation: [3, 3] initial rotation matrix (object→camera).
        init_translation: [3,] initial translation vector (mm).
        max_iterations: maximum ICP iterations.
        max_correspondence_dist_mm: per-correspondence rejection threshold (mm).
        voxel_size_mm: voxel size for downsampling before ICP; 0 to disable.
        method: "point_to_point" or "point_to_plane".

    Returns:
        rotation: [3, 3] refined rotation matrix (float32).
        translation: [3,] refined translation vector (mm, float32).
        converged: True when ICP found at least one correspondence.
    """
    if isinstance(model_pts_obj, torch.Tensor):
        model_pts_obj = model_pts_obj.detach().cpu().numpy()
    if isinstance(query_pts_cam, torch.Tensor):
        query_pts_cam = query_pts_cam.detach().cpu().numpy()
    if isinstance(init_rotation, torch.Tensor):
        init_rotation = init_rotation.detach().cpu().numpy()
    if isinstance(init_translation, torch.Tensor):
        init_translation = init_translation.detach().cpu().numpy()

    model_pts_obj = np.asarray(model_pts_obj, dtype=np.float64)
    query_pts_cam = np.asarray(query_pts_cam, dtype=np.float64)

    init_T = np.eye(4, dtype=np.float64)
    init_T[:3, :3] = np.asarray(init_rotation, dtype=np.float64)
    init_T[:3, 3] = np.asarray(init_translation, dtype=np.float64)

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(model_pts_obj)

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(query_pts_cam)

    if voxel_size_mm > 0:
        src_pcd = src_pcd.voxel_down_sample(voxel_size=voxel_size_mm)
        tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size=voxel_size_mm)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations,
    )

    if method == "point_to_plane":
        tgt_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max_correspondence_dist_mm * 2,
                max_nn=30,
            )
        )
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    try:
        result = o3d.pipelines.registration.registration_icp(
            src_pcd,
            tgt_pcd,
            max_correspondence_distance=max_correspondence_dist_mm,
            init=init_T,
            estimation_method=estimation,
            criteria=criteria,
        )
    except Exception as exc:
        logger.warning("ICP failed: %s", exc)
        return (
            np.asarray(init_rotation, dtype=np.float32),
            np.asarray(init_translation, dtype=np.float32),
            False,
        )

    converged = result.fitness > 0.0
    if converged:
        logger.debug(
            "ICP: fitness=%.4f rmse=%.2f mm n_corr=%d",
            result.fitness,
            result.inlier_rmse,
            len(result.correspondence_set),
        )
    else:
        logger.warning("ICP: no correspondences found (fitness=0).")

    T = np.array(result.transformation)
    return (
        T[:3, :3].astype(np.float32),
        T[:3, 3].astype(np.float32),
        converged,
    )


class ICPRefiner:
    """Drop-in replacement for Refiner that uses point-cloud ICP instead of photometric optimisation.

    Requires:
      - ``template_root / obj_XXXXXX / model.ply`` (BOP model point cloud).
      - A scene depth image reachable via ``depth_path``.

    The query point cloud is built by back-projecting the scene depth under the
    provided detection mask (mask_path or CNOS segmentation RLE).  ICP then
    aligns the model points (in object frame) to those query points.
    """

    def __init__(
        self,
        *,
        template_root: Path,
        device: torch.device,
        max_iterations: int = 50,
        max_correspondence_dist_mm: float = 50.0,
        voxel_size_mm: float = 3.0,
        method: Literal["point_to_point", "point_to_plane"] = "point_to_point",
    ) -> None:
        self.template_root = template_root
        self.device = device
        self.max_iterations = max_iterations
        self.max_correspondence_dist_mm = max_correspondence_dist_mm
        self.voxel_size_mm = voxel_size_mm
        self.method: Literal["point_to_point", "point_to_plane"] = method

        self._model_pts_cache: dict[int, Any] = {}

    def _load_model_pts(self, obj_id: int) -> Any:
        cached = self._model_pts_cache.get(obj_id)
        if cached is not None:
            return cached
        ply_path = self.template_root / f"obj_{obj_id:06d}" / "model.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"model.ply not found for obj {obj_id}: {ply_path}")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points, dtype=np.float64)
        self._model_pts_cache[obj_id] = pts
        return pts

    def _build_query_pts(
        self,
        depth_path: str | Path,
        mask: Any,
        intrinsics: torch.Tensor,
    ) -> Any:
        """Back-project masked depth pixels to camera-frame 3-D points (mm)."""
        import json as _json

        depth_path = Path(depth_path)
        if not depth_path.exists():
            return None

        depth_scale = 1.0
        scene_camera_path = depth_path.parent.parent / "scene_camera.json"
        if scene_camera_path.exists():
            with open(scene_camera_path, encoding="utf-8") as fh:
                scene_camera = _json.load(fh)
            im_id = int(depth_path.stem)
            depth_scale = float(
                scene_camera.get(str(im_id), {}).get("depth_scale", 1.0)
            )

        depth_raw: Any = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            return None
        depth_mm: Any = depth_raw.astype(np.float64) * depth_scale

        valid: Any = (mask > 0) & (depth_mm > 0)
        if valid.sum() == 0:
            return None

        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])

        ys, xs = np.where(valid)
        z = depth_mm[ys, xs]
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        return np.stack([x, y, z], axis=1)

    def refine(
        self,
        *,
        obj_id: int,
        coarse_rotation: torch.Tensor,
        coarse_translation: torch.Tensor,
        coarse_score: float,
        intrinsics: torch.Tensor,
        image_path: str,
        mask_path: str | Path | None,
        segmentation: dict[str, Any] | None,
        seed: int,
        depth_path: str | Path | None = None,
        sample_key: str | None = None,
    ) -> dict[str, object]:
        if depth_path is None:
            logger.warning(
                "ICPRefiner: no depth_path for obj %d — returning coarse pose.", obj_id
            )
            return {
                "rotation": coarse_rotation,
                "translation": coarse_translation,
                "refined_score": coarse_score,
                "refinement_successful": False,
                "failure_reason": "no_depth",
            }

        # Load detection mask.
        mask_np: Any = None
        if mask_path is not None:
            mask_np = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_np is None and segmentation is not None:
            mask_np = rle_decode(segmentation)
        if mask_np is None:
            logger.warning(
                "ICPRefiner: no mask available for obj %d — returning coarse pose.",
                obj_id,
            )
            return {
                "rotation": coarse_rotation,
                "translation": coarse_translation,
                "refined_score": coarse_score,
                "refinement_successful": False,
                "failure_reason": "no_mask",
            }

        try:
            model_pts = self._load_model_pts(obj_id)
        except FileNotFoundError as exc:
            logger.warning("ICPRefiner: %s — returning coarse pose.", exc)
            return {
                "rotation": coarse_rotation,
                "translation": coarse_translation,
                "refined_score": coarse_score,
                "refinement_successful": False,
                "failure_reason": str(exc),
            }

        query_pts = self._build_query_pts(depth_path, mask_np, intrinsics)
        if query_pts is None or query_pts.shape[0] == 0:
            logger.warning(
                "ICPRefiner: empty query point cloud for obj %d — returning coarse pose.",
                obj_id,
            )
            return {
                "rotation": coarse_rotation,
                "translation": coarse_translation,
                "refined_score": coarse_score,
                "refinement_successful": False,
                "failure_reason": "empty_query_pts",
            }

        refined_R: Any
        refined_t: Any
        refined_R, refined_t, converged = run_icp(
            model_pts_obj=model_pts,
            query_pts_cam=query_pts,
            init_rotation=coarse_rotation,
            init_translation=coarse_translation,
            max_iterations=self.max_iterations,
            max_correspondence_dist_mm=self.max_correspondence_dist_mm,
            voxel_size_mm=self.voxel_size_mm,
            method=self.method,
        )

        refined_rotation = torch.from_numpy(refined_R).to(
            device=coarse_rotation.device, dtype=coarse_rotation.dtype
        )
        refined_translation = torch.from_numpy(refined_t).to(
            device=coarse_translation.device, dtype=coarse_translation.dtype
        )
        return {
            "rotation": refined_rotation,
            "translation": refined_translation,
            "refined_score": coarse_score,
            "refinement_successful": converged,
            "failure_reason": None if converged else "icp_no_correspondences",
        }
