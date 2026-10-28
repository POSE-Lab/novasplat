import numpy as np
import open3d as o3d
import torch

try:
    from gedi import GeDi
except ModuleNotFoundError as exc:
    # Fallback for local checkouts where GeDi is provided in thirdparty/gedi.
    import sys
    from pathlib import Path

    gedi_root = Path(__file__).resolve().parents[2] / "thirdparty" / "gedi"
    if gedi_root.exists():
        sys.path.insert(0, str(gedi_root))
        from gedi import GeDi
    else:
        raise ModuleNotFoundError(
            "GeDi module not found. Expected either an importable 'gedi' package "
            "or local path 'thirdparty/gedi'."
        ) from exc


class GeDiExtractor:
    def __init__(self, config: dict):
        self.config = config
        self.model = GeDi(config)

    def extract(
        self,
        pcd: o3d.geometry.PointCloud,
        points: np.ndarray | None = None,
        precomputed_support: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract GeDi descriptors from a point cloud.

        Args:
            pcd: Open3D point cloud (support cloud).
            points: Optional keypoint positions as numpy array.
            precomputed_support: Optional pre-computed voxel-downsampled
                support tensor on CUDA. When provided, skips
                ``pcd.voxel_down_sample`` to avoid redundant work across
                multi-scale extraction.
        """
        # Note: estimate_normals() is intentionally skipped.
        # GeDi.compute() receives only positions, not normals.

        # If points are given, use them directly
        if points is not None:
            points_np = points
            points = torch.from_numpy(points_np).float().cuda()

        else:
            num_points = np.asarray(pcd.points).shape[0]
            num_samples = min(num_points, self.config["num_keypoints"])

            if num_samples == 0:
                raise ValueError(
                    "Point cloud is empty; cannot extract GeDi descriptors."
                )
            indices = np.linspace(0, num_points - 1, num_samples, dtype=np.int64)
            points = torch.from_numpy(np.asarray(pcd.points)[indices]).float().cuda()
            points_np = points.cpu().numpy()

        if precomputed_support is not None:
            pcd_torch = precomputed_support
        else:
            pcd_vox = pcd.voxel_down_sample(voxel_size=self.config["voxel_size"])
            pcd_torch = torch.from_numpy(np.asarray(pcd_vox.points)).float().cuda()

        descriptors = self.model.compute(points, pcd_torch)

        if isinstance(descriptors, torch.Tensor):
            descriptors = descriptors.cpu().numpy()

        return descriptors, points_np
