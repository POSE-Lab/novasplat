import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from novasplat.datasets.image_mask_dataset import rle_decode

from .optimizer import OptimizerConfig, optimize_pose
from .renderer import load_splats_checkpoint, render_rgb_alpha, render_rgb_alpha_depth

logger = logging.getLogger(__name__)


class Refiner:
    def __init__(
        self,
        *,
        result_dir: Path,
        device: torch.device,
        optimizer_config: OptimizerConfig | None = None,
        debug_dir: Path | None = None,
        template_root: Path | None = None,
    ):
        self.result_dir = result_dir
        self.device = device
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.debug_dir = debug_dir
        self.template_root = template_root

        self._splats_cache: dict[int, dict[str, torch.Tensor]] = {}
        self._model_pts_cache: dict[int, torch.Tensor] = {}

    def _build_w2c(
        self, rotation: torch.Tensor, translation: torch.Tensor
    ) -> torch.Tensor:
        bottom = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], dtype=rotation.dtype, device=rotation.device
        )
        return torch.cat(
            [torch.cat([rotation, translation.unsqueeze(1)], dim=1), bottom], dim=0
        )

    def _resolve_ckpt_path(self, obj_id: int) -> Path | None:
        obj_tag = f"obj_{obj_id:06d}"
        
        base_dirs = [self.result_dir]
        if getattr(self, 'template_root', None) is not None:
            base_dirs.append(self.template_root.parent)
        if getattr(self, 'data_dir', None) is not None:
            base_dirs.append(self.data_dir)
            
        for base_dir in base_dirs:
            for obj_dir_name in [obj_tag, str(obj_id)]:
                ckpt_dir = base_dir / "onboarding" / obj_dir_name / "splat" / "ckpts"
                if not ckpt_dir.exists():
                    continue
                ckpts = list(ckpt_dir.glob("ckpt_*_rank0.pt"))
                if not ckpts:
                    continue
                
                def get_iter(p: Path) -> int:
                    try:
                        return int(p.name.split("_")[1])
                    except (IndexError, ValueError):
                        return -1
                
                best_ckpt = max(ckpts, key=get_iter)
                return best_ckpt
        return None

    def _load_splats(self, obj_id: int) -> dict[str, torch.Tensor]:
        cached = self._splats_cache.get(obj_id)
        if cached is not None:
            return cached

        ckpt_path = self._resolve_ckpt_path(obj_id)
        if ckpt_path is None:
            raise FileNotFoundError(
                "Could not locate onboarding gsplat checkpoint required for refinement."
            )
        splats = load_splats_checkpoint(ckpt_path, device=self.device)
        self._splats_cache[obj_id] = splats
        return splats

    def _load_model_pts(self, obj_id: int) -> torch.Tensor | None:
        cached = self._model_pts_cache.get(obj_id)
        if cached is not None:
            return cached
        if self.template_root is None:
            return None
        ply_path = self.template_root / f"obj_{obj_id:06d}" / "model.ply"
        if not ply_path.exists():
            logger.warning("model.ply not found: %s", ply_path)
            return None
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)
        # Stride-based downsample to keep cdist tractable (BOP models are ~5K pts).
        max_pts = 2048
        if pts.shape[0] > max_pts:
            stride = pts.shape[0] // max_pts
            pts = pts[::stride][:max_pts]
        self._model_pts_cache[obj_id] = pts
        return pts

    def _render_initial_frame(
        self,
        splats: dict[str, torch.Tensor],
        rotation: torch.Tensor,
        translation: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (binary alpha mask [1,1,H,W], rendered depth [H,W]) at the coarse pose."""
        w2c = self._build_w2c(rotation, translation)
        with torch.no_grad():
            _, alpha, depth_hw = render_rgb_alpha_depth(
                splats=splats,
                world_to_camera=w2c,
                intrinsics=intrinsics,
                width=width,
                height=height,
            )
        alpha_bchw = alpha.permute(0, 3, 1, 2)
        mask_bchw = (alpha_bchw > self.optimizer_config.rendered_mask_threshold).to(
            torch.float32
        )
        return mask_bchw, depth_hw

    def _prepare_query_pts(
        self,
        depth_path: str | Path,
        intrinsics: torch.Tensor,
        rendered_mask_bchw: torch.Tensor,
        rendered_depth_hw: torch.Tensor,
        depth_band_mm: float = 30.0,
    ) -> torch.Tensor | None:
        depth_path = Path(depth_path)
        if not depth_path.exists():
            return None

        # Infer depth_scale (BOP stores it in scene_camera.json alongside cam_K).
        depth_scale = 1.0
        scene_camera_path = depth_path.parent.parent / "scene_camera.json"
        if scene_camera_path.exists():
            with open(scene_camera_path, encoding="utf-8") as f:
                scene_camera = json.load(f)
            im_id = int(depth_path.stem)
            depth_scale = float(
                scene_camera.get(str(im_id), {}).get("depth_scale", 1.0)
            )

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            return None
        depth_mm = torch.from_numpy(depth_raw.astype(np.float32) * depth_scale).to(
            self.device
        )

        # Keep only pixels that are:
        #   1. inside the initial rendered footprint
        #   2. have valid scene depth
        #   3. within depth_band_mm of the rendered object depth at that pixel
        # Condition 3 rejects table surface, adjacent objects, and background —
        # the z-exploit used the full scene depth range to reduce Chamfer by
        # shifting the object forward, which this band prevents.
        depth_diff = torch.abs(depth_mm - rendered_depth_hw)
        mask_2d = (
            (rendered_mask_bchw[0, 0] > 0)
            & (depth_mm > 0)
            & (depth_diff < depth_band_mm)
        )
        if mask_2d.sum() == 0:
            return None

        H, W = depth_mm.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        ys, xs = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        z = depth_mm[mask_2d]
        x = (xs[mask_2d] - cx) * z / fx
        y = (ys[mask_2d] - cy) * z / fy
        return torch.stack([x, y, z], dim=1)  # [M, 3] camera frame, mm

    def _load_target_image_mask(
        self,
        image_path: str,
        mask_path: str | Path | None,
        segmentation: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        image_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_np is None:
            raise FileNotFoundError(f"Could not read query image: {image_path}")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB).astype("float32") / 255.0

        mask_np = None
        if mask_path is not None:
            mask_np = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_np is None and segmentation is not None:
            mask_np = rle_decode(segmentation)
            if mask_np.shape[:2] != image_np.shape[:2]:
                raise ValueError(
                    "Decoded segmentation mask shape does not match image shape: "
                    f"mask={mask_np.shape}, image={image_np.shape}"
                )
        if mask_np is None:
            raise FileNotFoundError(
                "Could not obtain query mask from mask_path or segmentation. "
                f"mask_path={mask_path}, segmentation_present={segmentation is not None}"
            )
        mask_np = (mask_np.astype("float32") / 255.0)[..., None]

        h, w = image_np.shape[:2]
        image = torch.from_numpy(image_np).to(self.device)
        mask = torch.from_numpy(mask_np).to(self.device)
        image_bchw = image.permute(2, 0, 1).unsqueeze(0)
        mask_bchw = mask.permute(2, 0, 1).unsqueeze(0)
        return image_bchw, mask_bchw, w, h

    def _save_debug_renders(
        self,
        *,
        splats: dict[str, torch.Tensor],
        coarse_rotation: torch.Tensor,
        coarse_translation: torch.Tensor,
        target_image_bchw: torch.Tensor,
        query_mask_bchw: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
        out_dir: Path,
        tag: str,
    ) -> None:
        w2c = self._build_w2c(coarse_rotation, coarse_translation)
        with torch.no_grad():
            rgb, alpha = render_rgb_alpha(
                splats=splats,
                world_to_camera=w2c,
                intrinsics=intrinsics,
                width=width,
                height=height,
            )

        def _to_bgr(t_bchw: torch.Tensor) -> np.ndarray:
            arr = t_bchw[0].permute(1, 2, 0).cpu().float().numpy()
            return cv2.cvtColor(
                np.clip(arr * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )

        cv2.imwrite(
            str(out_dir / f"render_{tag}.png"), _to_bgr(rgb.permute(0, 3, 1, 2))
        )

        # Save target+mask composite once (first call, tag="initial")
        if tag == "initial":
            target_bgr = _to_bgr(target_image_bchw)
            mask_np = (query_mask_bchw[0, 0].cpu().numpy() * 255).astype(np.uint8)
            mask_overlay = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(out_dir / "target.png"), target_bgr)
            cv2.imwrite(str(out_dir / "mask.png"), mask_overlay)

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
        segmentation: dict | None,
        seed: int,
        depth_path: str | Path | None = None,
        sample_key: str | None = None,
    ) -> dict[str, object]:
        _t0 = time.perf_counter()
        target_image_bchw, query_mask_bchw, width, height = (
            self._load_target_image_mask(
                image_path=image_path,
                mask_path=mask_path,
                segmentation=segmentation,
            )
        )
        _t_image_load = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        splats = self._load_splats(obj_id)
        _t_splat_load = time.perf_counter() - _t0

        # Precompute 3D point clouds for geometric Chamfer loss (once per sample).
        model_pts_obj: torch.Tensor | None = None
        query_pts_cam: torch.Tensor | None = None
        _t_geo_prep = 0.0
        if self.optimizer_config.geo_weight > 0 and depth_path is not None:
            _t0 = time.perf_counter()
            model_pts_obj = self._load_model_pts(obj_id)
            if model_pts_obj is not None:
                initial_mask, initial_depth = self._render_initial_frame(
                    splats=splats,
                    rotation=coarse_rotation,
                    translation=coarse_translation,
                    intrinsics=intrinsics,
                    width=width,
                    height=height,
                )
                query_pts_cam = self._prepare_query_pts(
                    depth_path=depth_path,
                    intrinsics=intrinsics,
                    rendered_mask_bchw=initial_mask,
                    rendered_depth_hw=initial_depth,
                )
            _t_geo_prep = time.perf_counter() - _t0

        if self.optimizer_config.geo_weight > 0 and (
            model_pts_obj is None or query_pts_cam is None
        ):
            logger.warning(
                "Geo loss INACTIVE for obj %d: depth_path=%s, model_pts=%s, query_pts=%s",
                obj_id,
                depth_path,
                "ok" if model_pts_obj is not None else "None",
                "ok" if query_pts_cam is not None else "None",
            )

        sample_debug_dir: Path | None = None
        if self.debug_dir is not None:
            key = sample_key or f"seed{seed}"
            sample_debug_dir = self.debug_dir / key
            sample_debug_dir.mkdir(parents=True, exist_ok=True)
            self._save_debug_renders(
                splats=splats,
                coarse_rotation=coarse_rotation,
                coarse_translation=coarse_translation,
                target_image_bchw=target_image_bchw,
                query_mask_bchw=query_mask_bchw,
                intrinsics=intrinsics,
                width=width,
                height=height,
                out_dir=sample_debug_dir,
                tag="initial",
            )

        _t0 = time.perf_counter()
        opt_result = optimize_pose(
            splats=splats,
            coarse_rotation=coarse_rotation,
            translation=coarse_translation,
            target_image_bchw=target_image_bchw,
            query_mask_bchw=query_mask_bchw,
            intrinsics=intrinsics,
            width=width,
            height=height,
            config=self.optimizer_config,
            seed=seed,
            model_pts_obj=model_pts_obj,
            query_pts_cam=query_pts_cam,
            debug_dir=sample_debug_dir,
        )
        _t_optimize = time.perf_counter() - _t0
        _n_iters = opt_result.get("iterations", 0)
        logger.info(
            "Refinement breakdown — image_load: %.3f s | splat_load: %.3f s | "
            "geo_prep: %.3f s | optimize (%d iters): %.3f s "
            "[render: %.3f s | photo_loss: %.3f s | backward: %.3f s | step: %.3f s]",
            _t_image_load,
            _t_splat_load,
            _t_geo_prep,
            _n_iters,
            _t_optimize,
            opt_result.get("t_render", 0.0),
            opt_result.get("t_photo_loss", 0.0),
            opt_result.get("t_backward", 0.0),
            opt_result.get("t_step", 0.0),
        )

        if sample_debug_dir is not None:
            self._save_debug_renders(
                splats=splats,
                coarse_rotation=opt_result["rotation"],
                coarse_translation=opt_result["translation"],
                target_image_bchw=target_image_bchw,
                query_mask_bchw=query_mask_bchw,
                intrinsics=intrinsics,
                width=width,
                height=height,
                out_dir=sample_debug_dir,
                tag="final",
            )

        return {
            "rotation": opt_result["rotation"],
            "translation": opt_result["translation"],
            "refined_score": float(coarse_score),
            "refinement_successful": True,
            "failure_reason": None,
        }
