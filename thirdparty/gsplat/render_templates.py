# Copyright (c) 2026 POSE-Lab, National Technical University of Athens
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import torch

from gsplat.rendering import rasterization

logger = logging.getLogger(__name__)


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Load gsplat checkpoint and return the splats parameters.

    Args:
        ckpt_path: The path to the checkpoint.
        device: The device to use for rendering.

    Returns:
        The splats parameters.
    """
    logger.info("Loading checkpoint from: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    return {key: value.to(device) for key, value in ckpt["splats"].items()}


def render_templates(
    output_dir: Path,
    splats: dict[str, torch.Tensor],
    poses: list[np.ndarray],
    intrinsics: np.ndarray,
    width: int,
    height: int,
    sh_degree: int = 3,
    device: str = "cuda",
) -> None:
    """
    Render templates from the given camera poses.

    Args:
        output_dir: The directory to save the rendered templates.
        splats: The splats parameters.
        poses: The camera poses.
        intrinsics: The intrinsic matrix.
        width: The width of the image.
        height: The height of the image.
        sh_degree: The degree of the spherical harmonics.
        device: The device to use for rendering.
    """
    rgb_dir = output_dir / "rgb"
    mask_dir = output_dir / "mask_visib"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    intrinsics_tensor = (
        torch.from_numpy(intrinsics.astype(np.float32)).unsqueeze(0).to(device)
    )

    for i, pose in tqdm(enumerate(poses), desc="Rendering templates", total=len(poses)):
        pose_tensor = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0).to(device)
        rgb, _, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
            viewmats=pose_tensor,
            Ks=intrinsics_tensor,
            width=width,
            height=height,
            packed=False,
            sh_degree=sh_degree,
            near_plane=0.01,
            far_plane=5000.0,
            rasterize_mode="classic",
            render_mode="RGB",
        )
        rgb_image = rgb.squeeze(0).cpu().numpy()

        rgb_clipped = np.clip(rgb_image, 0.0, 1.0)
        rgb_uint8 = (rgb_clipped * 255).astype(np.uint8)
        bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(rgb_dir / f"rgb_{i:04d}.png"), bgr_uint8)

        mask = np.all(rgb_uint8 == [0, 0, 0], axis=-1)
        mask_uint8 = ((~mask) * 255).astype(np.uint8)
        cv2.imwrite(str(mask_dir / f"mask_{i:04d}.png"), mask_uint8)