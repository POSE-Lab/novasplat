import logging
from pathlib import Path

import torch
from gsplat.rendering import rasterization

logger = logging.getLogger(__name__)


def _infer_sh_degree(splats: dict[str, torch.Tensor]) -> int:
    if "shN" not in splats:
        return int((splats["sh0"].shape[1]) ** 0.5) - 1
    return int((splats["sh0"].shape[1] + splats["shN"].shape[1]) ** 0.5) - 1


def load_splats_checkpoint(
    ckpt_path: Path, device: torch.device
) -> dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "splats" not in ckpt:
        raise KeyError(f"Missing 'splats' in checkpoint: {ckpt_path}")
    required = ["means", "quats", "scales", "opacities", "sh0"]
    for key in required:
        if key not in ckpt["splats"]:
            raise KeyError(f"Missing '{key}' in checkpoint splats: {ckpt_path}")

    splats = {k: v.detach().to(device) for k, v in ckpt["splats"].items()}
    if "shN" not in splats:
        n = splats["sh0"].shape[0]
        splats["shN"] = torch.zeros((n, 0, 3), device=device, dtype=splats["sh0"].dtype)
    return splats


def render_rgb_alpha(
    splats: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb, alpha, _ = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
        viewmats=world_to_camera.unsqueeze(0),
        Ks=intrinsics.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=_infer_sh_degree(splats),
        rasterize_mode="classic",
        render_mode="RGB",
    )
    # rgb: [1, H, W, 3], alpha: [1, H, W, 1]
    return rgb, alpha


def render_rgb_alpha_depth(
    splats: dict[str, torch.Tensor],
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb_depth, alpha, _ = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
        viewmats=world_to_camera.unsqueeze(0),
        Ks=intrinsics.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=_infer_sh_degree(splats),
        rasterize_mode="classic",
        render_mode="RGB+ED",
    )
    # rgb_depth: [1, H, W, 4]; depth is the last channel (expected z in splat units)
    rgb = rgb_depth[..., :3]
    depth = rgb_depth[0, :, :, 3]  # [H, W]
    return rgb, alpha, depth
