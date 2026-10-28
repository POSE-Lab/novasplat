"""Mask operations."""

import torch
import torch.nn.functional as F


def pixel_mask_to_patch_mask(
    mask: torch.Tensor,
    patch_size: int,
):
    """
    Convert pixel-space mask to patch-space mask by average pooling.

    Args:
        mask: [B, 1, H, W], values in [0,1]
        patch_size: ViT patch size. `H` and `W` must be divisible by `patch_size`.

    Returns:
        patch_mask: [B, H//P, W//P]
    """
    h, w = mask.shape[-2], mask.shape[-1]
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"Mask height and width must be divisible by patch_size for pooling, "
            f"but got H={h}, W={w}, patch_size={patch_size}."
        )
    return F.avg_pool2d(mask, kernel_size=patch_size, stride=patch_size).squeeze(1)
