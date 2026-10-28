"""Utility functions for image transformations."""

from PIL import Image

import torch
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_to_patches(
    pil_images: list[Image.Image],
    image_size: int,
    patch_size: int,
    interpolation_mode: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """Resize images so that their dimensions are multiples of patch_size."""

    if not pil_images:
        raise ValueError("resize_to_patches: input image list is empty.")
    if patch_size <= 0:
        raise ValueError("resize_to_patches: patch_size must be > 0.")
    if image_size <= patch_size:
        raise ValueError(
            "resize_to_patches: image_size must be greater than patch_size."
        )

    resized = []
    for img in pil_images:
        w, h = img.size
        n_patches_h = image_size // patch_size
        n_patches_w = (w * image_size) // (h * patch_size)
        new_h = n_patches_h * patch_size
        new_w = n_patches_w * patch_size

        resized.append(
            TF.to_tensor(
                TF.resize(img, (new_h, new_w), interpolation=interpolation_mode)
            )
        )

    # --- pad to max size in batch ---
    max_h = max(t.shape[1] for t in resized)
    max_w = max(t.shape[2] for t in resized)

    padded = []
    for t in resized:
        pad_h = max_h - t.shape[1]
        pad_w = max_w - t.shape[2]

        # pad: (left, right, top, bottom)
        padded.append(
            torch.nn.functional.pad(
                t,
                (0, pad_w, 0, pad_h),
                value=0.0,
            )
        )

    return torch.stack(padded)


def inv_resize_to_patches(original_size: tuple, image_size: tuple, patch_size: int):
    """
    This function resizes the original image to a new size, pads it, and maps the coordinates
    from the resized image back to the original image.

    Args:
    - original_size (tuple): The size of the original image (width, height).
    - image_size (tuple): The target image size (height, width).
    - patch_size (int): The size of the patches.

    Returns:
    - u (torch.Tensor): The coordinates for the height dimension in the resized image.
    - v (torch.Tensor): The coordinates for the width dimension in the resized image.
    """
    # Given values
    w, h = original_size

    if isinstance(image_size, tuple):
        image_size_h, image_size_w = image_size
    elif isinstance(image_size, int):  # If only height is provided
        image_size_h = image_size
        image_size_w = None  # Width not provided, it will be calculated
    else:
        raise ValueError(
            "image_size must be a tuple (height, width) or a single integer"
        )

    # Step 1: Calculate the number of patches and new image dimensions
    n_patches_h = image_size_h // patch_size
    n_patches_w = (w * image_size_h) // (h * patch_size)

    # Calculate new image dimensions (must be multiples of patch_size)
    new_h = n_patches_h * patch_size
    new_w = n_patches_w * patch_size

    # Step 2: Calculate u, v (coordinates in resized image corresponding to original image)
    scale_h = new_h / h
    scale_w = new_w / w

    # Generate grid of coordinates for the resized image
    grid_h, grid_w = torch.meshgrid(
        torch.arange(new_h), torch.arange(new_w), indexing="ij"
    )

    # Normalize the grid to [0, 1]
    norm_grid_h = grid_h.float() / new_h
    norm_grid_w = grid_w.float() / new_w

    # Map the normalized grid back to the original image coordinates
    u = norm_grid_h * h
    v = norm_grid_w * w

    # Padding
    if image_size_w is not None:
        pad_w = image_size_w - new_w

        # Padding for u (height)
        u_padding = u[:, 0].unsqueeze(1).repeat(1, pad_w)
        u = torch.cat((u, u_padding), dim=1)

        # Padding for v (width) with consistent step
        v_step = v[0, 1] - v[0, 0]
        v_padding = v[:, -1].unsqueeze(1) + v_step * torch.arange(
            1, pad_w + 1
        ).unsqueeze(0)
        v = torch.cat((v, v_padding), dim=1)

    # Ensure coordinates are within the bounds of the original image (optional, if you want bounds check)
    # u = torch.clamp(u, 0, h - 1)
    # v = torch.clamp(v, 0, w - 1)

    return u, v


def normalize_images(x: torch.Tensor) -> torch.Tensor:
    """Normalize images using ImageNet mean and std.

    Accepts either a single image tensor (C, H, W) or a batch (N, C, H, W).
    If the number of channels is 1 (grayscale), the channel is replicated to
    create a 3-channel tensor. If it is 4 (e.g. RGBA), the alpha channel is
    discarded and the first 3 channels are used. Any other channel count
    results in a ValueError.
    """
    # Support both single-image (C, H, W) and batched (N, C, H, W) inputs.
    squeeze_batch = False
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze_batch = True
    elif x.dim() != 4:
        raise ValueError(
            f"normalize_images expects a 3D or 4D tensor, got shape {tuple(x.shape)}"
        )

    # x is now (N, C, H, W)
    n, c, h, w = x.shape

    if c == 1:
        # Grayscale: replicate channel to create RGB
        x = x.repeat(1, 3, 1, 1)
    elif c == 4:
        # RGBA: drop alpha channel
        x = x[:, :3, :, :]
    elif c != 3:
        raise ValueError(
            f"normalize_images only supports 1, 3, or 4 channel images, got {c} channels"
        )

    x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)

    if squeeze_batch:
        x = x.squeeze(0)

    return x
