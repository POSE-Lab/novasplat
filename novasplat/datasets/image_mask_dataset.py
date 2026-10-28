"""Dataset for images and optional masks."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def rle_decode(rle):
    """
    rle: dict with 'counts' and 'size' (height, width)
    """
    counts = rle['counts']
    h, w = rle['size']
    
    mask_1d = np.zeros(h * w, dtype=np.uint8)
    
    current_val = 0
    current_pos = 0
    for count in counts:
        if current_val == 1:
            mask_1d[current_pos : current_pos + count] = 255
        current_pos += count
        current_val = 1 - current_val
        
    mask = mask_1d.reshape((w, h)).T
    return mask

def crop_to_mask_bbox(
    image: Image.Image, mask: Image.Image, depth: Optional[Image.Image]
) -> tuple[Image.Image, Image.Image, Optional[Image.Image], np.ndarray]:
    """
    Crop image, mask, and optional depth to the tight bounding box of the mask.

    The returned bounding box follows the PIL ``Image.crop`` convention:
    ``(left, upper, right, lower)`` in pixel coordinates, where ``right`` and
    ``lower`` are exclusive (i.e. the box is ``[x_min, y_min, x_max_exclusive, y_max_exclusive]``).

    Returns
    -------
    image_cropped : PIL.Image.Image
        Cropped RGB image.
    mask_cropped : PIL.Image.Image
        Cropped mask image.
    depth_cropped : Optional[PIL.Image.Image]
        Cropped depth image if provided, otherwise ``None``.
    bbox_xyxy : numpy.ndarray
        1D array of 4 ints: ``[x_min, y_min, x_max_exclusive, y_max_exclusive]``.
    """
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        raise ValueError("Mask is completely empty; cannot compute bounding box.")

    bbox_xyxy = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=int)

    image_cropped = image.crop(tuple(bbox_xyxy))
    mask_cropped = mask.crop(tuple(bbox_xyxy))

    depth_cropped = depth.crop(tuple(bbox_xyxy)) if depth is not None else None

    return image_cropped, mask_cropped, depth_cropped, bbox_xyxy


class ImageMaskDepthDataset(Dataset):
    """
    Dataset supporting:
    - Directory scanning mode
    - Explicit sample list mode (recommended for BOP-style usage)
    """

    def __init__(
        self,
        image_dir: Optional[Path] = None,
        mask_dir: Optional[Path] = None,
        depth_dir: Optional[Path] = None,
        camera_params_path: Optional[Path] = None,
        samples: Optional[List[Dict[str, Any]]] = None,
    ):
        self.samples = []
        self.camera_params_path = camera_params_path

        # -------------------------------------------------
        # MODE 1: Explicit sample list
        # -------------------------------------------------
        if samples is not None:
            if not samples:
                raise ValueError("Provided samples list is empty.")
            self.samples = samples
            return

        # -------------------------------------------------
        # MODE 2: Directory scanning (legacy)
        # -------------------------------------------------
        if image_dir is None:
            raise ValueError("Either 'samples' or 'image_dir' must be provided.")

        image_paths = []
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            image_paths.extend(Path(image_dir).glob(pattern))
        image_paths = sorted(image_paths)

        if not image_paths:
            raise ValueError(f"No images found in directory: {image_dir}")

        mask_paths = None
        if mask_dir:
            mask_paths = []
            for pattern in ("*.png", "*.jpg", "*.jpeg"):
                mask_paths.extend(Path(mask_dir).glob(pattern))
            mask_paths = sorted(mask_paths)

            if len(image_paths) != len(mask_paths):
                raise ValueError("Number of images and masks do not match.")

        depth_paths = None
        if depth_dir:
            depth_paths = sorted(Path(depth_dir).glob("*.npz"))

            if len(image_paths) != len(depth_paths):
                raise ValueError("Number of images and depths do not match.")

        for i, img_path in enumerate(image_paths):
            self.samples.append(
                {
                    "image_path": img_path,
                    "mask_path": mask_paths[i] if mask_paths else None,
                    "depth_path": depth_paths[i] if depth_paths else None,
                    "pose": None,
                    "intrinsics": None,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        if sample.get("mask_path"):
            mask = Image.open(sample["mask_path"]).convert("L")
        elif sample.get("segmentation"):
            mask = Image.fromarray(rle_decode(sample["segmentation"])).convert("L")
        else:
            mask = Image.new("L", image.size, 255)

        depth = None
        depth_path = sample.get("depth_path")

        if depth_path is not None:
            depth_path = Path(depth_path)

            suffix = depth_path.suffix.lower()

            if suffix == ".npz":
                with np.load(depth_path, allow_pickle=True) as depth_npz:
                    if "depth" not in depth_npz:
                        raise ValueError(f"'depth' key not found in {depth_path}")
                    depth_array = depth_npz["depth"]

            elif suffix in [".png", ".tiff", ".tif"]:
                # Typical BOP depth (uint16 PNG)
                depth_array = np.array(Image.open(depth_path))

            else:
                raise ValueError(f"Unsupported depth format: {suffix}")

            depth_array = np.asarray(depth_array).squeeze()

            if depth_array.ndim != 2:
                raise ValueError(
                    f"Invalid depth shape {depth_array.shape} from {depth_path}"
                )

            # Convert to float32 for consistency
            depth_array = depth_array.astype(np.float32)

            depth = Image.fromarray(depth_array, mode="F")

        pose = sample.get("pose")
        intrinsics = sample.get("intrinsics")

        # Directory mode fallback
        if self.camera_params_path and pose is None and intrinsics is None:
            if not hasattr(self, "_camera_params_loaded"):
                with np.load(
                    self.camera_params_path, allow_pickle=True
                ) as camera_params:
                    self._camera_poses = camera_params["poses"]
                    self._camera_intrinsics = camera_params["intrinsics"]
                self._camera_params_loaded = True
            pose = self._camera_poses[idx]
            intrinsics = self._camera_intrinsics

        image, mask, depth, bbox = crop_to_mask_bbox(image, mask, depth)

        return {
            "image": image,
            "mask": mask,
            "image_path": sample["image_path"],
            "depth": depth,
            "pose": pose,
            "intrinsics": intrinsics,
            "bbox": bbox,
        }
