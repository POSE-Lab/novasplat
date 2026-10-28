"""Masking utilities."""

import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_mask_to_image(image_path: Path, mask_path: Path, output_path: Path) -> bool:
    """
    Apply a binary mask to an image.

    Args:
        image_path: Path to the input image
        mask_path: Path to the binary mask (should be grayscale)
        output_path: Path where the masked image will be saved

    Returns:
        bool: True if successful, False otherwise
    """
    if not mask_path.exists():
        logger.warning("No mask found for %s", image_path)
        return False
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error("Error: Could not load image %s", image_path)
            return False

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error("Error: Could not load mask %s", mask_path)
            return False

        if mask.shape[:2] != image.shape[:2]:
            logger.error(
                "Error: Image and mask have different dimensions. "
                "Image: %s, Mask: %s",
                image.shape[:2],
                mask.shape[:2],
            )
            return False

        binary_mask = (mask > 127).astype(np.uint8)
        masked_image = image * binary_mask[:, :, np.newaxis]
        cv2.imwrite(str(output_path), masked_image)
        return True

    except (OSError, ValueError, cv2.error) as e:
        logger.error("Error processing %s: %s", image_path, e)
        return False


def apply_masks_to_images(
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    image_suffix: str = ".jpg",
    mask_suffix: str = ".png",
) -> None:
    """Apply binary masks to images in parallel.

    Args:
        images_dir: The directory containing the images.
        masks_dir: The directory containing the masks.
        output_dir: The directory to save the masked images.
        image_suffix: The suffix of the image files.
        mask_suffix: The suffix of the mask files.
    """
    logger.info("Applying masks to images")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob(f"*{image_suffix}"))

    if not image_files:
        logger.error(
            "No image files found in %s with suffix %s", images_dir, image_suffix
        )
        return

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_image = {}
        for image_path in image_files:
            mask_path = masks_dir / f"{image_path.name}{mask_suffix}"
            output_path = output_dir / image_path.name

            future = executor.submit(
                apply_mask_to_image, image_path, mask_path, output_path
            )
            future_to_image[future] = image_path

        for future in as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                success = future.result()
                if success:
                    logger.debug("Successfully processed %s", image_path)
            except (OSError, ValueError, cv2.error) as e:
                logger.error("Error processing %s: %s", image_path, e)


def calculate_bbox_from_mask(mask_pixels, white_value=255):
    """
    Calculates the bounding box containing all white pixels of a binary mask.

    Args:
        mask_pixels (ndarray): Mask's pixels
        white_value (int): Value considered as white (default=255)


    Returns:
        list: [x, y, w, h] where (x,y) is the top left corner
        box or None if no white pixels are found
    """
    coords = np.where(mask_pixels == white_value)
    if coords[0].size == 0:
        return None
    min_row, min_col = coords[0].min(), coords[1].min()
    max_row, max_col = coords[0].max(), coords[1].max()

    return [min_col, min_row, max_col - min_col, max_row - min_row]
