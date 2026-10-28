import logging
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import argparse

logger = logging.getLogger(__name__)


from novasplat.utils.transforms import inv_resize_to_patches
from novasplat.datasets.bop import BOPValDataset

PATCH_SIZE = 16


def plot_images_with_lines(image_1, image_2, points_1, points_2, save_path=None):
    """
    This function takes two PIL images (image_1, image_2) and two lists of corresponding points
    (points_1, points_2), then displays the images side by side with lines connecting the points.

    Args:
    - image_1 (PIL.Image): First image
    - image_2 (PIL.Image): Second image
    - points_1 (list of tuples): List of (u, v) coordinates for image_1
    - points_2 (list of tuples): List of (u, v) coordinates for image_2
    - save_path (Path | str | None): If given, save the figure to this path instead of showing it.
    """

    # Convert images to numpy arrays for easier manipulation
    img1_array = np.array(image_1)
    img2_array = np.array(image_2)

    # Get the image sizes
    img1_height, img1_width = img1_array.shape[:2]
    img2_height, img2_width = img2_array.shape[:2]

    # Create a blank canvas to combine both images
    combined_image = np.zeros(
        (max(img1_height, img2_height), img1_width + img2_width, 3), dtype=np.uint8
    )

    # Place image_1 on the left side of the canvas
    combined_image[:img1_height, :img1_width] = img1_array

    # Place image_2 on the right side of the canvas
    combined_image[:img2_height, img1_width:] = img2_array

    # Create a figure to plot the combined image
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(combined_image)
    ax.axis("off")  # Hide axes

    # Draw lines between corresponding points
    for (u1, v1), (u2, v2) in zip(points_1, points_2):
        ax.plot(u1, v1, "ro")  # Mark points in image 1 with red dots
        ax.plot(u2 + img1_width, v2, "ro")  # Mark points in image 2 with red dots

        # Draw a line between the points from image_1 to image_2 across the combined canvas
        ax.plot(
            [u1, u2 + img1_width], [v1, v2], color="yellow", linestyle="-", linewidth=2
        )

    # Display the plot
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def pca_feature_map_to_rgb(feat_a, feat_b, q=3):
    """
    Perform joint PCA on two feature maps and convert them to RGB images.

    Args:
        feat_a: torch.Tensor [C, H, W]
        feat_b: torch.Tensor [C, H, W]
        q: number of PCA components (default=3)

    Returns:
        img_a: numpy array [H, W, 3] (uint8)
        img_b: numpy array [H, W, 3] (uint8)
    """

    C, H1, W1 = feat_a.shape
    _, H2, W2 = feat_b.shape

    # Flatten → [N, C]
    desc_a_all = feat_a.reshape(C, -1).permute(1, 0)
    desc_b_all = feat_b.reshape(C, -1).permute(1, 0)

    # Joint PCA
    fit_feats = torch.cat([desc_a_all, desc_b_all], dim=0)

    # Center
    fit_feats = fit_feats - fit_feats.mean(dim=0, keepdim=True)

    # PCA via low-rank SVD
    _, _, v = torch.pca_lowrank(fit_feats, q=q)

    # Project to PCA space
    desc_a_all = desc_a_all @ v[:, :q]
    desc_b_all = desc_b_all @ v[:, :q]

    # Normalize & reshape helper
    def to_rgb(desc, H, W):
        # Per-channel normalization (better visuals)
        desc = desc - desc.mean(dim=0, keepdim=True)
        desc = desc / (desc.std(dim=0, keepdim=True) + 1e-6)

        # Scale to 0–255
        desc = (desc - desc.min()) / (desc.max() - desc.min() + 1e-8)
        desc = (255 * desc).clamp(0, 255).byte()

        return desc.reshape(H, W, q).cpu().numpy()

    img_a = to_rgb(desc_a_all, H1, W1)
    img_b = to_rgb(desc_b_all, H2, W2)

    return img_a, img_b


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature correspondence visualization with PCA"
    )

    parser.add_argument(
        "--data_val",
        type=str,
        help="Path to validation features",
    )

    parser.add_argument(
        "--templates_dir",
        type=str,
        help="Path to template features directory",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        help="Path to BOP dataset",
    )

    parser.add_argument(
        "--val_ids",
        type=int,
        nargs="+",
        default=[90, 40, 200, 75, 63, 28, 123, 52],
        help="Validation image IDs",
    )

    parser.add_argument(
        "--template_id", type=int, default=134, help="Template image ID"
    )

    parser.add_argument(
        "--num_corr", type=int, default=150, help="Number of correspondences"
    )

    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Similarity threshold"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output figures. If omitted, figures are displayed interactively.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    data_val = torch.load(args.data_val, weights_only=False)

    dataset = BOPValDataset(
        root_dir=args.dataset_root,
        split="val",
    )

    for val_img_id in args.val_ids:
        obj_id = str(dataset[val_img_id]["obj_id"]).zfill(6)

        template_path = f"{args.templates_dir}/obj_{obj_id}/features/dinov3.pt"

        try:
            data_onb = torch.load(template_path, weights_only=False)
        except (FileNotFoundError, EOFError, RuntimeError) as e:
            logger.warning(f"Skipping {obj_id}: {type(e).__name__}: {e}")
            continue

        data_1 = data_val[val_img_id]
        data_2 = data_onb[args.template_id]

        out_dir = None
        if args.output_dir is not None:
            out_dir = Path(args.output_dir) / f"obj_{obj_id}"
            out_dir.mkdir(parents=True, exist_ok=True)

        # ---------------- DEBUG 1 ----------------
        ## Debug 1: Visualize cropped images and corresponding object masks for both views
        plt.subplot(221)
        plt.imshow(data_1["cropped_image"])
        plt.subplot(222)
        plt.imshow(data_2["cropped_image"])
        plt.subplot(223)
        plt.imshow(data_1["mask"])
        plt.subplot(224)
        plt.imshow(data_2["mask"])
        if out_dir is not None:
            plt.savefig(out_dir / f"val{val_img_id}_debug1.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        # ---------------- DEBUG 2 ----------------
        ## Debug 2: Inspect feature patch grid and reconstruct a coarse spatial map from patch features
        plt.subplot(141)
        plt.imshow(data_1["cropped_image"])
        plt.title("Cropped")
        plt.subplot(142)
        plt.imshow(data_1["resized_image"][0])
        plt.title("Resized")
        plt.subplot(143)
        plt.imshow(data_1["features"][0])
        plt.title("Feature[0]")

        im = torch.zeros(
            (data_1["cropped_image"].size[1], data_1["cropped_image"].size[0])
        )
        bbox = data_1["bbox"]

        pu, pv = inv_resize_to_patches(
            (bbox[2] - bbox[0], bbox[3] - bbox[1]),
            data_1["resized_image"].shape[1:],
            PATCH_SIZE,
        )

        feat_flat = data_1["features"][0].flatten()
        pu = pu[PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE].flatten()
        pv = pv[PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE].flatten()

        for i in range(len(feat_flat)):
            idx_1 = int(pu[i].round().item())
            idx_2 = int(pv[i].round().item())
            im[
                idx_1 - PATCH_SIZE // 2 : idx_1 + PATCH_SIZE // 2 - 1,
                idx_2 - PATCH_SIZE // 2 : idx_2 + PATCH_SIZE // 2 - 1,
            ] = feat_flat[i]

        plt.subplot(144)
        plt.imshow(im)
        plt.title("Recon")
        if out_dir is not None:
            plt.savefig(out_dir / f"val{val_img_id}_debug2.png", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        # ---------------- DEBUG 3 ----------------
        ## Debug 3: Compute feature similarities and visualize pixel-level correspondences between full images
        image_1 = Image.open(data_1["image_path"])
        image_2 = Image.open(data_2["image_path"])

        f1 = F.normalize(data_1["features"].flatten(1), p=2, dim=0)
        f2 = F.normalize(data_2["features"].flatten(1), p=2, dim=0)

        f1 = f1 * data_1["mask"][
            PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE
        ].flatten().unsqueeze(0)
        f2 = f2 * data_2["mask"][
            PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE
        ].flatten().unsqueeze(0)

        # Check the shapes
        logger.info("Shape of f1: %s", f1.shape)
        logger.info("Shape of f2: %s", f2.shape)

        # Calculate similarity matrix (dot product of transposed f1 and f2)
        sim = f1.T @ f2

        sim_flattened = sim.flatten()
        threshold_topk = torch.topk(sim_flattened, args.num_corr, largest=True).values[
            -1
        ]
        user_threshold = getattr(args, "threshold", None)
        if user_threshold is not None:
            # Use the stricter of the top-k threshold and the user-provided minimum similarity
            threshold = max(threshold_topk, user_threshold)
            logger.info(
                "Top-%d similarity threshold: %s, user-specified minimum threshold: %s, effective threshold: %s",
                args.num_corr,
                threshold_topk,
                user_threshold,
                threshold,
            )
        else:
            threshold = threshold_topk
            logger.info(
                "Threshold for top-%d correspondences: %s",
                args.num_corr,
                threshold,
            )
        matches = (
            sim >= threshold
        )  # Mask where similarity is greater than or equal to the threshold

        # Get the row and column indices where the condition is True
        idx_1, idx_2 = torch.nonzero(matches, as_tuple=True)

        logger.info("Number of correspondences: %d", len(idx_1))

        for i, d in enumerate([data_1, data_2]):
            bbox = d["bbox"]
            x0, y0 = bbox[:2]
            pu, pv = inv_resize_to_patches(
                (bbox[2] - bbox[0], bbox[3] - bbox[1]),
                d["resized_image"].shape[1:],
                PATCH_SIZE,
            )
            pu = pu + y0
            pv = pv + x0

            pu = pu[
                PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE
            ].flatten()
            pv = pv[
                PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE
            ].flatten()
            if i == 0:
                pu_1 = pu[idx_1]
                pv_1 = pv[idx_1]
            elif i == 1:
                pu_2 = pu[idx_2]
                pv_2 = pv[idx_2]

        points_1 = []
        points_2 = []
        for i in range(len(pu_1)):
            points_1.append((pv_1[i].item(), pu_1[i].item()))
            points_2.append((pv_2[i].item(), pu_2[i].item()))

        plot_images_with_lines(
            image_1,
            image_2,
            points_1,
            points_2,
            save_path=out_dir / f"val{val_img_id}_debug3.png" if out_dir else None,
        )

        # ---------------- DEBUG 4 ----------------
        ## Debug 4: Visualize correspondences on feature embeddings using joint PCA
        mm_1, mm_2 = pca_feature_map_to_rgb(data_1["features"], data_2["features"])

        pu = torch.arange(mm_1.shape[0])
        pv = torch.arange(mm_1.shape[1])
        pu, pv = torch.meshgrid(pu, pv)
        pu = pu.flatten()
        pv = pv.flatten()
        pu_1 = pu[idx_1]
        pv_1 = pv[idx_1]

        pu = torch.arange(mm_2.shape[0])
        pv = torch.arange(mm_2.shape[1])
        pu, pv = torch.meshgrid(pu, pv)
        pu = pu.flatten()
        pv = pv.flatten()
        pu_2 = pu[idx_2]
        pv_2 = pv[idx_2]

        points_1 = []
        points_2 = []
        for i in range(len(pu_1)):
            points_1.append((pv_1[i].item(), pu_1[i].item()))
            points_2.append((pv_2[i].item(), pu_2[i].item()))

        plot_images_with_lines(
            np.array(mm_1),
            np.array(mm_2),
            points_1,
            points_2,
            save_path=out_dir / f"val{val_img_id}_debug4.png" if out_dir else None,
        )
