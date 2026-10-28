import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh

from novasplat.utils.camera import compute_xyz
from novasplat.utils.pose_estimation import (
    compute_fused_similarity,
    compute_similarity,
    geometry_ransac_se3,
    get_top_matches,
)


def resolve_gedi_dim(gedi_dim_override: int | None) -> int:
    if gedi_dim_override is not None:
        return gedi_dim_override

    try:
        from novasplat.features.dense_features import GEDI_CONFIG
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import GEDI_CONFIG from novasplat.features.dense_features. "
            "Install GeDi dependencies or pass --gedi_dim explicitly."
        ) from exc

    r_lrf = GEDI_CONFIG["r_lrf"]
    n_scales = len(r_lrf) if isinstance(r_lrf, (list, tuple)) else 1
    return GEDI_CONFIG["dim"] * n_scales


# -------------------------
# IO / Utils
# -------------------------
def load_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return data


def unnormalize_image(img):
    if torch.is_tensor(img):
        img = img.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = img.permute(1, 2, 0).cpu().numpy()
    return (img.clip(0, 1) * 255).astype(np.uint8)


# -------------------------
# GT TRANSFORM
# -------------------------
def load_gt_transform(image_path: Path, obj_id: int, gt_instance_id: int | None = None):
    scene_dir = image_path.parent.parent  # FIXED
    gt_path = scene_dir / "scene_gt.json"

    with open(gt_path, "r") as f:
        gt = json.load(f)

    frame_id = str(int(image_path.stem))
    objects = gt[frame_id]

    if gt_instance_id is not None:
        if not (0 <= gt_instance_id < len(objects)):
            raise ValueError(
                f"gt_instance_id={gt_instance_id} out of range for frame {frame_id}"
            )
        obj = objects[gt_instance_id]
        if int(obj["obj_id"]) != int(obj_id):
            raise ValueError(
                "GT instance/object mismatch: "
                f"requested obj_id={obj_id}, gt_instance_id={gt_instance_id}, "
                f"frame obj_id={obj['obj_id']}"
            )
        R = torch.tensor(obj["cam_R_m2c"], dtype=torch.float32).view(3, 3)
        t = torch.tensor(obj["cam_t_m2c"], dtype=torch.float32)
        return R, t

    for obj in objects:
        if obj["obj_id"] == obj_id:
            R = torch.tensor(obj["cam_R_m2c"], dtype=torch.float32).view(3, 3)
            t = torch.tensor(obj["cam_t_m2c"], dtype=torch.float32)
            return R, t

    raise ValueError(f"Object {obj_id} not found in frame {frame_id}")


def load_predicted_pose(csv_path: Path, image_path: Path, obj_id: int):
    """
    Load predicted pose (R, t) from a CSV file.

    Expected CSV columns:
        - scene_id
        - im_id
        - obj_id
        - R (flattened 3x3, space-separated)
        - t (3-vector, space-separated)

    Returns:
        R (3x3 torch.FloatTensor)
        t (3,) torch.FloatTensor

    Notes:
        - Converts from model-to-camera to camera-to-model convention
        - Matches the transformation used elsewhere in the pipeline
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Extract identifiers from image path
    scene_id = int(image_path.parent.parent.stem)
    im_id = int(image_path.stem)

    row = df[
        (df["scene_id"] == scene_id) & (df["im_id"] == im_id) & (df["obj_id"] == obj_id)
    ]

    if row.empty:
        raise ValueError(
            f"No prediction found for scene_id={scene_id}, "
            f"im_id={im_id}, obj_id={obj_id}"
        )

    # Parse rotation and translation
    R = torch.tensor(
        [float(x) for x in row["R"].values[0].strip().split()], dtype=torch.float32
    ).view(3, 3)

    t = torch.tensor(
        [float(x) for x in row["t"].values[0].strip().split()], dtype=torch.float32
    )

    # Convert convention (m2c → c2m)
    R = R.T
    t = -R @ t

    return R, t


def apply_inverse_transform(points, R, t):
    return (R.T @ (points - t).T).T


# -------------------------
# PLY EXPORT
# -------------------------
def export_colored_pointcloud_ply(filename, points, colors):
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.detach().cpu().numpy()

    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def draw_lines_ply(points_1, points_2, save_path, radius=0.002):
    if torch.is_tensor(points_1):
        points_1 = points_1.detach().cpu().numpy()
    else:
        points_1 = np.array(points_1)

    if torch.is_tensor(points_2):
        points_2 = points_2.detach().cpu().numpy()
    else:
        points_2 = np.array(points_2)

    cylinders = []
    for p1, p2 in zip(points_1, points_2):
        vec = p2 - p1
        height = np.linalg.norm(vec)
        if height < 1e-6:
            continue

        cylinder = trimesh.creation.cylinder(radius=radius, height=height)

        direction = vec / height
        T = trimesh.geometry.align_vectors([0, 0, 1], direction)
        cylinder.apply_transform(T)
        cylinder.apply_translation((p1 + p2) / 2)

        cylinders.append(cylinder)

    if not cylinders:
        print("No lines to export.")
        return

    trimesh.util.concatenate(cylinders).export(save_path)
    print(f"Saved lines → {save_path}")


# -------------------------
# CORE LOGIC
# -------------------------


# -------------------------
# MAIN PIPELINE
# -------------------------
def run(args):
    print("Loading data...")
    dino = torch.load(args.dino, weights_only=False)

    print("Computing XYZ...")
    XYZ_dense = compute_xyz(dino, args.val_idx, patch=False)

    mask_dense = (
        dino[args.val_idx]["mask"] * (dino[args.val_idx]["depth"] > 0)
    ).flatten()

    XYZ_dense = XYZ_dense[mask_dense > 0]

    if args.features == "predicted":
        if args.predicted_csv is None:
            raise ValueError("--predicted_csv is required when --features predicted")
        csv_path = Path(args.predicted_csv)
        image_path = Path(dino[args.val_idx]["image_path"])
        obj_id = args.gt_obj_id
        R, t = load_predicted_pose(csv_path, image_path, obj_id)

        XYZ_dense_transformed = (R @ XYZ_dense.T).T + t

        img = unnormalize_image(dino[args.val_idx]["resized_image"])

        export_colored_pointcloud_ply(
            args.output_dir / f"transformed_rgb_{args.features}.ply",
            XYZ_dense_transformed,
            torch.from_numpy(img)[
                mask_dense.view(img.shape[0], img.shape[1]).cpu().bool()
            ],
        )
        return

    if args.fused_a is None or args.fused_b is None:
        raise ValueError(
            "--fused_a and --fused_b are required unless --features predicted is used"
        )

    # Features from Onboarding PCD
    fused_b = torch.load(args.fused_b, weights_only=False)
    gedi_dim = (
        resolve_gedi_dim(args.gedi_dim)
        if args.features in {"gedi", "dino", "fused"}
        else 0
    )
    desc_dim_b = fused_b["descriptors"].shape[1]
    needs_gedi_split = args.features in {"gedi", "dino"} or (
        args.features == "fused" and args.gedi_weight != 0.5
    )
    if needs_gedi_split and not (0 < gedi_dim < desc_dim_b):
        raise ValueError(
            "Invalid GeDi split for onboarding descriptors: "
            f"gedi_dim={gedi_dim}, total_dim={desc_dim_b}. "
            "Pass --gedi_dim with the descriptor size used at feature fusion time."
        )
    if args.features == "gedi":
        feat_b = fused_b["descriptors"].T[:gedi_dim, :]
    elif args.features == "dino":
        feat_b = fused_b["descriptors"].T[gedi_dim:, :]
    elif args.features == "fused":
        feat_b = fused_b["descriptors"].T
    else:
        raise ValueError(f"Unknown feature mode: {args.features}")

    keypoints_b = fused_b["keypoints"]

    # Features from Val Image
    fused_a = torch.load(args.fused_a, weights_only=False)
    desc_dim_a = fused_a["descriptors"].shape[1]
    if needs_gedi_split and not (0 < gedi_dim < desc_dim_a):
        raise ValueError(
            "Invalid GeDi split for validation descriptors: "
            f"gedi_dim={gedi_dim}, total_dim={desc_dim_a}. "
            "Pass --gedi_dim with the descriptor size used at feature fusion time."
        )
    if desc_dim_a != desc_dim_b:
        raise ValueError(
            "Descriptor dimensionality mismatch between fused_a and fused_b: "
            f"{desc_dim_a} vs {desc_dim_b}."
        )
    if args.features == "gedi":
        feat_a = fused_a["descriptors"].T[:gedi_dim, :]
    elif args.features == "dino":
        feat_a = fused_a["descriptors"].T[gedi_dim:, :]
    elif args.features == "fused":
        feat_a = fused_a["descriptors"].T
    else:
        raise ValueError(f"Unknown feature mode: {args.features}")
    keypoints_a = fused_a["keypoints"]

    if (
        args.features == "dino"
        and "dinov3_features_count" in fused_a
        and "dinov3_features_count" in fused_b
    ):
        support_a = (
            torch.as_tensor(fused_a["dinov3_features_count"]) >= args.min_dino_support
        )
        support_b = (
            torch.as_tensor(fused_b["dinov3_features_count"]) >= args.min_dino_support
        )

        feat_a = feat_a[:, support_a]
        feat_b = feat_b[:, support_b]
        keypoints_a = keypoints_a[support_a]
        keypoints_b = keypoints_b[support_b]

        print(
            "DINO support filtering "
            f"(min={args.min_dino_support}): "
            f"val {int(support_a.sum())}/{support_a.numel()}, "
            f"onb {int(support_b.sum())}/{support_b.numel()}"
        )

    # -------------------------
    # APPLY INVERSE GT TRANSFORM
    # -------------------------
    if args.apply_gt:
        print(f"Applying inverse GT transform (obj_id={args.gt_obj_id})...")
        image_path = Path(dino[args.val_idx]["image_path"])

        R, t = load_gt_transform(
            image_path,
            args.gt_obj_id,
            gt_instance_id=args.gt_instance_id,
        )

        R = R.to(XYZ_dense.device)
        t = t.to(XYZ_dense.device)

        XYZ_dense = apply_inverse_transform(XYZ_dense, R, t)
        keypoints_a = apply_inverse_transform(keypoints_a, R, t)

    print("Exporting RGB pointcloud...")
    img = unnormalize_image(dino[args.val_idx]["resized_image"])
    export_colored_pointcloud_ply(
        args.output_dir / "patches_rgb.ply",
        XYZ_dense + args.offset,
        torch.from_numpy(img)[mask_dense.view(img.shape[0], img.shape[1]).cpu().bool()],
    )

    print("Computing similarity...")
    if args.features == "fused" and args.gedi_weight != 0.5:
        print(f"Using weighted fused similarity (gedi_weight={args.gedi_weight})")
        sim = compute_fused_similarity(
            feat_a, feat_b, gedi_dim, gedi_weight=args.gedi_weight
        )
    else:
        sim = compute_similarity(feat_a, feat_b)

    matcher = args.matcher
    if matcher == "auto":
        matcher = "mnn" if args.features == "dino" else "dedup_asymmetric"

    idx_1, idx_2 = get_top_matches(
        sim,
        args.topk,
        method=matcher,
        ratio_thresh=args.ratio_thresh,
        ratio_min_margin=args.ratio_min_margin,
    )
    print(f"Using matcher: {matcher}. Correspondences found: {idx_1.numel()}")
    if idx_1.numel() < 3:
        raise RuntimeError(
            "Insufficient correspondences for SE(3) estimation. "
            f"Found {idx_1.numel()} matches with matcher '{matcher}'."
        )

    XYZ_1 = keypoints_a[idx_1] + args.offset
    XYZ_2 = keypoints_b[idx_2]

    print("Drawing correspondences...")
    draw_lines_ply(
        XYZ_1,
        XYZ_2,
        save_path=args.output_dir / f"lines_{args.features}.ply",
        radius=args.radius,
    )

    # Ransac
    # R, t = ransac_se3_gpu(keypoints_a, keypoints_b, feat_a.T, feat_b.T, idx_1, idx_2)
    # R, t = least_squares_se3(keypoints_a, keypoints_b, idx_1, idx_2)
    R, t = geometry_ransac_se3(keypoints_a, keypoints_b, idx_1, idx_2)

    # Inverse transform
    if args.apply_gt:
        R = R.T
        t = -R @ t

    print("R", R)
    print("t", t)

    # -------------------------
    # Transform and Save
    # -------------------------
    print("Exporting transformed RGB pointcloud...")
    R = R.to(XYZ_dense.device)
    t = t.to(XYZ_dense.device)
    XYZ_dense_transformed = (R @ XYZ_dense.T).T + t

    export_colored_pointcloud_ply(
        args.output_dir / f"transformed_rgb_{args.features}.ply",
        XYZ_dense_transformed,
        torch.from_numpy(img)[mask_dense.view(img.shape[0], img.shape[1]).cpu().bool()],
    )


# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fused_a", type=Path, required=False)
    parser.add_argument("--fused_b", type=Path, required=False)

    parser.add_argument("--dino", type=Path, required=True)

    parser.add_argument("--val_idx", type=int, default=0)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--offset", type=float, default=60.0)

    parser.add_argument("--output_dir", type=Path, default=Path("debug"))

    parser.add_argument("--apply_gt", action="store_true")
    parser.add_argument("--gt_obj_id", type=int, default=1)
    parser.add_argument(
        "--gt_instance_id",
        type=int,
        default=None,
        help=(
            "Optional GT instance index in scene_gt for frames that contain "
            "multiple instances of the same object id."
        ),
    )

    parser.add_argument(
        "--features",
        choices=["gedi", "dino", "fused", "predicted"],
        default="fused",
        help="Select which features to use",
    )
    parser.add_argument("--predicted_csv", type=Path, required=False)
    parser.add_argument(
        "--gedi_dim",
        type=int,
        required=False,
        help=(
            "Override GeDi descriptor dimension used to split fused descriptors. "
            "Useful when GeDi dependencies are unavailable."
        ),
    )
    parser.add_argument(
        "--matcher",
        choices=["auto", "dedup_asymmetric", "mnn", "mnn_ratio"],
        default="auto",
        help=(
            "Matching strategy for selecting correspondences. "
            "auto uses mnn for --features dino and "
            "dedup_asymmetric otherwise."
        ),
    )
    parser.add_argument(
        "--ratio_thresh",
        type=float,
        default=1.02,
        help=(
            "Only used with --matcher mnn_ratio. "
            "Requires best/second-best similarity ratio >= this value."
        ),
    )
    parser.add_argument(
        "--ratio_min_margin",
        type=float,
        default=0.0,
        help=(
            "Only used with --matcher mnn_ratio. "
            "Requires best-second_best similarity margin >= this value."
        ),
    )
    parser.add_argument(
        "--min_dino_support",
        type=float,
        default=1.0,
        help=(
            "Minimum per-keypoint DINO support count used only when "
            "--features dino and support metadata is available in fused files."
        ),
    )
    parser.add_argument(
        "--gedi_weight",
        type=float,
        default=0.2,
        help=(
            "Only used with --features fused. Weight applied to the GeDi "
            "similarity before combining with the DINO similarity: "
            "sim = gedi_weight * sim_gedi + (1 - gedi_weight) * sim_dino. "
            "Default 0.2 gives an 80/20 DINO/GeDi split. "
            "Use 0.1-0.3 to reduce GeDi noise."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    run(args)
