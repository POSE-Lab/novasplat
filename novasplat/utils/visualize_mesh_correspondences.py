"""
Visualize strongest correspondences between two objects.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt

from novasplat.utils.pose_estimation import ransac_se3_gpu

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Load Features + Keypoints
# -------------------------------------------------------


def load_object_data(obj_id: int, base_dir: Path):
    obj_dir = base_dir / f"obj_{obj_id:06d}"

    features = torch.load(obj_dir / "features" / "fused.pt", weights_only=False)

    gedi_data = torch.load(obj_dir / "features" / "gedi.pt", weights_only=False)

    keypoints = (
        torch.from_numpy(gedi_data["keypoints"])
        if isinstance(gedi_data["keypoints"], np.ndarray)
        else gedi_data["keypoints"]
    )

    return features, keypoints


# -------------------------------------------------------
# Similarity
# -------------------------------------------------------


def compute_cosine_similarity(features_a, features_b):
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)
    return torch.mm(features_a, features_b.t())


def find_topk_correspondences(similarity, k=20):
    max_sim, best_matches = similarity.max(dim=1)

    all_corr = []
    for idx_a in range(similarity.shape[0]):
        idx_b = best_matches[idx_a].item()
        score = max_sim[idx_a].item()
        all_corr.append((idx_a, idx_b, score))

    all_corr.sort(key=lambda x: x[2], reverse=True)
    return all_corr[:k]


# -------------------------------------------------------
# Mesh Utilities
# -------------------------------------------------------


def compute_translation_from_mesh(mesh_a, mesh_b, scale=1.5):
    """
    Compute translation along X so meshes are clearly separated.
    """
    bbox_a = mesh_a.get_axis_aligned_bounding_box()
    extent_a = bbox_a.get_extent()

    shift = extent_a[0] * scale
    return np.array([shift, 0.0, 0.0])


def translate_mesh(mesh, translation):
    mesh = mesh.translate(translation, relative=True)
    return mesh


# -------------------------------------------------------
# Save Correspondences (ONLY top-k EDGES)
# -------------------------------------------------------


def save_correspondences_only(
    keypoints_a,
    keypoints_b,
    correspondences,
    translation_b,
    output_path,
):
    """
    Save only top-k correspondence edges (2 * k vertices total).
    """

    vertices = []
    edges = []

    # Convert translation_b to numpy array if it's a torch.Tensor
    if isinstance(translation_b, torch.Tensor):
        translation_b_np = translation_b.cpu().numpy()
    else:
        translation_b_np = translation_b

    for i, (idx_a, idx_b, _) in enumerate(correspondences):
        pt_a = keypoints_a[idx_a].cpu().numpy()
        # Ensure translation_b is a numpy array
        if hasattr(translation_b, "cpu") and hasattr(translation_b, "numpy"):
            translation_b_np = translation_b.cpu().numpy()
        else:
            translation_b_np = translation_b
        pt_b = np.add(keypoints_b[idx_b].cpu().numpy(), translation_b_np)

        vertices.append(pt_a)
        vertices.append(pt_b)

        edges.append((2 * i, 2 * i + 1))

    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")

    logger.info(f"Saved correspondences to {output_path}")


# -------------------------------------------------------
# Similarity Plot
# -------------------------------------------------------


def plot_similarity_matrix(similarity, output_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity.numpy(), cmap="viridis", aspect="auto")
    plt.colorbar(label="Cosine Similarity")
    plt.title("Feature Similarity Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved similarity matrix to {output_path}")


def write_ply(filename, points, colors):
    """
    points: (N,3) tensor
    colors: (N,3) uint8 tensor
    """
    points = points.detach().cpu().numpy()
    colors = colors.detach().cpu().numpy()

    # Clamp color values to [0, 255] and convert to uint8
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    N = points.shape[0]

    header = f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(filename, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


def save_registration_ply(
    pts1, pts2, R, t, before_filename="before.ply", after_filename="after.ply"
):
    """
    pts1: (N,3) source points (torch tensor)
    pts2: (M,3) target points (torch tensor)
    R: (3,3) rotation matrix
    t: (3,) translation vector
    """

    device = pts1.device

    red = torch.tensor([255, 0, 0], dtype=torch.uint8, device=device)
    green = torch.tensor([0, 255, 0], dtype=torch.uint8, device=device)

    # BEFORE
    colors1 = red.repeat(pts1.shape[0], 1)
    colors2 = green.repeat(pts2.shape[0], 1)

    before_points = torch.cat([pts1, pts2], dim=0)
    before_colors = torch.cat([colors1, colors2], dim=0)

    write_ply(before_filename, before_points, before_colors)

    # AFTER
    # Ensure R and t are on the same device as pts1
    R = R.to(pts1.device)
    t = t.to(pts1.device)
    pts1_transformed = (R @ pts1.T).T + t

    after_points = torch.cat([pts1_transformed, pts2], dim=0)
    after_colors = torch.cat([colors1, colors2], dim=0)

    write_ply(after_filename, after_points, after_colors)

    logger.info("PLY files saved:")
    logger.info(before_filename)
    logger.info(after_filename)


def write_ply(filename, points, colors):
    """
    points: (N,3) tensor
    colors: (N,3) uint8 tensor
    """
    points = points.detach().cpu().numpy()
    colors = colors.detach().cpu().numpy()

    N = points.shape[0]

    header = f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(filename, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


def save_registration_ply(
    pts1, pts2, R, t, before_filename="before.ply", after_filename="after.ply"
):
    """
    pts1: (N,3) source points (torch tensor)
    pts2: (M,3) target points (torch tensor)
    R: (3,3) rotation matrix
    t: (3,) translation vector
    """

    device = pts1.device

    red = torch.tensor([255, 0, 0], dtype=torch.uint8, device=device)
    green = torch.tensor([0, 255, 0], dtype=torch.uint8, device=device)

    # BEFORE
    colors1 = red.repeat(pts1.shape[0], 1)
    colors2 = green.repeat(pts2.shape[0], 1)

    before_points = torch.cat([pts1, pts2], dim=0)
    before_colors = torch.cat([colors1, colors2], dim=0)

    write_ply(before_filename, before_points, before_colors)

    # AFTER
    # Ensure R and t are on the same device as pts1
    R = R.to(pts1.device)
    t = t.to(pts1.device)
    pts1_transformed = (R @ pts1.T).T + t

    after_points = torch.cat([pts1_transformed, pts2], dim=0)
    after_colors = torch.cat([colors1, colors2], dim=0)

    write_ply(after_filename, after_points, after_colors)

    print("PLY files saved:")
    print(before_filename)
    print(after_filename)


# -------------------------------------------------------
# Main
# -------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_a", type=int, default=4)
    parser.add_argument("--obj_b", type=int, default=15)
    parser.add_argument("--base_dir", type=str, default="output/onboarding")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nLoading object {args.obj_a}")
    features_a, keypoints_a = load_object_data(args.obj_a, base_dir)
    logger.info(
        f"Features shape: {features_a.shape}, Keypoints shape: {keypoints_a.shape}"
    )

    logger.info(f"Loading object {args.obj_b}")
    features_b, keypoints_b = load_object_data(args.obj_b, base_dir)
    logger.info(
        f"Features shape: {features_b.shape}, Keypoints shape: {keypoints_b.shape}"
    )

    features_a = features_a[:, :]  # Select only gedi, only dino or both
    features_b = features_b[:, :]  # Select only gedi, only dino or both

    logger.info("\nComputing similarity...")
    similarity = compute_cosine_similarity(features_a, features_b)

    logger.info(f"Max similarity: {similarity.max().item():.4f}")
    logger.info(f"Mean similarity: {similarity.mean().item():.4f}")

    # Save similarity heatmap
    sim_out = output_dir / f"similarity_obj{args.obj_a}_obj{args.obj_b}.png"
    plot_similarity_matrix(similarity, sim_out)

    # Select top correspondences
    correspondences = find_topk_correspondences(similarity, k=args.top_k)
    logger.info(f"Selected {len(correspondences)} correspondences")

    start_time = time.time()
    rotation, translation = ransac_se3_gpu(
        keypoints_a,
        keypoints_b,
        features_a,
        features_b,
        correspondences,
    )
    elapsed = time.time() - start_time
    logger.info(f"Estimated rotation:\n{rotation.cpu().numpy()}")
    logger.info(f"Estimated translation:\n{translation.cpu().numpy()}")
    logger.info(f"ransac_se3_gpu took {elapsed:.4f} seconds")

    save_registration_ply(
        keypoints_a,
        keypoints_b,
        rotation,
        translation,
        before_filename=output_dir / "before_registration.ply",
        after_filename=output_dir / "after_registration.ply",
    )

    # Load meshes
    mesh_a_path = base_dir / f"obj_{args.obj_a:06d}" / "model.ply"
    mesh_b_path = base_dir / f"obj_{args.obj_b:06d}" / "model.ply"

    mesh_a = o3d.io.read_triangle_mesh(str(mesh_a_path))
    mesh_b = o3d.io.read_triangle_mesh(str(mesh_b_path))

    if mesh_a.is_empty() or mesh_b.is_empty():
        raise RuntimeError("One of the meshes is empty.")

    # Compute automatic translation
    translation_b = compute_translation_from_mesh(mesh_a, mesh_b)

    # Translate mesh B
    mesh_b_translated = translate_mesh(mesh_b, translation_b)

    # Save meshes
    mesh_a_out = output_dir / f"mesh_obj{args.obj_a}.ply"
    mesh_b_out = output_dir / f"mesh_obj{args.obj_b}_translated.ply"

    o3d.io.write_triangle_mesh(str(mesh_a_out), mesh_a)
    o3d.io.write_triangle_mesh(str(mesh_b_out), mesh_b_translated)

    logger.info(f"Saved mesh A to {mesh_a_out}")
    logger.info(f"Saved translated mesh B to {mesh_b_out}")

    # Save correspondences
    corr_out = (
        output_dir
        / f"correspondences_top{args.top_k}_obj{args.obj_a}_obj{args.obj_b}.ply"
    )

    save_correspondences_only(
        keypoints_a,
        keypoints_b,
        correspondences,
        translation_b,
        corr_out,
    )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
