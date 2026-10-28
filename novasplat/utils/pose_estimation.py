from typing import List, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F


def estimate_inlier_threshold(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    explicit_thresh: float | None,
    relative_scale: float = 0.1,
    min_thresh: float = 5.0,
) -> float:
    """Estimate a robust inlier threshold for RANSAC in the current units.

    The geometry code is used with both normalized coordinates and object-space
    coordinates expressed in dataset units (for example millimeters). A fixed
    threshold like 0.05 works only in the normalized case, so by default we
    scale the threshold from the correspondence extent.
    """
    if explicit_thresh is not None:
        if explicit_thresh <= 0:
            raise ValueError(f"inlier_thresh must be > 0, got {explicit_thresh}")
        return float(explicit_thresh)

    if pts1.numel() == 0 or pts2.numel() == 0:
        return min_thresh

    diag1 = torch.linalg.vector_norm(pts1.max(dim=0).values - pts1.min(dim=0).values)
    diag2 = torch.linalg.vector_norm(pts2.max(dim=0).values - pts2.min(dim=0).values)
    scale = torch.minimum(diag1, diag2)
    return float(torch.clamp(scale * relative_scale, min=min_thresh).item())


def edge_lengths(p: torch.Tensor) -> torch.Tensor:
    """Compute edge lengths for triplets of points.

    Args:
        p: (K,3,3) tensor of K triplets of 3D points.

    Returns:
        (K,3) tensor of edge lengths [d01, d02, d12] for each triplet.
    """
    d01 = torch.norm(p[:, 0] - p[:, 1], dim=1)
    d02 = torch.norm(p[:, 0] - p[:, 2], dim=1)
    d12 = torch.norm(p[:, 1] - p[:, 2], dim=1)
    return torch.stack([d01, d02, d12], dim=1)


def sample_distinct_triplets(
    num_points: int,
    num_iterations: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample triplets of distinct correspondence indices."""
    if num_points < 3:
        raise ValueError("At least 3 correspondences are required for triplet sampling")

    triplets = torch.randint(0, num_points, (num_iterations, 3), device=device)
    duplicate_mask = (
        (triplets[:, 0] == triplets[:, 1])
        | (triplets[:, 0] == triplets[:, 2])
        | (triplets[:, 1] == triplets[:, 2])
    )

    while duplicate_mask.any():
        num_bad = int(duplicate_mask.sum().item())
        triplets[duplicate_mask] = torch.randint(
            0, num_points, (num_bad, 3), device=device
        )
        duplicate_mask = (
            (triplets[:, 0] == triplets[:, 1])
            | (triplets[:, 0] == triplets[:, 2])
            | (triplets[:, 1] == triplets[:, 2])
        )

    return triplets


def non_degenerate_triplet_mask(
    p: torch.Tensor,
    area_ratio_tol: float = 1e-3,
    min_area: float = 1e-10,
) -> torch.Tensor:
    """Reject near-collinear triplets using a scale-aware area threshold."""
    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 2] - p[:, 0]
    double_area = torch.norm(torch.cross(v1, v2, dim=1), dim=1)

    # Compare against a per-triplet relative threshold (proportional to max edge^2)
    # so the degeneracy test is robust across different coordinate scales.
    max_edge = edge_lengths(p).max(dim=1).values
    area_thresh = torch.clamp(area_ratio_tol * max_edge.square(), min=min_area)
    return double_area > area_thresh


def sample_pruned_triplets(
    corr_pts1: torch.Tensor,
    corr_pts2: torch.Tensor,
    num_iterations: int,
    edge_tol: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample and prune triplets; progressively relax pruning if needed."""
    if corr_pts1.shape[0] != corr_pts2.shape[0]:
        raise ValueError("corr_pts1 and corr_pts2 must have the same number of rows")

    num_corr = int(corr_pts1.shape[0])
    if num_corr < 3:
        raise ValueError("At least 3 correspondences are required for triplet sampling")

    # Retry with gradually relaxed geometric constraints to avoid brittle failures
    # on hard samples while keeping strict behavior as the default.
    schedules = (
        (1.0, 1e-3),
        (1.5, 5e-4),
        (2.5, 1e-4),
    )

    for edge_scale, area_ratio_tol in schedules:
        triplets = sample_distinct_triplets(num_corr, num_iterations, device)
        p1 = corr_pts1[triplets]  # (K,3,3)
        p2 = corr_pts2[triplets]

        edges1 = edge_lengths(p1)
        edges2 = edge_lengths(p2)

        denom = torch.maximum(edges1, edges2) + 1e-6
        edge_ratio = torch.abs(edges1 - edges2) / denom
        edge_valid = (edge_ratio < (edge_tol * edge_scale)).all(dim=1)
        non_degenerate = non_degenerate_triplet_mask(p1, area_ratio_tol=area_ratio_tol)
        valid_mask = edge_valid & non_degenerate

        if valid_mask.any():
            return p1[valid_mask], p2[valid_mask]

    raise RuntimeError("No valid triplets after pruning.")


def ransac_se3_gpu(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    feats1: torch.Tensor,
    feats2: torch.Tensor,
    correspondences: List[Tuple[int, int, float]],
    num_iterations: int = 10000,
    edge_tol: float = 0.1,
    inlier_thresh: float | None = None,
    device: str = "cuda",
):
    """RANSAC-based SE(3) estimation on GPU.

        Args:
        pts1, pts2: (N,3) tensors of 3D points from two meshes.
        feats1, feats2: (N,64) tensors of features for the points.
        correspondences: list of (idx1, idx2, score) tuples indicating putative matches.
        num_iterations: number of RANSAC iterations.
        edge_tol: relative tolerance for edge length consistency.
        inlier_thresh: distance threshold for inliers. When omitted, a
            scale-aware threshold is estimated from the correspondence extent.
        device: str, optional. Device to run computation on (default: "cuda").
            If set to a CUDA device (e.g., "cuda" or "cuda:0") but CUDA is unavailable, will automatically fall back to CPU.

    Returns:
        R: (3,3) rotation matrix
        t: (3,) translation vector
    """
    if len(correspondences) < 3:
        raise ValueError("At least 3 correspondences are required for SE(3) estimation")

    if (
        isinstance(device, str)
        and device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        # Fallback to CPU if CUDA is not available
        device = "cpu"
    pts1 = pts1.to(device)
    pts2 = pts2.to(device)
    feats1 = feats1.to(device)
    feats2 = feats2.to(device)

    # ------------------------------------
    # Convert correspondences to tensors
    # ------------------------------------
    idx1 = torch.tensor([c[0] for c in correspondences], device=device)
    idx2 = torch.tensor([c[1] for c in correspondences], device=device)

    corr_pts1 = pts1[idx1]  # (M,3)
    corr_pts2 = pts2[idx2]  # (M,3)
    inlier_thresh = estimate_inlier_threshold(corr_pts1, corr_pts2, inlier_thresh)
    corr_feats1 = feats1[idx1]  # (M,64)
    corr_feats2 = feats2[idx2]  # (M,64)

    # Normalize features for cosine similarity
    corr_feats1 = F.normalize(corr_feats1, dim=1)
    corr_feats2 = F.normalize(corr_feats2, dim=1)

    p1, p2 = sample_pruned_triplets(
        corr_pts1=corr_pts1,
        corr_pts2=corr_pts2,
        num_iterations=num_iterations,
        edge_tol=edge_tol,
        device=device,
    )

    K = p1.shape[0]

    # ------------------------------------
    # Batched SVD (Kabsch algorithm)
    # ------------------------------------
    c1 = p1.mean(dim=1, keepdim=True)
    c2 = p2.mean(dim=1, keepdim=True)

    X = p1 - c1
    Y = p2 - c2

    H = torch.matmul(X.transpose(1, 2), Y)  # (K,3,3)

    U, _, Vt = torch.linalg.svd(H)

    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # Reflection fix
    det = torch.det(R)
    Vt[det < 0, 2, :] *= -1
    R[det < 0] = torch.matmul(Vt[det < 0].transpose(1, 2), U[det < 0].transpose(1, 2))

    t = c2.squeeze(1) - torch.matmul(R, c1.squeeze(1).unsqueeze(-1)).squeeze(-1)

    # ------------------------------------
    # Evaluate all hypotheses in parallel
    # ------------------------------------
    # Transform all corr_pts1 with all hypotheses
    pts1_all = corr_pts1.unsqueeze(0).expand(K, -1, -1)  # (K,M,3)
    R_exp = R.unsqueeze(1)  # (K,1,3,3)

    transformed = torch.matmul(R_exp, pts1_all.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(
        1
    )

    residuals = torch.norm(transformed - corr_pts2.unsqueeze(0), dim=2)

    inliers = residuals < inlier_thresh

    # Feature-aware scoring
    cos_sim = (corr_feats1 * corr_feats2).sum(dim=1)  # (M,)
    cos_sim = cos_sim.unsqueeze(0).expand(K, -1)

    scores = (cos_sim * inliers.float()).sum(dim=1) / (
        inliers.sum(dim=1).float() + 1e-8
    )

    best_idx = torch.argmax(scores)

    best_R = R[best_idx]
    best_t = t[best_idx]

    return best_R, best_t


def compute_similarity(feat_1, feat_2, device=None):
    """
    Compute pairwise cosine similarity between two feature tensors.

    Each input is expected to have feature/channel dimension first. The tensors
    are flattened from dimension 1 onward, so inputs shaped like ``[D, N]`` or
    ``[D, H, W]`` become feature matrices of shape ``[D, N1]`` and ``[D, N2]``.
    The returned similarity matrix therefore has shape ``[N1, N2]``.

    The heavy matrix multiplication is performed on a selectable device
    (CUDA if available by default) to improve performance on large feature sets.

    Args:
        feat_1: First feature tensor with feature dimension first.
        feat_2: Second feature tensor with feature dimension first.
        device: Optional device on which to perform normalization and matrix
            multiplication. Defaults to CUDA when available, otherwise CPU.

    Returns:
        A similarity matrix of shape ``[N1, N2]``, moved back to the original
        device of ``feat_1``.
    """
    # Remember original device so we can restore the result there.
    orig_device = feat_1.device

    # Select computation device if not explicitly provided.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move data to computation device.
    feats1 = feat_1.flatten(1).to(device)
    feats2 = feat_2.flatten(1).to(device)
    # vm = valid_mask.flatten().to(device)

    # Normalize on the computation device.
    f1 = F.normalize(feats1, dim=0)
    f2 = F.normalize(feats2, dim=0)

    sim = f1.T @ f2

    # Move similarity back to the original device to preserve existing behavior.
    return sim.to(orig_device)


def compute_fused_similarity(
    feat_1,
    feat_2,
    gedi_dim: int,
    gedi_weight: float = 0.5,
    device=None,
):
    """Compute pairwise similarity for fused descriptors as a weighted sum.

    Each input is expected to have feature dimension first, shaped ``[D, N]``
    where ``D = gedi_dim + dino_dim``.  GeDi and DINO similarities are
    computed independently (each part is per-keypoint L2-normalised) and then
    combined:

        sim = gedi_weight * sim_gedi + (1 - gedi_weight) * sim_dino

    ``gedi_weight = 0.5`` (the default) is mathematically equivalent to plain
    ``compute_similarity`` on the concatenated descriptor.  Values below 0.5
    reduce GeDi's influence; values above 0.5 increase it.

    Args:
        feat_1: First fused feature tensor, shape ``[gedi_dim + dino_dim, N1]``.
        feat_2: Second fused feature tensor, shape ``[gedi_dim + dino_dim, N2]``.
        gedi_dim: Number of leading dimensions that belong to GeDi.
        gedi_weight: Weight applied to GeDi similarity (must be in ``[0, 1]``).
        device: Optional device for the computation (defaults to CUDA if available).

    Returns:
        Similarity matrix of shape ``[N1, N2]`` on the original device of
        ``feat_1``.
    """
    if not (0.0 <= gedi_weight <= 1.0):
        raise ValueError(f"gedi_weight must be in [0, 1], got {gedi_weight}")

    orig_device = feat_1.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f1 = feat_1.flatten(1).to(device)
    f2 = feat_2.flatten(1).to(device)

    gedi_f1 = F.normalize(f1[:gedi_dim], dim=0)
    gedi_f2 = F.normalize(f2[:gedi_dim], dim=0)
    dino_f1 = F.normalize(f1[gedi_dim:], dim=0)
    dino_f2 = F.normalize(f2[gedi_dim:], dim=0)

    sim_gedi = gedi_f1.T @ gedi_f2
    sim_dino = dino_f1.T @ dino_f2

    sim = gedi_weight * sim_gedi + (1.0 - gedi_weight) * sim_dino
    return sim.to(orig_device)


def get_top_matches(
    sim,
    topk,
    method: str = "dedup_asymmetric",
    ratio_thresh: float = 1.02,
    ratio_min_margin: float = 0.0,
):
    if sim.numel() == 0 or topk <= 0:
        return torch.empty(0, dtype=torch.long, device=sim.device), torch.empty(
            0, dtype=torch.long, device=sim.device
        )

    if method == "mnn" or method == "mnn_ratio":
        nn12_sims, nn12 = sim.max(dim=1)
        _, nn21 = sim.max(dim=0)

        idx1_all = torch.arange(sim.size(0), device=sim.device)
        mutual = nn21[nn12] == idx1_all

        if method == "mnn_ratio":
            if ratio_thresh <= 1.0:
                raise ValueError(
                    f"ratio_thresh must be > 1.0 for mnn_ratio, got {ratio_thresh}"
                )
            if sim.size(1) < 2:
                mutual &= False
            else:
                top2_sims, _ = torch.topk(sim, k=2, dim=1)
                ratio = top2_sims[:, 0] / (top2_sims[:, 1] + 1e-8)
                margin = top2_sims[:, 0] - top2_sims[:, 1]
                ratio_ok = (ratio >= ratio_thresh) & (margin >= ratio_min_margin)
                mutual &= ratio_ok

        valid_idx1 = idx1_all[mutual]
        if valid_idx1.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long, device=sim.device),
                torch.empty(0, dtype=torch.long, device=sim.device),
            )

        valid_idx2 = nn12[valid_idx1]
        valid_sims = nn12_sims[valid_idx1]
        order = torch.argsort(valid_sims, descending=True)
        valid_idx1 = valid_idx1[order]
        valid_idx2 = valid_idx2[order]

        actual_topk = min(topk, valid_idx1.numel())
        return valid_idx1[:actual_topk], valid_idx2[:actual_topk]

    if method != "dedup_asymmetric":
        raise ValueError(
            f"Unknown matching method '{method}'. "
            "Expected one of: dedup_asymmetric, mnn, mnn_ratio."
        )

    # Deduplicated asymmetric matcher.
    nn12_sims, nn12 = sim.max(dim=1)
    _, sorted_idx1 = torch.sort(nn12_sims, descending=True)
    sorted_idx2 = nn12[sorted_idx1]

    idx2_np = sorted_idx2.cpu().numpy()
    _, unique_indices = np.unique(idx2_np, return_index=True)
    unique_indices.sort()

    valid_indices = torch.from_numpy(unique_indices).to(sim.device)
    valid_idx1 = sorted_idx1[valid_indices]
    valid_idx2 = sorted_idx2[valid_indices]

    actual_topk = min(topk, valid_idx1.numel())
    return valid_idx1[:actual_topk], valid_idx2[:actual_topk]


def least_squares_se3(
    pts1: torch.Tensor, pts2: torch.Tensor, idx1: torch.Tensor, idx2: torch.Tensor
):
    """
    Computes the optimal SE(3) transformation (R, t) that minimizes
    || (R @ p1 + t) - p2 ||_2^2 for corresponding points using SVD (Kabsch).
    """
    ref_device = pts1.device
    ref_dtype = pts1.dtype

    pts1 = pts1.to(device=ref_device, dtype=ref_dtype)
    pts2 = pts2.to(device=ref_device, dtype=ref_dtype)
    idx1 = idx1.to(ref_device)
    idx2 = idx2.to(ref_device)

    if len(idx1) < 3:
        raise ValueError("At least 3 correspondences are required for SE(3) estimation")

    p1 = pts1[idx1]
    p2 = pts2[idx2]

    # Centroids
    c1 = p1.mean(dim=0, keepdim=True)
    c2 = p2.mean(dim=0, keepdim=True)

    # Center points
    X = p1 - c1
    Y = p2 - c2

    # Cross-covariance matrix
    H = X.transpose(0, 1) @ Y  # (3, 3)
    U, _, Vt = torch.linalg.svd(H)

    # Optimal rotation
    R = Vt.transpose(0, 1) @ U.transpose(0, 1)

    # Reflection fix
    if torch.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.transpose(0, 1) @ U.transpose(0, 1)

    # Optimal translation
    t = c2.squeeze(0) - R @ c1.squeeze(0)

    return R, t


def geometry_ransac_se3(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    idx1: torch.Tensor,
    idx2: torch.Tensor,
    num_iterations: int = 10000,
    edge_tol: float = 0.1,
    inlier_thresh: float | None = None,
    device: str = "cuda",
):
    """
    Robust geometric SE(3) transformation estimation using RANSAC.
    Input/Output matches `least_squares_se3`, but it performs a sophisticated
    batched hypothesis generation and prunes outliers by voting.
    """
    if len(idx1) < 3:
        raise ValueError("At least 3 correspondences are required for SE(3) estimation")

    if (
        isinstance(device, str)
        and device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        device = "cpu"

    pts1 = pts1.to(device)
    pts2 = pts2.to(device)
    idx1 = idx1.to(device)
    idx2 = idx2.to(device)

    corr_pts1 = pts1[idx1]  # (M,3)
    corr_pts2 = pts2[idx2]  # (M,3)
    inlier_thresh = estimate_inlier_threshold(corr_pts1, corr_pts2, inlier_thresh)

    p1, p2 = sample_pruned_triplets(
        corr_pts1=corr_pts1,
        corr_pts2=corr_pts2,
        num_iterations=num_iterations,
        edge_tol=edge_tol,
        device=device,
    )
    K = p1.shape[0]

    # Batched SVD (Kabsch)
    c1 = p1.mean(dim=1, keepdim=True)
    c2 = p2.mean(dim=1, keepdim=True)
    X = p1 - c1
    Y = p2 - c2
    H = torch.matmul(X.transpose(1, 2), Y)
    U, _, Vt = torch.linalg.svd(H)

    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
    det = torch.det(R)
    Vt[det < 0, 2, :] *= -1
    R[det < 0] = torch.matmul(Vt[det < 0].transpose(1, 2), U[det < 0].transpose(1, 2))

    t = c2.squeeze(1) - torch.matmul(R, c1.squeeze(1).unsqueeze(-1)).squeeze(-1)

    # Evaluate hypotheses in chunks to avoid materializing tensors of shape
    # proportional to (K, M, 3) for all hypotheses at once.
    chunk_size = min(1024, K)
    corr_pts2_exp = corr_pts2.unsqueeze(0)
    best_score = None
    best_mean_residual = None
    best_idx = None
    best_inliers = None

    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        R_chunk = R[start:end]
        t_chunk = t[start:end]

        transformed = torch.einsum(
            "kij,mj->kmi", R_chunk, corr_pts1
        ) + t_chunk.unsqueeze(1)
        residuals = torch.norm(transformed - corr_pts2_exp, dim=2)
        inliers = residuals < inlier_thresh

        # Prefer the hypothesis with the most inliers and break ties using the
        # smallest mean inlier residual.
        scores = inliers.sum(dim=1)
        residual_sums = (residuals * inliers.float()).sum(dim=1)
        mean_inlier_residual = residual_sums / scores.clamp_min(1).float()

        max_score = scores.max()
        tied = scores == max_score
        tied_offsets = torch.nonzero(tied, as_tuple=False).flatten()
        tied_best_local = tied_offsets[torch.argmin(mean_inlier_residual[tied_offsets])]
        chunk_best_score = scores[tied_best_local]
        chunk_best_mean_residual = mean_inlier_residual[tied_best_local]

        if (
            best_score is None
            or chunk_best_score > best_score
            or (
                chunk_best_score == best_score
                and chunk_best_mean_residual < best_mean_residual
            )
        ):
            best_score = chunk_best_score
            best_mean_residual = chunk_best_mean_residual
            best_idx = start + tied_best_local
            best_inliers = inliers[tied_best_local]

    best_R = R[best_idx]
    best_t = t[best_idx]

    # Optional Refinement step over all inliers of the best model
    if best_inliers.sum() >= 3:
        inlier_pts1 = corr_pts1[best_inliers]
        inlier_pts2 = corr_pts2[best_inliers]

        c1_ref = inlier_pts1.mean(dim=0, keepdim=True)
        c2_ref = inlier_pts2.mean(dim=0, keepdim=True)
        X_ref = inlier_pts1 - c1_ref
        Y_ref = inlier_pts2 - c2_ref

        H_ref = X_ref.transpose(0, 1) @ Y_ref
        U_ref, _, Vt_ref = torch.linalg.svd(H_ref)

        R_ref = Vt_ref.transpose(0, 1) @ U_ref.transpose(0, 1)
        if torch.det(R_ref) < 0:
            Vt_ref[2, :] *= -1
            R_ref = Vt_ref.transpose(0, 1) @ U_ref.transpose(0, 1)

        t_ref = c2_ref.squeeze(0) - R_ref @ c1_ref.squeeze(0)
        return R_ref, t_ref

    return best_R, best_t


def save_pt_to_ply(pt_path, ply_path):
    # IMPORTANT: PyTorch >=2.6
    pts = torch.load(pt_path, weights_only=False)

    if torch.is_tensor(pts):
        pts = pts.detach().cpu().numpy()

    if pts.shape[0] == 3:
        pts = pts.T

    assert pts.ndim == 2 and pts.shape[1] == 3, f"Expected [N,3], got {pts.shape}"

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"Saved: {ply_path}")
