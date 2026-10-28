# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm

from novasplat.datasets.bop import BOPValDataset
from novasplat.datasets.image_mask_dataset import ImageMaskDepthDataset
from novasplat.models.dinov3_extractor import DINOv3Extractor
from novasplat.models.gedi_extractor import GeDiExtractor
from novasplat.utils.camera import (
    backproject_rgbd,
    compute_pca_diameter,
    compute_xyz,
    points_to_pcd,
)
from novasplat.utils.pcl import load_point_cloud
from novasplat.utils.transforms import (
    inv_resize_to_patches,
    normalize_images,
    resize_to_patches,
)

PATCH_SIZE = 16
IMAGE_SIZE = 480
MASK_FG_THRESHOLD: float = 0.5

MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}

GEDI_CONFIG = {
    "dim": 32,  # descriptor output dimension per scale
    "samples_per_batch": 500,  # batches to process the data on GPU
    "samples_per_patch_lrf": 1024,  # num. of points to process with LRF
    "samples_per_patch_out": 512,  # num. of points to sample for pointnet++
    # r_lrf can be a single float or a list of floats for multi-scale extraction.
    # Each value is a fraction of the object diameter (e.g. [0.15, 0.25]).
    # Descriptors from all scales are concatenated: dim = 32 * len(r_lrf).
    "r_lrf": [0.15, 0.25],
    # path to checkpoint
    "fchkpt_gedi_net": "thirdparty/gedi/data/chkpts/3dmatch/chkpt.tar",
    "num_keypoints": 5000,  # number of keypoints to extract
    "voxel_size": 0.1,  # voxel size for downsampling
}


logger = logging.getLogger(__name__)


# Canonical production feature settings, shared by onboarding, template-only
# feature extraction, and inference. Keep run_feature_debug.sh aligned with these.
PROD_NUM_LAST_DINO_LAYERS = 4
PROD_DINO_AGG_K = 1
PROD_DINO_AGG_DIST_THR = 0.03


def resolve_dinov3_weights(model_name: str) -> str | None:
    """Resolve a local DINOv3 checkpoint path when available.

    Search order:
    1) Explicit env vars: DINOV3_WEIGHTS, DINOV3_CHECKPOINT
    2) Known local model cache under thirdparty/cnos25/models
    """
    for env_name in ("DINOV3_WEIGHTS", "DINOV3_CHECKPOINT"):
        env_path = os.environ.get(env_name)
        if env_path and Path(env_path).is_file():
            return env_path

    local_models_dir = Path("thirdparty/cnos25/models")
    if local_models_dir.is_dir():
        matches = sorted(local_models_dir.glob(f"{model_name}_pretrain_*.pth"))
        if matches:
            return str(matches[0])

    return None


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "images": [b["image"] for b in batch],  # PIL
        "masks": [b["mask"] for b in batch],  # PIL, single-channel
        "depths": [b["depth"] for b in batch],  # np array
        "paths": [b["image_path"] for b in batch],
        "poses": [b["pose"] for b in batch],
        "intrinsics": [b["intrinsics"] for b in batch],
        "bboxes": [b["bbox"] for b in batch],
    }


def aggregate_features_to_keypoints(
    points_1: torch.Tensor,  # [N1, 3] (gedi_kp)
    points_2: torch.Tensor,  # [N2, 3] (XYZ)
    feats_2: torch.Tensor,  # [N2, C]
    k: int = 5,
    distance_threshold: float | None = None,
    use_topk: bool = True,
):
    """
    Aggregate features from points_2 -> points_1 using nearest neighbors.

    Returns:
        aggregated_feats: [N1, C]
        counts_raw: [N1]   real number of valid neighbors per keypoint
    """
    if points_2.shape[0] == 0:
        return (
            torch.zeros(
                points_1.shape[0],
                feats_2.shape[1],
                device=points_1.device,
                dtype=feats_2.dtype,
            ),
            torch.zeros(
                points_1.shape[0],
                device=points_1.device,
                dtype=feats_2.dtype,
            ),
        )

    feats_2 = F.normalize(feats_2, p=2, dim=1)

    dists = torch.cdist(points_1, points_2)  # [N1, N2]

    if use_topk:
        k_eff = min(k, points_2.shape[0])
        knn_dists, knn_idx = torch.topk(dists, k=k_eff, largest=False, dim=1)
    else:
        knn_dists, knn_idx = dists.min(dim=1, keepdim=True)

    if distance_threshold is not None:
        valid_mask = knn_dists < distance_threshold
    else:
        valid_mask = torch.ones_like(knn_dists, dtype=torch.bool)

    gathered_feats = feats_2[knn_idx]  # [N1, k_eff, C]

    valid_mask_f = valid_mask.unsqueeze(-1).to(gathered_feats.dtype)  # [N1, k_eff, 1]
    gathered_feats = gathered_feats * valid_mask_f

    summed = gathered_feats.sum(dim=1)  # [N1, C]
    counts_raw = valid_mask_f.sum(dim=1).squeeze(-1)  # [N1]
    counts_safe = counts_raw.clamp(min=1.0)

    aggregated_feats = summed / counts_safe.unsqueeze(1)
    return aggregated_feats, counts_raw


def collect_sparse_patch_keypoints(
    features_dict: dict[int, dict[str, Any]],
    max_keypoints: int | None = None,
):
    """Collect sparse patch-center 3D points across all frames.

    This is useful when we want GeDi keypoints to be anchored to the same
    sparse DINO patch geometry on both onboarding and eval data.
    """
    keypoints_list: list[torch.Tensor] = []

    for frame_idx in sorted(features_dict.keys()):
        xyz_sparse = compute_xyz(features_dict, frame_idx, patch=True)
        sparse_valid_mask = (
            features_dict[frame_idx]["mask"] * (features_dict[frame_idx]["depth"] > 0)
        )[PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE].flatten()
        xyz_sparse = xyz_sparse[sparse_valid_mask > 0]

        if xyz_sparse.numel() == 0:
            continue

        keypoints_list.append(xyz_sparse.detach().cpu())

    if not keypoints_list:
        return np.empty((0, 3), dtype=np.float32)

    keypoints = torch.cat(keypoints_list, dim=0)

    if max_keypoints is not None and keypoints.shape[0] > max_keypoints:
        # Deterministic subsampling keeps the experiment reproducible.
        keep_idx = (
            torch.linspace(
                0,
                keypoints.shape[0] - 1,
                steps=max_keypoints,
                dtype=torch.float32,
            )
            .round()
            .long()
        )
        keypoints = keypoints[keep_idx]

    return keypoints.numpy().astype(np.float32, copy=False)


def extract_dino_features(
    model_path: Path,
    image_dir: Path,
    model_name: str = "dinov3_vitl16",
    mask_dir: Optional[Path] = None,
    depth_dir: Optional[Path] = None,
    camera_params_path: Optional[Path] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    samples: Optional[list[dict[str, Any]]] = None,
    num_last_dino_layers: int = 1,
    extractor: Optional[DINOv3Extractor] = None,
) -> tuple[dict[int, dict[str, Any]], ImageMaskDepthDataset]:
    """
    Extract dense DINOv3 features for a set of images
    (and optional masks/depth).

    Parameters
    ----------
    model_path:
        Path to the checkpoint or weights for the DINOv3 model.
    image_dir:
        Directory containing the input RGB images.
    model_name:
        Optional DINOv3 model variant to use (default: ``dinov3_vitl16``).
    mask_dir:
        Optional directory with per-image foreground masks.
    depth_dir:
        Optional directory with per-image depth maps.
    camera_params_path:
        Optional path to camera intrinsics/poses used for 3D-aware features.
    batch_size:
        Batch size for the internal DataLoader.
    num_workers:
        Number of worker processes for the internal DataLoader.
    num_last_dino_layers:
        Number of final DINOv3 transformer layers to concatenate per patch.
        Using more than one layer produces richer descriptors at the cost of
        higher dimensionality and memory usage.
    Returns
    -------
    tuple[dict, ImageMaskDepthDataset]
        A tuple ``(features, dataset)`` where:

        * ``features`` is a dictionary mapping image paths (or identifiers) to
          their extracted dense feature tensors and any associated metadata.
        * ``dataset`` is the underlying :class:`ImageMaskDepthDataset` instance
          used to build the DataLoader.

    Notes
    -----
    The dataset is returned in addition to the feature dictionary to allow
    callers to perform additional processing (e.g. re-iterating over images,
    masks, or depth maps) and for backward compatibility with existing code
    that may rely on accessing the dataset directly, even if some internal
    callers do not currently use it.
    """
    logger.info(
        (
            "Initializing ImageMaskDataset with images from %s, "
            "masks from %s, and depth from %s"
        ),
        image_dir,
        mask_dir,
        depth_dir,
    )
    if samples is not None:
        logger.info("Initializing ImageMaskDepthDataset from explicit samples")
        dataset = ImageMaskDepthDataset(samples=samples)
    else:
        logger.info(
            (
                "Initializing ImageMaskDataset with images from %s, "
                "masks from %s, and depth from %s"
            ),
            image_dir,
            mask_dir,
            depth_dir,
        )
        dataset = ImageMaskDepthDataset(
            image_dir, mask_dir, depth_dir, camera_params_path
        )

    logger.info(
        "Creating DataLoader: batch_size=%d, num_workers=%d",
        batch_size,
        num_workers,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    if model_name not in MODEL_TO_NUM_LAYERS:
        supported_models = ", ".join(MODEL_TO_NUM_LAYERS.keys())
        logger.error(
            "Unsupported model_name '%s'. Supported model names are: %s",
            model_name,
            supported_models,
        )
        raise ValueError(
            "Unsupported model_name '%s'. Supported model names are: %s"
            % (model_name, supported_models)
        )
    if extractor is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading DINOv3 model '%s' from %s to %s",
            model_name,
            model_path,
            device,
        )
        weights = resolve_dinov3_weights(model_name)
        if weights is not None:
            logger.info("Using local DINOv3 weights: %s", weights)
            model = torch.hub.load(
                str(model_path),
                model_name,
                source="local",
                weights=weights,
            )
        else:
            logger.warning(
                "No local DINOv3 weights found; falling back to upstream download."
            )
            model = torch.hub.load(str(model_path), model_name, source="local")
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "Expected torch.hub.load to return torch.nn.Module, "
                "got %s" % type(model)
            )

        n_layers = MODEL_TO_NUM_LAYERS[model_name]
        logger.info(
            "Initializing DINOv3Extractor with %d total layers and %d output layers",
            n_layers,
            num_last_dino_layers,
        )
        extractor = DINOv3Extractor(model, n_layers, num_last_dino_layers, device)
    else:
        # Reuse the preloaded extractor's model device for input tensors.
        device = next(extractor.model.parameters()).device
        logger.debug("Using preloaded DINOv3Extractor on %s", device)

    features_dict: dict[int, dict[str, Any]] = {}

    logger.info("Starting feature extraction on %d images", len(dataset))
    with torch.inference_mode():
        idx = 0
        for batch in tqdm(loader, desc="Extracting features"):
            images = resize_to_patches(
                batch["images"],
                IMAGE_SIZE,
                PATCH_SIZE,
            )
            images = normalize_images(images)
            images = images.to(device, non_blocking=True)

            masks = resize_to_patches(
                batch["masks"],
                IMAGE_SIZE,
                PATCH_SIZE,
                TF.InterpolationMode.NEAREST,
            )
            depths = resize_to_patches(
                batch["depths"],
                IMAGE_SIZE,
                PATCH_SIZE,
                TF.InterpolationMode.NEAREST,
            )
            feats = extractor.extract(images)
            bsz = feats.shape[0]
            for b in range(bsz):
                features_dict[idx] = {
                    "features": feats[b].cpu(),
                    "image_path": batch["paths"][b],
                    "bbox": batch["bboxes"][b],
                    "pose": batch["poses"][b],
                    "intrinsics": batch["intrinsics"][b],
                    "mask": masks[b].squeeze(),  # batch["masks"][b],
                    "depth": depths[b].squeeze(),  # batch["depths"][b],
                    "cropped_image": batch["images"][b],
                    "resized_image": images[b].cpu(),
                }
                idx += 1
    logger.info("Extracted features for %d images.", len(features_dict))
    return features_dict, dataset


def save_features(
    output_path: Union[str, Path],
    features_dict: Any,
):
    """Save dense feature maps, patch masks, and image paths to disk.
    Args:
        output_path: Filesystem path where the output file will be written.
            This can be any path accepted by :class:`pathlib.Path`.
        features_dict: Dictionary containing extracted features and associated
            metadata.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving features to %s", output_path)
    torch.save(features_dict, output_path)
    logger.info("Features saved successfully.")


def _normalize_point_cloud_global(
    points: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Center and scale a point cloud to a unit sphere.

    Returns normalized points, centroid, and scalar scale so callers can invert
    keypoints back to the original metric coordinate system.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "Expected points with shape [N, 3] for global normalization, "
            f"got {tuple(points.shape)}"
        )

    centroid = points.mean(dim=0, keepdim=True)
    centered = points - centroid
    scale = torch.linalg.norm(centered, dim=1).max().item()
    if scale <= eps:
        raise ValueError(
            "Point cloud scale is too small for stable global normalization "
            f"(scale={scale:.3e})."
        )

    normalized = centered / scale
    return normalized, centroid.squeeze(0), float(scale)


def _to_normalized_keypoints(
    keypoints: Any,
    centroid: torch.Tensor,
    scale: float,
) -> Any:
    """Map keypoints from metric coordinates to normalized coordinates."""
    kp_t = torch.as_tensor(keypoints, device=centroid.device, dtype=torch.float32)
    kp_norm = (kp_t - centroid.unsqueeze(0)) / scale
    return kp_norm.cpu().numpy().astype(np.float32, copy=False)


def _to_metric_keypoints(
    keypoints: Any,
    centroid: torch.Tensor,
    scale: float,
) -> Any:
    """Map keypoints from normalized coordinates back to metric coordinates."""
    kp_t = torch.as_tensor(keypoints, device=centroid.device, dtype=torch.float32)
    kp_metric = kp_t * scale + centroid.unsqueeze(0)
    return kp_metric.cpu().numpy().astype(np.float32, copy=False)


def build_gedi_extractors() -> list["GeDiExtractor"]:
    """Pre-instantiate one GeDiExtractor per LRF scale.

    Loading the checkpoint once and reusing the same model instances across
    samples avoids repeated disk I/O and GPU memory re-allocation per sample.
    The actual per-sample radius (r_lrf_abs) is patched onto the extractor
    inside :func:`run_gedi_on_ply` before each call to ``extract``.
    """
    r_lrf_cfg = GEDI_CONFIG["r_lrf"]
    r_lrf_list: list[float] = (
        list(r_lrf_cfg) if isinstance(r_lrf_cfg, (list, tuple)) else [r_lrf_cfg]
    )
    extractors: list[GeDiExtractor] = []
    for _ in r_lrf_list:
        init_config = dict(GEDI_CONFIG)
        # r_lrf is a placeholder; the actual per-sample radius is set in
        # run_gedi_on_ply before each call to extract().
        init_config["r_lrf"] = 1.0
        extractors.append(GeDiExtractor(init_config))
    return extractors


def load_dino_extractor(
    dinov3_repo: Union[str, Path] = Path("thirdparty/dinov3"),
    model_name: str = "dinov3_vitl16",
    num_last_dino_layers: int = PROD_NUM_LAST_DINO_LAYERS,
    device: Optional[torch.device] = None,
) -> DINOv3Extractor:
    """Load a DINOv3 model once and wrap it in a :class:`DINOv3Extractor`.

    Callers that process many images/objects should build the extractor once
    and pass it into :func:`extract_dino_features` (or
    :func:`extract_and_fuse_features`) to avoid reloading the model per call.
    """
    if model_name not in MODEL_TO_NUM_LAYERS:
        supported = ", ".join(MODEL_TO_NUM_LAYERS.keys())
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Supported: {supported}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = resolve_dinov3_weights(model_name)
    if weights is not None:
        logger.info("Loading DINOv3 '%s' with local weights: %s", model_name, weights)
        model = torch.hub.load(
            str(dinov3_repo), model_name, source="local", weights=weights
        )
    else:
        logger.warning(
            "No local DINOv3 weights found; falling back to upstream download."
        )
        model = torch.hub.load(str(dinov3_repo), model_name, source="local")

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "Expected torch.hub.load to return torch.nn.Module, got %s" % type(model)
        )

    return DINOv3Extractor(
        model, MODEL_TO_NUM_LAYERS[model_name], num_last_dino_layers, device
    )


def run_gedi_on_ply(
    ply_path: Union[Path, np.ndarray[Any, Any], torch.Tensor],
    output_path: Path | None = None,
    keypoints: np.ndarray[Any, Any] | None = None,
    device: str = "cuda",
    global_normalize: bool = True,
    gedi_extractors: list["GeDiExtractor"] | None = None,
):
    """Extract multi-scale GeDi descriptors from a point cloud.

    ``GEDI_CONFIG["r_lrf"]`` may be a single float **or** a list of floats.
    When a list is provided, GeDi is run once per radius (each expressed as a
    fraction of the object diameter) using the **same** set of keypoints, and
    the resulting descriptors are concatenated along the feature dimension:

        descriptor_dim = 32 * len(r_lrf)

    For example, ``r_lrf = [0.15, 0.3]`` yields 64-dimensional descriptors.

    When ``global_normalize`` is True (default), the support cloud and input
    keypoints are globally centered/scaled to a unit sphere before running
    GeDi. Returned keypoints are then mapped back to the original metric frame
    so downstream fusion and pose estimation remain unchanged.
    """
    import time as _time
    _gedi_timings: dict[str, float] = {}
    shared_keypoints = keypoints
    if shared_keypoints is not None and len(shared_keypoints) == 0:
        logger.warning(
            "run_gedi_on_ply received empty keypoints array; "
            "falling back to internal sampling."
        )
        shared_keypoints = None

    _t_pcd = _time.perf_counter()
    if isinstance(ply_path, Path):
        pcd = load_point_cloud(ply_path, device)
        pcd.remove_non_finite_points()
        points = torch.from_numpy(np.asarray(pcd.points)).to(device)
    elif isinstance(ply_path, np.ndarray):
        points = torch.as_tensor(ply_path, device=device)
        # Filter non-finite points in torch directly
        finite_mask = torch.isfinite(points).all(dim=1)
        if not finite_mask.all():
            points = points[finite_mask]
    elif torch.is_tensor(ply_path):
        points = ply_path.to(device)
        # Filter non-finite points in torch directly
        finite_mask = torch.isfinite(points).all(dim=1)
        if not finite_mask.all():
            points = points[finite_mask]
    else:
        raise TypeError(
            f"ply_path must be a Path or numpy array or torch tensor, "
            f"got {type(ply_path)}"
        )
    _gedi_timings["pcd_creation"] = _time.perf_counter() - _t_pcd

    _t_norm = _time.perf_counter()
    if global_normalize:
        points_for_gedi, norm_center, norm_scale = _normalize_point_cloud_global(points)
        if shared_keypoints is not None:
            shared_keypoints = _to_normalized_keypoints(
                shared_keypoints,
                norm_center,
                norm_scale,
            )
    else:
        points_for_gedi = points
        norm_center = None
        norm_scale = None

    diameter = compute_pca_diameter(points_for_gedi)
    _gedi_timings["normalize_and_diameter"] = _time.perf_counter() - _t_norm

    # Normalise r_lrf to always be a list so the loop below is uniform.
    r_lrf_cfg = GEDI_CONFIG["r_lrf"]
    r_lrf_list: list[float]
    if isinstance(r_lrf_cfg, (list, tuple)):
        r_lrf_list = list(r_lrf_cfg)
    else:
        r_lrf_list = [r_lrf_cfg]

    all_descriptors: list[np.ndarray[Any, Any]] = []
    # keypoints: np.ndarray[Any, Any] | None = None  # shared across scales

    # Pre-compute voxel-downsampled support tensor ONCE — shared across all scales.
    # Pure PyTorch voxel downsample: avoids expensive Open3D PCD creation.
    _t_vox = _time.perf_counter()
    voxel_size = GEDI_CONFIG["voxel_size"]
    _pts_gedi = points_for_gedi.float()
    _voxel_ids = torch.div(_pts_gedi, voxel_size, rounding_mode="floor").long()
    # Encode 3D voxel indices into a unique 1D key
    _mins = _voxel_ids.min(dim=0).values
    _voxel_ids = _voxel_ids - _mins
    _maxs = _voxel_ids.max(dim=0).values + 1
    _keys = (
        _voxel_ids[:, 0] * (_maxs[1] * _maxs[2])
        + _voxel_ids[:, 1] * _maxs[2]
        + _voxel_ids[:, 2]
    )
    _unique_keys, _inverse = torch.unique(_keys, return_inverse=True)
    # Scatter-mean: average points within each voxel
    _n_voxels = _unique_keys.shape[0]
    _sum = torch.zeros(_n_voxels, 3, device=_pts_gedi.device, dtype=_pts_gedi.dtype)
    _count = torch.zeros(_n_voxels, device=_pts_gedi.device, dtype=_pts_gedi.dtype)
    _sum.scatter_add_(0, _inverse.unsqueeze(1).expand(-1, 3), _pts_gedi)
    _count.scatter_add_(0, _inverse, torch.ones(_pts_gedi.shape[0], device=_pts_gedi.device, dtype=_pts_gedi.dtype))
    _support_tensor = (_sum / _count.unsqueeze(1)).cuda()
    _gedi_timings["voxel_downsample"] = _time.perf_counter() - _t_vox

    # Create a lazy PCD only if fallback keypoint sampling is needed
    # (i.e., when shared_keypoints is None and GeDi needs to sample from the cloud)
    _pcd_for_gedi_lazy = None
    def _get_pcd_for_gedi():
        nonlocal _pcd_for_gedi_lazy
        if _pcd_for_gedi_lazy is None:
            _pcd_for_gedi_lazy = points_to_pcd(points_for_gedi, device=device)
        return _pcd_for_gedi_lazy

    for scale_idx, r_lrf_frac in enumerate(r_lrf_list):
        _t_scale = _time.perf_counter()
        r_lrf_abs = r_lrf_frac * diameter
        logger.debug(
            "Scale %d/%d: r_lrf_frac=%.3f -> r_lrf_abs=%.4f",
            scale_idx + 1,
            len(r_lrf_list),
            r_lrf_frac,
            r_lrf_abs,
        )

        gedi_config = dict(GEDI_CONFIG)
        gedi_config["r_lrf"] = r_lrf_abs

        if gedi_extractors is not None:
            # Reuse pre-built extractor: patch the per-sample radius in-place
            # to avoid reloading the checkpoint on every sample.
            gedi_extractor = gedi_extractors[scale_idx]
            gedi_extractor.model.r_lrf = r_lrf_abs
            gedi_extractor.model.lrf.r_lrf = r_lrf_abs
        else:
            gedi_extractor = GeDiExtractor(gedi_config)

        try:
            descs, kps = gedi_extractor.extract(
                _get_pcd_for_gedi(), shared_keypoints,
                precomputed_support=_support_tensor,
            )
            if shared_keypoints is not None:
                if not np.allclose(kps, shared_keypoints):
                    max_diff = float(np.max(np.abs(kps - shared_keypoints)))
                    raise ValueError(
                        "GeDi returned mismatched keypoints at scale "
                        f"{scale_idx + 1}/{len(r_lrf_list)}: "
                        f"expected shape {shared_keypoints.shape}, got {kps.shape}, "
                        f"max abs diff {max_diff:.6g}"
                    )
        except RuntimeError as exc:
            logger.warning(
                (
                    "Scale %d: failed with r_lrf_abs=%.4f, "
                    "retrying with r_lrf=10000. Error: %s"
                ),
                scale_idx + 1,
                r_lrf_abs,
                exc,
            )
            gedi_extractor.model.r_lrf = 10000.0
            gedi_extractor.model.lrf.r_lrf = 10000.0
            if gedi_extractors is None:
                gedi_extractor = GeDiExtractor({**gedi_config, "r_lrf": 10000.0})
            descs, kps = gedi_extractor.extract(
                _get_pcd_for_gedi(), shared_keypoints,
                precomputed_support=_support_tensor,
            )
        _gedi_timings[f"scale_{scale_idx}_extract"] = _time.perf_counter() - _t_scale

        if shared_keypoints is None:
            # Pin keypoints from the first scale; subsequent scales will
            # reuse the same spatial locations via the extractor's sampling.
            shared_keypoints = kps

        all_descriptors.append(descs)

    if shared_keypoints is None:
        raise RuntimeError(
            "Multi-scale GeDi extraction completed without producing keypoints."
        )

    if global_normalize:
        if norm_center is None or norm_scale is None:
            raise RuntimeError("Missing normalization transform for GeDi inversion.")
        shared_keypoints = _to_metric_keypoints(
            shared_keypoints, norm_center, norm_scale
        )

    # Concatenate along the feature dimension: [N, 32*n_scales]
    descriptors = np.concatenate(all_descriptors, axis=1)
    logger.debug(
        "Multi-scale GeDi: %d scales, descriptor shape %s, keypoints %s",
        len(r_lrf_list),
        descriptors.shape,
        shared_keypoints.shape,
    )

    # Log internal GeDi timing breakdown
    timing_parts = [f"{k}={v:.3f}s" for k, v in _gedi_timings.items()]
    logger.warning("GeDi internal: %s", ", ".join(timing_parts))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "keypoints": shared_keypoints,
                "descriptors": descriptors,
            },
            output_path,
        )
        logger.debug("Saved %d GeDi descriptors to %s", len(descriptors), output_path)
    return descriptors, shared_keypoints


def fuse_features(
    features_dict: dict[int, dict[str, Any]],
    gedi_descriptors: Union[np.ndarray[Any, Any], torch.Tensor],
    gedi_keypoints: Union[np.ndarray[Any, Any], torch.Tensor],
    distance_threshold: float = 0.15,
    dino_agg_k: int = 10,
    min_dino_support_for_fusion: float = 1.0,
    min_dino_agreement_for_fusion: float = 0.2,
    max_points_per_frame: int | None = None,
):
    """
    Fuse DINOv3 image features with GeDi 3D descriptors.

    For each backprojected RGB-D point, this function finds the nearest GeDi
    keypoint in 3D and accumulates the corresponding DINOv3 features if the
    3D distance between the point and the keypoint is within a specified
    threshold.

    Args:
        features_dict: A dictionary containing per-image data such as dense
            DINOv3 feature maps, depth maps, masks, and backprojected 3D
            point coordinates. The exact structure follows the outputs of the
            feature extraction pipeline in this module.
        gedi_descriptors: GeDi descriptors associated with the 3D keypoints.
            Can be a NumPy array or a torch tensor.
        gedi_keypoints: 3D coordinates of GeDi keypoints. Can be a NumPy
            array or a torch tensor with shape ``[N_k, 3]`` in world units.
        distance_threshold: Maximum allowed 3D distance between a backprojected
            point and a GeDi keypoint for them to be considered a match.
            This value is **relative** to the object diameter: the provided
            threshold (default ``0.15``) is multiplied by the estimated object
            diameter to obtain the absolute distance threshold in world units.
            Callers who wish to specify an absolute threshold should divide
            their desired distance by the object diameter before passing it.
        dino_agg_k: Number of nearest RGB-D patch descriptors aggregated per
            GeDi keypoint and per frame. Smaller values preserve local texture
            cues better; larger values provide smoother but less distinctive
            descriptors.
        min_dino_support_for_fusion: Minimum accumulated neighborhood support
            required for keeping a fused DINO descriptor at a GeDi keypoint.
        min_dino_agreement_for_fusion: Minimum cross-frame directional
            agreement required for keeping a fused DINO descriptor. The score
            is in ``[0, 1]`` where higher means descriptors aggregated from
            different frames point to similar directions.
        max_points_per_frame: Optional cap on the number of valid dense points
            used per frame for DINO aggregation. When provided, points are
            deterministically subsampled to keep memory bounded.
    """

    if dino_agg_k <= 0:
        raise ValueError(f"dino_agg_k must be > 0, got {dino_agg_k}")
    if distance_threshold <= 0:
        raise ValueError(f"distance_threshold must be > 0, got {distance_threshold}")
    if min_dino_support_for_fusion < 0:
        raise ValueError(
            "min_dino_support_for_fusion must be >= 0, got "
            f"{min_dino_support_for_fusion}"
        )
    if max_points_per_frame is not None and max_points_per_frame <= 0:
        raise ValueError(
            "max_points_per_frame must be > 0 when provided, got "
            f"{max_points_per_frame}"
        )
    if not (0.0 <= min_dino_agreement_for_fusion <= 1.0):
        raise ValueError(
            "min_dino_agreement_for_fusion must be in [0, 1], got "
            f"{min_dino_agreement_for_fusion}"
        )

    if isinstance(gedi_keypoints, np.ndarray):
        gedi_keypoints = torch.from_numpy(gedi_keypoints)
    elif torch.is_tensor(gedi_keypoints):
        gedi_keypoints = gedi_keypoints.detach()
    else:
        raise TypeError(
            f"gedi_keypoints must be a numpy array or torch tensor, "
            f"got {type(gedi_keypoints)}"
        )

    if isinstance(gedi_descriptors, np.ndarray):
        gedi_descriptors = torch.from_numpy(gedi_descriptors)
    elif torch.is_tensor(gedi_descriptors):
        gedi_descriptors = gedi_descriptors.detach()
    else:
        raise TypeError(
            f"gedi_descriptors must be a numpy array or torch tensor, "
            f"got {type(gedi_descriptors)}"
        )
    num_keypoints = gedi_descriptors.shape[0]
    dino_dim = features_dict[0]["features"].shape[0]

    dinov3_features = torch.zeros(num_keypoints, dino_dim)
    dinov3_features_count = torch.zeros(num_keypoints)
    dinov3_direction_sum = torch.zeros(num_keypoints, dino_dim)
    dinov3_direction_weight = torch.zeros(num_keypoints)

    diameter = compute_pca_diameter(gedi_keypoints)

    distance_threshold = diameter * distance_threshold
    logger.debug(
        "Object diameter: %.2f, setting distance threshold = %.2f",
        diameter,
        distance_threshold,
    )

    for frame_data in features_dict.values():
        bbox = frame_data["bbox"]
        x0, y0 = bbox[:2]
        resized_shape = (
            frame_data["resized_image"].shape[1:]
            if "resized_image" in frame_data
            else frame_data["resized_image_shape"]
        )
        pu, pv = inv_resize_to_patches(
            (bbox[2] - bbox[0], bbox[3] - bbox[1]),
            resized_shape,
            PATCH_SIZE,
        )
        pu = pu + y0
        pv = pv + x0
        depth = frame_data["depth"]
        mask = frame_data["mask"]
        intrinsics = frame_data["intrinsics"]
        pose = frame_data["pose"]

        pu = pu[PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE].flatten()
        pv = pv[PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE].flatten()
        mask = mask[PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE]
        depth = depth[PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE]

        valid = depth.flatten() * mask.flatten() > 0
        if valid.sum() == 0:
            continue

        XYZ = backproject_rgbd(
            pv.flatten()[valid],
            pu.flatten()[valid],
            depth.flatten()[valid],
            intrinsics,
            pose,
        )

        if XYZ.shape[0] == 0:
            continue

        feat_a = frame_data["features"]
        feat_flat = feat_a.flatten(1).T
        feats_valid = feat_flat[valid]

        if max_points_per_frame is not None and XYZ.shape[0] > max_points_per_frame:
            keep_idx = (
                torch.linspace(
                    0,
                    XYZ.shape[0] - 1,
                    steps=max_points_per_frame,
                    dtype=torch.float32,
                    device=XYZ.device,
                )
                .round()
                .long()
            )
            XYZ = XYZ[keep_idx]
            feats_valid = feats_valid[keep_idx]

        gedi_kp = gedi_keypoints.to(device=XYZ.device, dtype=XYZ.dtype)

        aggregated_feats, counts = aggregate_features_to_keypoints(
            gedi_kp,
            XYZ,
            feats_valid,
            k=dino_agg_k,
            distance_threshold=distance_threshold,
        )

        dinov3_features += aggregated_feats * counts.unsqueeze(1)
        dinov3_features_count += counts

        valid_local = counts > 0
        if valid_local.any():
            local_dirs = F.normalize(aggregated_feats[valid_local], p=2, dim=1)
            local_weights = counts[valid_local]
            dinov3_direction_sum[valid_local] += local_dirs * local_weights.unsqueeze(1)
            dinov3_direction_weight[valid_local] += local_weights

    raw_dinov3_features_count = dinov3_features_count.clone()
    valid_global = dinov3_features_count > 0
    dinov3_features[valid_global] = dinov3_features[
        valid_global
    ] / dinov3_features_count[valid_global].unsqueeze(1)

    dinov3_agreement = torch.zeros(num_keypoints)
    valid_direction = dinov3_direction_weight > 0
    if valid_direction.any():
        dinov3_agreement[valid_direction] = (
            torch.norm(dinov3_direction_sum[valid_direction], dim=1)
            / dinov3_direction_weight[valid_direction]
        )

    fusion_confident_mask = (
        raw_dinov3_features_count >= min_dino_support_for_fusion
    ) & (dinov3_agreement >= min_dino_agreement_for_fusion)
    dinov3_features[~fusion_confident_mask] = 0
    dinov3_features_count = raw_dinov3_features_count * fusion_confident_mask.float()

    # return dinov3_features
    gedi_descriptors = F.normalize(gedi_descriptors, p=2, dim=1)
    dinov3_features = F.normalize(dinov3_features, p=2, dim=1)

    fused_output = {
        "descriptors": torch.cat(
            [gedi_descriptors, dinov3_features],
            dim=1,
        ),
        "keypoints": gedi_keypoints,
        # Raw per-keypoint support before global normalization. Useful to
        # filter unstable DINO descriptors at visualization/matching time.
        "dinov3_features_count": dinov3_features_count,
        "dinov3_features_count_raw": raw_dinov3_features_count,
        "dinov3_agreement": dinov3_agreement,
        "fusion_confident_mask": fusion_confident_mask,
    }
    logger.info(
        "Fusion confidence gate kept %d/%d keypoints "
        "(min_support=%.2f, min_agreement=%.2f)",
        int(fusion_confident_mask.sum().item()),
        fusion_confident_mask.numel(),
        min_dino_support_for_fusion,
        min_dino_agreement_for_fusion,
    )
    logger.debug("Fused features shape: %s", fused_output["descriptors"].shape)
    return fused_output


def extract_and_fuse_features(
    template_dir: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    dinov3_repo: Union[str, Path] = Path("thirdparty/dinov3"),
    model_name: str = "dinov3_vitl16",
    num_last_dino_layers: int = PROD_NUM_LAST_DINO_LAYERS,
    dino_agg_k: int = PROD_DINO_AGG_K,
    dino_agg_dist_thr: float = PROD_DINO_AGG_DIST_THR,
    min_dino_support_for_fusion: float = 1.0,
    min_dino_agreement_for_fusion: float = 0.2,
    batch_size: int = 8,
    num_workers: int = 4,
    dino_extractor: Optional[DINOv3Extractor] = None,
    gedi_extractors: Optional[list["GeDiExtractor"]] = None,
) -> dict[str, Any]:
    """Run the DINOv3 + GeDi + fusion pipeline for a single template object.

    Expects the standard rendered-template layout under ``template_dir``::

        rgb/                     RGB renders
        mask_visib/              foreground masks
        depth/*.npz              per-frame depth maps
        rendered_camera_params.npz   intrinsics + per-frame poses
        model.ply                object point cloud (GeDi support)

    Writes ``dinov3.pt``, ``gedi.pt`` and ``fused.pt`` into ``output_dir`` and
    returns the fused feature dict. Pass ``dino_extractor`` / ``gedi_extractors``
    to reuse preloaded models across many objects.
    """
    template_dir = Path(template_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting DINOv3 features for %s", template_dir)
    features_dict, _ = extract_dino_features(
        Path(dinov3_repo),
        template_dir / "rgb",
        model_name,
        template_dir / "mask_visib",
        template_dir / "depth",
        template_dir / "rendered_camera_params.npz",
        batch_size=batch_size,
        num_workers=num_workers,
        num_last_dino_layers=num_last_dino_layers,
        extractor=dino_extractor,
    )
    save_features(output_dir / "dinov3.pt", features_dict)

    logger.info("Extracting GeDi descriptors for %s", template_dir / "model.ply")
    gedi_descriptors, gedi_keypoints = run_gedi_on_ply(
        template_dir / "model.ply",
        output_dir / "gedi.pt",
        gedi_extractors=gedi_extractors,
    )

    logger.info("Fusing DINOv3 and GeDi features")
    fused_features = fuse_features(
        features_dict,
        gedi_descriptors,
        gedi_keypoints,
        distance_threshold=dino_agg_dist_thr,
        dino_agg_k=dino_agg_k,
        min_dino_support_for_fusion=min_dino_support_for_fusion,
        min_dino_agreement_for_fusion=min_dino_agreement_for_fusion,
    )
    save_features(output_dir / "fused.pt", fused_features)
    return fused_features


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3_model_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--pcl_path", type=str)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dinov3_vitl16")
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--depth_dir", type=str, default=None)
    parser.add_argument("--camera_params_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["onboarding", "eval"],
        required=True,
        help="onboarding: process custom data | eval: run BOP validation pipeline",
    )
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--val_idx", type=int, default=0)
    parser.add_argument(
        "--onboarding_use_sparse_gedi_keypoints",
        action="store_true",
        help=(
            "In onboarding mode, use sparse patch-center 3D points as fixed "
            "GeDi keypoints so the anchor strategy matches eval mode."
        ),
    )
    parser.add_argument(
        "--num_last_dino_layers",
        type=int,
        default=1,
        help=(
            "Number of final DINOv3 layers to concatenate into each patch "
            "descriptor."
        ),
    )
    parser.add_argument(
        "--dino_agg_k",
        type=int,
        default=10,
        help=(
            "Number of nearest RGB-D patch descriptors aggregated into each "
            "GeDi keypoint during DINO fusion. Lower values preserve local "
            "texture cues better (e.g., 1-3)."
        ),
    )
    parser.add_argument(
        "--dino_agg_dist_thr",
        type=float,
        default=0.05,
        help=(
            "Relative 3D distance threshold for assigning RGB-D patch "
            "features to GeDi keypoints during DINO fusion. Smaller values "
            "reduce cross-surface mixing."
        ),
    )
    parser.add_argument(
        "--min_dino_support_for_fusion",
        type=float,
        default=1.0,
        help=(
            "Minimum accumulated local support required to keep a fused DINO "
            "descriptor at a GeDi keypoint."
        ),
    )
    parser.add_argument(
        "--min_dino_agreement_for_fusion",
        type=float,
        default=0.2,
        help=(
            "Minimum cross-frame descriptor agreement in [0, 1] required to "
            "keep a fused DINO descriptor at a GeDi keypoint."
        ),
    )

    args = parser.parse_args()

    if args.mode == "onboarding":
        if args.image_dir is None or args.pcl_path is None:
            parser.error("--image_dir and --pcl_path are required in onboarding mode.")
    elif args.mode == "eval":
        if args.data_dir is None:
            parser.error("--data_dir is required in eval mode.")

    logger.info("Starting DINOv3 feature extraction...")

    if args.mode == "onboarding":
        extracted_features, _ = extract_dino_features(
            Path(args.dinov3_model_path),
            Path(args.image_dir),
            args.model_name,
            Path(args.mask_dir) if args.mask_dir else None,
            Path(args.depth_dir) if args.depth_dir else None,
            Path(args.camera_params_path) if args.camera_params_path else None,
            args.batch_size,
            args.num_workers,
            num_last_dino_layers=args.num_last_dino_layers,
        )
    else:
        split = "val"
        bop_test = BOPValDataset(root_dir=args.data_dir, split=split)

        samples = []
        for i, sample in enumerate(bop_test):
            if i != args.val_idx:
                continue
            samples.append(
                {
                    "image_path": sample["image_path"],
                    "depth_path": sample["depth_path"],
                    "mask_path": sample["mask_path"],
                    "intrinsics": sample["intrinsics"],
                    "pose": None,
                    "obj_id": sample["obj_id"],
                    "scene_id": sample["scene_id"],
                    "im_id": sample["im_id"],
                }
            )
            break

        print("Object ID: ", sample["obj_id"])
        extracted_features, _ = extract_dino_features(
            Path(args.dinov3_model_path),
            Path(""),
            args.model_name,
            samples=samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_last_dino_layers=args.num_last_dino_layers,
        )

    save_features(
        Path(args.output_path) / "dinov3.pt",
        extracted_features,
    )
    logger.info("DINOv3 feature extraction completed.")

    logger.info("Starting GeDi descriptor extraction...")

    if args.mode == "onboarding":
        gedi_keypoints_onb = None
        if args.onboarding_use_sparse_gedi_keypoints:
            gedi_keypoints_onb = collect_sparse_patch_keypoints(
                extracted_features,
                max_keypoints=GEDI_CONFIG["num_keypoints"],
            )
            if gedi_keypoints_onb.shape[0] == 0:
                logger.warning(
                    "No valid sparse onboarding patch keypoints found; "
                    "falling back to GeDi internal keypoint sampling."
                )
                gedi_keypoints_onb = None
            else:
                logger.info(
                    "Onboarding GeDi uses sparse patch-center keypoints: %d",
                    gedi_keypoints_onb.shape[0],
                )

        extracted_gedi_descriptors, extracted_gedi_keypoints = run_gedi_on_ply(
            Path(args.pcl_path),
            Path(args.output_path) / "gedi.pt",
            keypoints=gedi_keypoints_onb,
        )
    else:
        fidx = 0
        XYZ_dense = compute_xyz(extracted_features, fidx, patch=False)
        mask_dense = (
            extracted_features[fidx]["mask"] * (extracted_features[fidx]["depth"] > 0)
        ).flatten()
        XYZ_dense = XYZ_dense[mask_dense > 0]

        if XYZ_dense.numel() == 0:
            sample_info = extracted_features[fidx]
            raise ValueError(
                "No valid dense points remain after applying mask/depth filtering "
                "for GeDi extraction in eval mode. "
                f"scene_id={sample_info.get('scene_id')}, "
                f"im_id={sample_info.get('im_id')}, "
                f"obj_id={sample_info.get('obj_id')}."
            )

        XYZ_sparse = compute_xyz(extracted_features, fidx, patch=True)
        mask_sparse = (
            extracted_features[fidx]["mask"] * (extracted_features[fidx]["depth"] > 0)
        )[PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE].flatten()
        XYZ_sparse = XYZ_sparse[mask_sparse > 0].cpu().numpy()
        logger.info(
            "Eval GeDi uses sparse patch-center keypoints: %d",
            XYZ_sparse.shape[0],
        )

        extracted_gedi_descriptors, extracted_gedi_keypoints = run_gedi_on_ply(
            XYZ_dense,
            Path(args.output_path) / "gedi.pt",
            keypoints=XYZ_sparse,
        )

    logger.info("GeDi descriptor extraction completed.")

    logger.info("Starting feature fusion...")

    fused_result = fuse_features(
        extracted_features,
        extracted_gedi_descriptors,
        extracted_gedi_keypoints,
        distance_threshold=args.dino_agg_dist_thr,
        dino_agg_k=args.dino_agg_k,
        min_dino_support_for_fusion=args.min_dino_support_for_fusion,
        min_dino_agreement_for_fusion=args.min_dino_agreement_for_fusion,
    )

    save_features(
        Path(args.output_path) / "fused.pt",
        fused_result,
    )

    logger.info("Feature fusion completed.")
