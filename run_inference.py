import argparse
import gc
import logging
import math
import os
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from tqdm import tqdm

from novasplat.datasets.bop import BOPValDataset
from novasplat.features.dense_features import (
    GEDI_CONFIG,
    MODEL_TO_NUM_LAYERS,
    build_gedi_extractors,
    extract_dino_features,
    fuse_features,
    resolve_dinov3_weights,
    run_gedi_on_ply,
    save_features,
)
from novasplat.models.dinov3_extractor import DINOv3Extractor
from novasplat.refinement import ICPRefiner, Refiner
from novasplat.refinement.optimizer import OptimizerConfig
from novasplat.utils.camera import PATCH_SIZE, compute_xyz
from novasplat.utils.pose_estimation import (
    compute_fused_similarity,
    compute_similarity,
    estimate_inlier_threshold,
    geometry_ransac_se3,
    get_top_matches,
    least_squares_se3,
)

logger = logging.getLogger(__name__)


# Keep production feature settings aligned with run_feature_debug.sh.
PROD_NUM_LAST_DINO_LAYERS = 4
PROD_DINO_AGG_K = 1
PROD_DINO_AGG_DIST_THR = 0.03
PROD_TOPK_MATCHES = 100


def _resolve_template_root(result_dir: Path, data_dir: Path) -> Path:
    """Resolve onboarding templates directory with robust fallback.

    Preference order:
    1) output/templates (templates produced by local onboarding run)
    2) dataset/templates (templates colocated under the dataset root)
    """
    candidates = [result_dir / "templates", data_dir / "templates"]

    for candidate in candidates:
        if not candidate.exists():
            continue
        # A valid template root should contain at least one fused template feature.
        if any(candidate.glob("obj_*/features/fused.pt")):
            logger.info("Using template root: %s", candidate)
            return candidate

    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find onboarding template features (expected obj_*/features/fused.pt). "
        f"Checked: {checked}. "
        "Run onboarding first, or point inference to the output directory containing templates."
    )


def _get_current_rss_mb() -> float | None:
    """Return current resident set size in MB on Linux, if available."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None

    try:
        with status_path.open("r", encoding="utf-8") as status_file:
            for line in status_file:
                if not line.startswith("VmRSS:"):
                    continue
                rss_kib = int(line.split()[1])
                return rss_kib / 1024.0
    except (OSError, ValueError):
        return None

    return None


def _load_template_features(obj_id: int, template_root: Path) -> dict[str, object]:
    """Load only the fused template data required by inference."""
    fused_path = template_root / f"obj_{obj_id:06d}" / "features" / "fused.pt"
    fused_data = torch.load(fused_path, weights_only=False)
    return {
        "descriptors": fused_data["descriptors"],
        "keypoints": fused_data["keypoints"],
    }


def parse_args():
    """
    Parse command line arguments for the Novasplat runner.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Run Novasplat")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to the input directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset name",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-s",
        "--split",
        help="Dataset split to run inference on (e.g., 'val', 'test')",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--scene-id",
        default=None,
        type=int,
        help="Restrict inference to a single BOP scene id for faster iteration.",
    )
    parser.add_argument(
        "--im-id",
        default=None,
        type=int,
        help="Restrict inference to a single image id.",
    )
    parser.add_argument(
        "--obj-id",
        default=None,
        type=int,
        help="Restrict inference to a single object id.",
    )
    parser.add_argument(
        "--instance-id",
        default=None,
        type=int,
        help="Restrict inference to a single GT instance id when available.",
    )
    parser.add_argument(
        "--sample-index",
        default=None,
        type=int,
        help=(
            "Restrict inference to the Nth sample after filtering. Useful when "
            "CNOS detection yields multiple samples for the same scene/image/object."
        ),
    )
    parser.add_argument(
        "--gedi-weight",
        default=0.2,
        type=float,
        help=(
            "Weight for GeDi in fused similarity: "
            "sim = w * sim_gedi + (1-w) * sim_dino. "
            "Use values in [0, 1]."
        ),
    )
    parser.add_argument(
        "--gedi-dim",
        default=None,
        type=int,
        help=(
            "Optional GeDi descriptor dimension in fused vectors. "
            "If not provided, derived from GEDI_CONFIG."
        ),
    )
    parser.add_argument(
        "--dino-chunk-size",
        default=32,
        type=int,
        help=(
            "Number of samples to process per extraction chunk. "
            "Lower values reduce peak RAM at the cost of extra overhead."
        ),
    )
    parser.add_argument(
        "--save-dino-features",
        action="store_true",
        help=(
            "Save extracted DINO features to disk. In chunked mode this writes "
            "one file per chunk under output/dino_chunks/."
        ),
    )
    parser.add_argument(
        "--use-cnos-detection",
        action="store_true",
        help="Use CNOS detection for candidate generation.",
    )
    parser.add_argument(
        "--max-dense-points",
        default=150000,
        type=int,
        help=(
            "Maximum dense valid points per frame used in GeDi/fusion paths. "
            "Deterministically subsamples when exceeded to avoid memory spikes."
        ),
    )
    parser.add_argument(
        "--max-samples",
        default=None,
        type=int,
        help=(
            "Optional cap on number of dataset samples to process. "
            "Useful for fast one-case debugging (e.g., --max-samples 1)."
        ),
    )
    parser.add_argument(
        "--sample-workers",
        default=None,
        type=int,
        help=(
            "Worker threads for per-sample GeDi/fusion/pose stage inside a chunk. "
            "Default: 1 on CUDA, min(4, CPU cores) on CPU-only runs."
        ),
    )
    parser.add_argument(
        "--max-inflight-samples",
        default=None,
        type=int,
        help=(
            "Upper bound on queued/running per-sample tasks per chunk. "
            "Default: 2 * sample_workers. Lower values reduce peak memory."
        ),
    )
    parser.add_argument(
        "--enable-refinement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable post-RANSAC photometric refinement with geometric guard "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--refinement-debug-dir",
        default=None,
        type=str,
        help=(
            "If set, save per-sample refinement debug output (step CSV, renders) "
            "under this directory. One sub-folder per sample."
        ),
    )
    parser.add_argument(
        "--refinement-geo-weight",
        default=0.001,
        type=float,
        help=(
            "Weight for the 3D Chamfer loss (model.ply vs query depth point cloud). "
            "Default 0.001. 0.0 disables it (photo-only); pair that with "
            "--refinement-epsilon-z 0.0."
        ),
    )
    parser.add_argument(
        "--refinement-epsilon-z",
        default=10.0,
        type=float,
        help=(
            "Max depth (z) correction in mm during refinement. Default 10.0mm. "
            "Set to 0.0 for photo-only refinement (--refinement-geo-weight 0.0), "
            "where the photometric loss would otherwise exploit z."
        ),
    )
    parser.add_argument(
        "--refinement-epsilon-xy",
        default=10.0,
        type=float,
        help=(
            "Max lateral (x/y) correction in mm during refinement. "
            "Default 10.0mm. Increase (e.g. 20.0) to allow larger lateral corrections."
        ),
    )
    parser.add_argument(
        "--refinement-silhouette-weight",
        default=0.0,
        type=float,
        help=(
            "Weight for the silhouette IoU loss (CNOS mask vs rendered alpha). "
            "Provides a centroid anchor to prevent lateral drift. "
            "Safe to enable (e.g. 0.1) when epsilon_z=0 — z-exploit cannot occur."
        ),
    )
    parser.add_argument(
        "--refinement-rotation-parametrization",
        default="rot6d",
        choices=["axis_angle", "rot6d"],
        help=(
            "Rotation parametrization for refinement. 'axis_angle' is the so(3) delta; "
            "'rot6d' (default) is the continuous 6D Gram-Schmidt delta. Both use "
            "the same rotation trust region (--refinement-epsilon-rotation-deg)."
        ),
    )
    parser.add_argument(
        "--refinement-epsilon-rotation-deg",
        type=float,
        default=10.0,
        help=(
            "Max angle (degrees) the refined rotation may deviate from the coarse pose. "
            "Default 10.0."
        ),
    )
    parser.add_argument(
        "--refiner",
        default="photometric",
        choices=["photometric", "icp"],
        help=(
            "Which refinement backend to use when --enable-refinement is set. "
            "'photometric' (default): gsplat-based photometric optimisation. "
            "'icp': Open3D point-to-point/point-to-plane ICP against the scene depth."
        ),
    )
    parser.add_argument(
        "--icp-max-correspondence-dist",
        default=50.0,
        type=float,
        help="ICP per-correspondence rejection threshold in mm (default: 50.0).",
    )
    parser.add_argument(
        "--icp-voxel-size",
        default=3.0,
        type=float,
        help="Voxel size in mm for ICP downsampling; 0 to disable (default: 3.0).",
    )
    parser.add_argument(
        "--icp-method",
        default="point_to_point",
        choices=["point_to_point", "point_to_plane"],
        help="ICP estimation method (default: point_to_point).",
    )
    return parser.parse_args()


def run(
    data_dir: Path,
    result_dir: Path,
    split: str,
    dataset_name: str,
    scene_id: int | None = None,
    im_id: int | None = None,
    obj_id: int | None = None,
    instance_id: int | None = None,
    sample_index: int | None = None,
    gedi_weight: float = 0.2,
    gedi_dim_override: int | None = None,
    dino_chunk_size: int = 32,
    save_dino_features: bool = False,
    max_dense_points: int = 150000,
    cnos_detection_file: Path | None = None,
    max_samples: int | None = None,
    sample_workers: int | None = None,
    max_inflight_samples: int | None = None,
    enable_refinement: bool = True,
    refinement_debug_dir: Path | None = None,
    refinement_geo_weight: float = 0.001,
    refinement_epsilon_z: float = 10.0,
    refinement_epsilon_xy: float = 10.0,
    refinement_silhouette_weight: float = 0.0,
    refinement_rotation_parametrization: str = "rot6d",
    refinement_epsilon_rotation_deg: float = 10.0,
    refiner_type: str = "photometric",
    icp_max_correspondence_dist: float = 50.0,
    icp_voxel_size: float = 3.0,
    icp_method: str = "point_to_point",
):
    """Run the Novasplat inference for a given data and result directory.

    Args:
        data_dir: The path to the data directory.
        result_dir: The path to the result directory.
        split: The dataset split to run inference on (e.g., 'val', 'test').
        dataset_name: The name of the dataset to run inference on.

    """
    if not (0.0 <= gedi_weight <= 1.0):
        raise ValueError(f"gedi_weight must be in [0, 1], got {gedi_weight}")
    if dino_chunk_size <= 0:
        raise ValueError(f"dino_chunk_size must be > 0, got {dino_chunk_size}")
    if max_dense_points <= 0:
        raise ValueError(f"max_dense_points must be > 0, got {max_dense_points}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be > 0 when set, got {max_samples}")
    if sample_workers is not None and sample_workers <= 0:
        raise ValueError(f"sample_workers must be > 0 when set, got {sample_workers}")
    if max_inflight_samples is not None and max_inflight_samples <= 0:
        raise ValueError(
            "max_inflight_samples must be > 0 when set, " f"got {max_inflight_samples}"
        )
    if sample_index is not None and sample_index < 0:
        raise ValueError(f"sample_index must be >= 0 when set, got {sample_index}")
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "log.txt"
    dataset_label = Path(dataset_name).name
    if dataset_label != dataset_name:
        logger.warning(
            "Dataset name '%s' looks path-like; using basename '%s' for output files.",
            dataset_name,
            dataset_label,
        )
    coarse_csv_path = result_dir / f"novasplat_{dataset_label}-{split}.csv"
    refined_csv_path = result_dir / f"novasplat_{dataset_label}-{split}_refined.csv"
    # Start each run with a clean per-sample error log.
    log_path.write_text("", encoding="utf-8")
    if coarse_csv_path.exists():
        coarse_csv_path.unlink()
    if refined_csv_path.exists():
        refined_csv_path.unlink()

    template_root = _resolve_template_root(result_dir, data_dir)

    # Keep tqdm bars readable by silencing per-sample INFO logs from feature fusion.
    logging.getLogger("novasplat.features.dense_features").setLevel(logging.WARNING)

    if gedi_dim_override is None:
        r_lrf = GEDI_CONFIG["r_lrf"]
        n_scales = len(r_lrf) if isinstance(r_lrf, (list, tuple)) else 1
        gedi_dim = GEDI_CONFIG["dim"] * n_scales
    else:
        gedi_dim = gedi_dim_override

    bop_test = BOPValDataset(
        root_dir=data_dir, split=split, cnos_detection_file=cnos_detection_file
    )
    samples = []
    logger.info("Running inference on %d samples...", len(bop_test))
    for sample in bop_test:
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
                "score": sample["score"] if "score" in sample else 1.0,
                "bbox": sample["bbox"] if "bbox" in sample else None,
                "segmentation": sample["segmentation"]
                if "segmentation" in sample
                else None,
                "instance_id": sample["instance_id"]
                if "instance_id" in sample
                else None,
                "gt_id": sample["gt_id"] if "gt_id" in sample else None,
            }
        )

    if scene_id is not None:
        original_count = len(samples)
        samples = [sample for sample in samples if sample["scene_id"] == scene_id]
        logger.info(
            "Limiting inference samples to scene %06d: %d -> %d",
            scene_id,
            original_count,
            len(samples),
        )

    if im_id is not None:
        original_count = len(samples)
        samples = [sample for sample in samples if sample["im_id"] == im_id]
        logger.info(
            "Limiting inference samples to image %06d: %d -> %d",
            im_id,
            original_count,
            len(samples),
        )

    if obj_id is not None:
        original_count = len(samples)
        samples = [sample for sample in samples if sample["obj_id"] == obj_id]
        logger.info(
            "Limiting inference samples to object %06d: %d -> %d",
            obj_id,
            original_count,
            len(samples),
        )

    if instance_id is not None:
        original_count = len(samples)
        samples = [
            sample
            for sample in samples
            if sample.get("instance_id") == instance_id
            or sample.get("gt_id") == instance_id
        ]
        logger.info(
            "Limiting inference samples to instance %06d: %d -> %d",
            instance_id,
            original_count,
            len(samples),
        )

    if sample_index is not None:
        if not samples:
            raise RuntimeError(
                "Cannot apply --sample-index after filtering because no samples remain."
            )
        if sample_index >= len(samples):
            raise IndexError(
                f"--sample-index {sample_index} is out of range for {len(samples)} filtered samples"
            )
        original_count = len(samples)
        samples = [samples[sample_index]]
        logger.info(
            "Limiting inference to sample index %d: %d -> %d",
            sample_index,
            original_count,
            len(samples),
        )

    if max_samples is not None:
        original_count = len(samples)
        samples = samples[:max_samples]
        logger.info(
            "Limiting inference samples: %d -> %d (--max-samples)",
            original_count,
            len(samples),
        )
    total_results_written = 0
    total_missing_features = 0
    total_no_dense = 0
    total_sparse_fallback = 0
    total_low_matches = 0
    total_errors = 0

    # Preload DINO extractor once to avoid reloading model every chunk.
    model_name = "dinov3_vitl16"
    local_weights = resolve_dinov3_weights(model_name)
    if local_weights is not None:
        dino_model = torch.hub.load(
            "thirdparty/dinov3",
            model_name,
            source="local",
            weights=local_weights,
        )
    else:
        dino_model = torch.hub.load("thirdparty/dinov3", model_name, source="local")
    if not isinstance(dino_model, torch.nn.Module):
        raise TypeError(
            "Expected torch.hub.load to return torch.nn.Module, got %s"
            % type(dino_model)
        )
    n_layers = MODEL_TO_NUM_LAYERS[model_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_extractor = DINOv3Extractor(
        dino_model,
        n_layers,
        PROD_NUM_LAST_DINO_LAYERS,
        device,
    )

    refiner = None
    if enable_refinement:
        try:
            if refiner_type == "icp":
                _icp_method: Literal["point_to_point", "point_to_plane"] = (
                    "point_to_plane"
                    if icp_method == "point_to_plane"
                    else "point_to_point"
                )
                refiner = ICPRefiner(
                    template_root=template_root,
                    device=device,
                    max_correspondence_dist_mm=icp_max_correspondence_dist,
                    voxel_size_mm=icp_voxel_size,
                    method=_icp_method,
                )
                logger.info("Refinement enabled (ICP).")
            else:
                refiner = Refiner(
                    result_dir=result_dir,
                    device=device,
                    debug_dir=refinement_debug_dir,
                    template_root=template_root,
                    optimizer_config=OptimizerConfig(
                        geo_weight=refinement_geo_weight,
                        epsilon_translation_z=refinement_epsilon_z,
                        epsilon_translation_xy=refinement_epsilon_xy,
                        silhouette_weight=refinement_silhouette_weight,
                        rotation_parametrization=refinement_rotation_parametrization,
                        epsilon_rotation_rad=math.radians(
                            refinement_epsilon_rotation_deg
                        ),
                    ),
                )
                logger.info("Refinement enabled (photometric).")
        except Exception as exc:
            logger.warning(
                "Failed to initialize refinement module (%s). Continuing with coarse-only inference.",
                exc,
            )
            refiner = None

    # Cache template features keyed by obj_id so torch.load is called at most
    # once per unique object rather than once per sample.
    template_cache: dict[int, dict] = {}
    template_cache_lock = threading.Lock()
    log_lock = threading.Lock()
    gedi_thread_local = threading.local()

    def _get_thread_gedi_extractors():
        extractors = getattr(gedi_thread_local, "extractors", None)
        if extractors is None:
            # run_gedi_on_ply mutates extractor radii per sample.
            # Keep one extractor set per worker thread to avoid races.
            extractors = build_gedi_extractors()
            gedi_thread_local.extractors = extractors
        return extractors

    def _process_sample(
        global_idx: int,
        sample: dict,
        frame_data: dict,
    ) -> dict:
        frame_features = None
        XYZ_dense = None
        XYZ_sparse = None
        gedi_descriptors = None
        gedi_keypoints = None
        fused_features = None
        features_a = None
        keypoints_a = None
        feat_onb = None
        features_b = None
        keypoints_b = None
        sim = None
        _t_coarse_start = time.perf_counter()
        _coarse_time = 0.0
        _refinement_time = 0.0

        try:
            frame_features = {0: frame_data}

            XYZ_dense = compute_xyz(frame_features, 0, patch=False)
            mask_dense = (frame_data["mask"] * (frame_data["depth"] > 0)).flatten()
            XYZ_dense = XYZ_dense[mask_dense > 0]
            if XYZ_dense.shape[0] == 0:
                return {
                    "global_idx": global_idx,
                    "coarse_result": None,
                    "refined_result": None,
                    "no_dense": 1,
                    "sparse_fallback": 0,
                    "low_matches": 0,
                    "error": None,
                    "coarse_time": 0.0,
                    "refinement_time": 0.0,
                }

            if XYZ_dense.shape[0] > max_dense_points:
                keep_idx = (
                    torch.linspace(
                        0,
                        XYZ_dense.shape[0] - 1,
                        steps=max_dense_points,
                        dtype=torch.float32,
                        device=XYZ_dense.device,
                    )
                    .round()
                    .long()
                )
                XYZ_dense = XYZ_dense[keep_idx]

            XYZ_sparse = compute_xyz(frame_features, 0, patch=True)
            mask_sparse = (frame_data["mask"] * (frame_data["depth"] > 0))[
                PATCH_SIZE // 2 :: PATCH_SIZE, PATCH_SIZE // 2 :: PATCH_SIZE
            ].flatten()
            XYZ_sparse = XYZ_sparse[mask_sparse > 0].cpu().numpy()

            sparse_fallback = 0
            gedi_input_keypoints = XYZ_sparse
            if XYZ_sparse.shape[0] < 3:
                sparse_fallback = 1
                gedi_input_keypoints = None

            gedi_descriptors, gedi_keypoints = run_gedi_on_ply(
                XYZ_dense,
                keypoints=gedi_input_keypoints,
                gedi_extractors=_get_thread_gedi_extractors(),
            )

            fused_features = fuse_features(
                frame_features,
                gedi_descriptors,
                gedi_keypoints,
                distance_threshold=PROD_DINO_AGG_DIST_THR,
                dino_agg_k=PROD_DINO_AGG_K,
                max_points_per_frame=max_dense_points,
            )

            fused_descriptors = fused_features["descriptors"]
            fused_keypoints = fused_features["keypoints"]
            features_b = torch.as_tensor(fused_descriptors, dtype=torch.float32)
            keypoints_b = torch.as_tensor(fused_keypoints, dtype=torch.float32)

            obj_id = sample["obj_id"]
            with template_cache_lock:
                feat_onb = template_cache.get(obj_id)
                if feat_onb is None:
                    feat_onb_loaded = _load_template_features(obj_id, template_root)
                    template_cache[obj_id] = feat_onb_loaded
                    feat_onb = feat_onb_loaded

            features_a = torch.as_tensor(feat_onb["descriptors"], dtype=torch.float32)
            keypoints_a = torch.as_tensor(feat_onb["keypoints"], dtype=torch.float32)

            desc_dim_a = features_a.shape[1]
            desc_dim_b = features_b.shape[1]
            if desc_dim_a != desc_dim_b:
                raise ValueError(
                    "Descriptor dimensionality mismatch between onboarding and eval: "
                    f"{desc_dim_a} vs {desc_dim_b}."
                )

            if gedi_weight != 0.5:
                if not (0 < gedi_dim < desc_dim_a):
                    raise ValueError(
                        "Invalid GeDi split for fused descriptors: "
                        f"gedi_dim={gedi_dim}, total_dim={desc_dim_a}. "
                        "Pass --gedi-dim explicitly to match saved templates."
                    )
                sim = compute_fused_similarity(
                    features_a.T,
                    features_b.T,
                    gedi_dim,
                    gedi_weight=gedi_weight,
                )
            else:
                sim = compute_similarity(features_a.T, features_b.T)

            idx_1, idx_2 = get_top_matches(sim, PROD_TOPK_MATCHES)

            if idx_1.numel() < 3:
                logger.warning(
                    (
                        "Insufficient correspondences for pose estimation "
                        "(scene=%s image=%s obj=%s, matches=%d); skipping sample."
                    ),
                    sample["scene_id"],
                    sample["im_id"],
                    sample["obj_id"],
                    idx_1.numel(),
                )
                return {
                    "global_idx": global_idx,
                    "coarse_result": None,
                    "refined_result": None,
                    "no_dense": 0,
                    "sparse_fallback": sparse_fallback,
                    "low_matches": 1,
                    "error": None,
                    "coarse_time": 0.0,
                    "refinement_time": 0.0,
                }

            try:
                rotation, translation = geometry_ransac_se3(
                    keypoints_a, keypoints_b, idx_1, idx_2
                )
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    "RANSAC failed for scene=%s image=%s obj=%s (%s); falling back to least squares.",
                    sample["scene_id"],
                    sample["im_id"],
                    sample["obj_id"],
                    exc,
                )
                rotation, translation = least_squares_se3(
                    keypoints_a, keypoints_b, idx_1, idx_2
                )

            device = rotation.device
            # Generate Symmetry Hypotheses
            rotZ_90 = torch.tensor(
                [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32, device=device
            )
            rotZ_180 = torch.tensor(
                [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32, device=device
            )
            rotZ_270 = torch.tensor(
                [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=torch.float32, device=device
            )
            rotY_180 = torch.tensor(
                [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32, device=device
            )

            hypotheses = [
                rotation,
                rotation @ rotZ_90,
                rotation @ rotZ_180,
                rotation @ rotZ_270,
                rotation @ rotY_180,
            ]

            best_selection_score = -1.0
            best_linear_score = 0.0
            best_rotation = rotation

            # Prep for Dense Verification
            keypoints_a_dev = keypoints_a.to(device)
            N_b = keypoints_b.shape[0]
            num_samples = min(N_b, 2000)
            if N_b > num_samples:
                gen = torch.Generator(device=device)
                gen.manual_seed(42 + global_idx)
                perm = torch.randperm(N_b, generator=gen, device=device)[:num_samples]
            else:
                perm = torch.arange(N_b, device=device)

            keypoints_b_dev = keypoints_b.to(device)
            features_b_dev = features_b.to(device)
            pts_b_sample = keypoints_b_dev[perm]
            feat_b_sample = features_b_dev[perm]
            feat_a_dev = features_a.to(device)

            inlier_thresh_dense = estimate_inlier_threshold(
                keypoints_a, keypoints_b, explicit_thresh=None
            )

            # Score each hypothesis
            for rot_hyp in hypotheses:
                transformed_a_full = (rot_hyp @ keypoints_a_dev.T).T + translation
                dists = torch.cdist(pts_b_sample, transformed_a_full)
                min_dist, min_idx = dists.min(dim=1)

                valid_mask = min_dist < inlier_thresh_dense
                feat_a_matched = feat_a_dev[min_idx]

                if gedi_weight != 0.5:
                    gedi_f1 = torch.nn.functional.normalize(
                        feat_b_sample[:, :gedi_dim], dim=1
                    )
                    gedi_f2 = torch.nn.functional.normalize(
                        feat_a_matched[:, :gedi_dim], dim=1
                    )
                    dino_f1 = torch.nn.functional.normalize(
                        feat_b_sample[:, gedi_dim:], dim=1
                    )
                    dino_f2 = torch.nn.functional.normalize(
                        feat_a_matched[:, gedi_dim:], dim=1
                    )
                    sim_gedi = (gedi_f1 * gedi_f2).sum(dim=1)
                    sim_dino = (dino_f1 * dino_f2).sum(dim=1)
                    pair_sim = gedi_weight * sim_gedi + (1.0 - gedi_weight) * sim_dino
                else:
                    f1 = torch.nn.functional.normalize(feat_b_sample, dim=1)
                    f2 = torch.nn.functional.normalize(feat_a_matched, dim=1)
                    pair_sim = (f1 * f2).sum(dim=1)

                if num_samples > 0:
                    clamped_sim = torch.clamp(pair_sim, min=0.0)
                    selection_score = (
                        ((clamped_sim**3) * valid_mask.float()).sum().item()
                    )
                    linear_score = (
                        pair_sim * valid_mask.float()
                    ).sum().item() / num_samples
                else:
                    selection_score = 0.0
                    linear_score = 0.0

                if selection_score > best_selection_score:
                    best_selection_score = selection_score
                    best_rotation = rot_hyp
                    best_linear_score = linear_score

            rotation = best_rotation
            feature_aware_score = best_linear_score
            _coarse_time = time.perf_counter() - _t_coarse_start

            coarse_score = sample.get("score", 1.0) * feature_aware_score
            coarse_R_str = " ".join([str(x) for x in rotation.flatten().tolist()])
            coarse_t_str = " ".join([str(x) for x in translation.flatten().tolist()])

            coarse_row = {
                "scene_id": sample["scene_id"],
                "im_id": sample["im_id"],
                "obj_id": sample["obj_id"],
                "score": coarse_score,
                "R": coarse_R_str,
                "t": coarse_t_str,
                "time": 0,
            }

            refined_row = {
                "scene_id": sample["scene_id"],
                "im_id": sample["im_id"],
                "obj_id": sample["obj_id"],
                "score": coarse_score,
                "R": coarse_R_str,
                "t": coarse_t_str,
                "time": 0,
            }

            if refiner is not None:
                try:
                    intrinsics = torch.as_tensor(
                        sample["intrinsics"],
                        dtype=torch.float32,
                        device=device,
                    )
                    _t_refine = time.perf_counter()
                    refinement_result = refiner.refine(
                        obj_id=sample["obj_id"],
                        coarse_rotation=rotation.to(device=device, dtype=torch.float32),
                        coarse_translation=translation.to(
                            device=device, dtype=torch.float32
                        ),
                        coarse_score=float(coarse_score),
                        intrinsics=intrinsics,
                        image_path=str(sample["image_path"]),
                        mask_path=sample.get("mask_path"),
                        segmentation=sample.get("segmentation"),
                        seed=42 + global_idx,
                        depth_path=sample.get("depth_path"),
                        sample_key=f"{sample['scene_id']}_{sample['im_id']}_{sample['obj_id']}",
                    )
                    _refinement_time = time.perf_counter() - _t_refine
                    refined_rotation = refinement_result["rotation"]
                    refined_translation = refinement_result["translation"]
                    refined_score = float(refinement_result["refined_score"])
                    refined_R_str = " ".join(
                        [str(x) for x in refined_rotation.flatten().tolist()]
                    )
                    refined_t_str = " ".join(
                        [str(x) for x in refined_translation.flatten().tolist()]
                    )
                    refined_row = {
                        "scene_id": sample["scene_id"],
                        "im_id": sample["im_id"],
                        "obj_id": sample["obj_id"],
                        "score": refined_score,
                        "R": refined_R_str,
                        "t": refined_t_str,
                        "time": 0,
                    }
                    if not refinement_result["refinement_successful"]:
                        logger.info(
                            "Refinement rejected for scene=%s image=%s obj=%s: %s",
                            sample["scene_id"],
                            sample["im_id"],
                            sample["obj_id"],
                            refinement_result["failure_reason"],
                        )
                except Exception as exc:
                    logger.warning(
                        "Refinement failed for scene=%s image=%s obj=%s (%s). Using coarse pose.",
                        sample["scene_id"],
                        sample["im_id"],
                        sample["obj_id"],
                        exc,
                    )

            return {
                "global_idx": global_idx,
                "coarse_result": coarse_row,
                "refined_result": refined_row,
                "no_dense": 0,
                "sparse_fallback": sparse_fallback,
                "low_matches": 0,
                "error": None,
                "coarse_time": _coarse_time,
                "refinement_time": _refinement_time,
            }
        except Exception:
            err_tb = traceback.format_exc()
            err_line = (
                f"Failed sample: {sample['image_path']} {sample['mask_path']} {sample['obj_id']}\n"
                f"Error:\n{err_tb}\n"
            )
            return {
                "global_idx": global_idx,
                "coarse_result": None,
                "refined_result": None,
                "no_dense": 0,
                "sparse_fallback": 0,
                "low_matches": 0,
                "error": err_line,
                "coarse_time": 0.0,
                "refinement_time": 0.0,
            }
        finally:
            frame_features = None
            XYZ_dense = None
            XYZ_sparse = None
            gedi_descriptors = None
            gedi_keypoints = None
            fused_features = None
            features_a = None
            keypoints_a = None
            feat_onb = None
            features_b = None
            keypoints_b = None
            sim = None

    if sample_workers is None:
        cpu_count = os.cpu_count() or 1
        sample_workers = 1 if torch.cuda.is_available() else min(4, cpu_count)
    max_inflight = (
        max_inflight_samples
        if max_inflight_samples is not None
        else max(1, sample_workers * 2)
    )
    logger.info(
        "Sample parallelism configured: workers=%d, max_inflight=%d",
        sample_workers,
        max_inflight,
    )

    total_time_detection = 0.0
    total_time_coarse = 0.0
    total_time_refinement = 0.0

    ## GEDi extraction - Fusion and RANSAC
    total_samples = len(samples)
    with ThreadPoolExecutor(max_workers=sample_workers) as executor:
        for chunk_start in range(0, total_samples, dino_chunk_size):
            chunk_end = min(chunk_start + dino_chunk_size, total_samples)
            chunk_samples = samples[chunk_start:chunk_end]
            chunk_missing_features = 0
            chunk_no_dense = 0
            chunk_sparse_fallback = 0
            chunk_low_matches = 0
            chunk_errors = 0
            logger.info(
                "Extracting DINO features for chunk [%d, %d)",
                chunk_start,
                chunk_end,
            )
            _t_det = time.perf_counter()
            features_dict, _dataset = extract_dino_features(
                model_path=Path("thirdparty/dinov3"),
                image_dir=Path(""),
                samples=chunk_samples,
                num_workers=0,
                extractor=dino_extractor,
            )
            total_time_detection += time.perf_counter() - _t_det

            if save_dino_features:
                chunk_dir = result_dir / "dino_chunks"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                chunk_path = chunk_dir / (
                    f"dinov3_features_{dataset_label}_{split}_{chunk_start:06d}_{chunk_end:06d}.pt"
                )
                save_features(chunk_path, features_dict)

            # Drop heavy image tensors that are not required for inference.
            # Keep only resized spatial shape for geometry projection.
            for frame in features_dict.values():
                resized_image = frame.get("resized_image")
                if resized_image is not None:
                    frame["resized_image_shape"] = tuple(resized_image.shape[1:])
                    frame.pop("resized_image", None)
                frame.pop("cropped_image", None)

            del _dataset
            gc.collect()

            pending: dict = {}
            chunk_coarse_results: list[tuple[int, dict]] = []
            chunk_refined_results: list[tuple[int, dict]] = []
            completed_in_chunk = 0

            for local_idx, sample in tqdm(
                enumerate(chunk_samples),
                total=len(chunk_samples),
                desc=f"Running inference [{chunk_start}:{chunk_end})",
            ):
                global_idx = chunk_start + local_idx
                frame_data = features_dict.pop(local_idx, None)
                if frame_data is None:
                    chunk_missing_features += 1
                    continue
                assert sample["image_path"] == frame_data["image_path"]

                fut = executor.submit(_process_sample, global_idx, sample, frame_data)
                pending[fut] = True

                if len(pending) >= max_inflight:
                    done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                    for completed in done:
                        pending.pop(completed, None)
                        outcome = completed.result()
                        chunk_no_dense += int(outcome["no_dense"])
                        chunk_sparse_fallback += int(outcome["sparse_fallback"])
                        chunk_low_matches += int(outcome["low_matches"])
                        total_time_coarse += outcome.get("coarse_time", 0.0)
                        total_time_refinement += outcome.get("refinement_time", 0.0)
                        if outcome["coarse_result"] is not None:
                            chunk_coarse_results.append(
                                (int(outcome["global_idx"]), outcome["coarse_result"])
                            )
                        if outcome["refined_result"] is not None:
                            chunk_refined_results.append(
                                (int(outcome["global_idx"]), outcome["refined_result"])
                            )
                        if outcome["error"] is not None:
                            chunk_errors += 1
                            with log_lock:
                                with open(log_path, "a", encoding="utf-8") as f:
                                    f.write(str(outcome["error"]))
                        completed_in_chunk += 1

                        if completed_in_chunk % 10 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

            while pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for completed in done:
                    pending.pop(completed, None)
                    outcome = completed.result()
                    chunk_no_dense += int(outcome["no_dense"])
                    chunk_sparse_fallback += int(outcome["sparse_fallback"])
                    chunk_low_matches += int(outcome["low_matches"])
                    total_time_coarse += outcome.get("coarse_time", 0.0)
                    total_time_refinement += outcome.get("refinement_time", 0.0)
                    if outcome["coarse_result"] is not None:
                        chunk_coarse_results.append(
                            (int(outcome["global_idx"]), outcome["coarse_result"])
                        )
                    if outcome["refined_result"] is not None:
                        chunk_refined_results.append(
                            (int(outcome["global_idx"]), outcome["refined_result"])
                        )
                    if outcome["error"] is not None:
                        chunk_errors += 1
                        with log_lock:
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(str(outcome["error"]))
                    completed_in_chunk += 1

                    if completed_in_chunk % 10 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            chunk_coarse_results.sort(key=lambda x: x[0])
            chunk_refined_results.sort(key=lambda x: x[0])
            chunk_coarse_rows = [row for _, row in chunk_coarse_results]
            chunk_refined_rows = [row for _, row in chunk_refined_results]
            chunk_written = len(chunk_coarse_rows)
            if chunk_written > 0:
                df_chunk = pd.DataFrame(chunk_coarse_rows)
                df_chunk.to_csv(
                    coarse_csv_path,
                    mode="a",
                    index=False,
                    header=(total_results_written == 0),
                )
                df_refined_chunk = pd.DataFrame(chunk_refined_rows)
                df_refined_chunk.to_csv(
                    refined_csv_path,
                    mode="a",
                    index=False,
                    header=(total_results_written == 0),
                )
                total_results_written += chunk_written
                df_chunk = None
                df_refined_chunk = None

            features_dict.clear()
            del features_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            chunk_coarse_results.clear()
            chunk_refined_results.clear()
            chunk_coarse_rows = None
            chunk_refined_rows = None
            rss_mb = _get_current_rss_mb()
            rss_msg = "unknown" if rss_mb is None else f"{rss_mb:.1f} MB"

            logger.info(
                (
                    "Chunk [%d, %d) summary: processed=%d, results_total=%d, "
                    "missing_features=%d, no_dense=%d, sparse_fallback=%d, "
                    "low_matches=%d, errors=%d, template_cache=%d, rss=%s"
                ),
                chunk_start,
                chunk_end,
                len(chunk_samples),
                total_results_written,
                chunk_missing_features,
                chunk_no_dense,
                chunk_sparse_fallback,
                chunk_low_matches,
                chunk_errors,
                len(template_cache),
                rss_msg,
            )

            total_missing_features += chunk_missing_features
            total_no_dense += chunk_no_dense
            total_sparse_fallback += chunk_sparse_fallback
            total_low_matches += chunk_low_matches
            total_errors += chunk_errors

    if total_results_written > 0:
        logger.debug(
            "Saved inference results to coarse=%s refined=%s",
            coarse_csv_path,
            refined_csv_path,
        )
    else:
        raise RuntimeError(
            "No pose estimates were produced. "
            f"processed={total_samples}, "
            f"missing_features={total_missing_features}, "
            f"no_dense={total_no_dense}, "
            f"sparse_fallback={total_sparse_fallback}, "
            f"low_matches={total_low_matches}, "
            f"errors={total_errors}. "
            "Check output/log.txt for per-sample exceptions and ensure onboarding templates/features exist."
        )

    total_images = len(set((s["scene_id"], s["im_id"]) for s in samples))
    timing_msg = (
        "Timing summary (%d samples, %d images) —\n"
        "Detection/Segmentation: %.2f s (%.3f s/sample, %.3f s/image)\n"
        "Coarse Pose: %.2f s (%.3f s/sample, %.3f s/image)\n"
        "Refinement: %.2f s (%.3f s/sample, %.3f s/image)\n"
        "Total (per image): %.3f s/image"
    ) % (
        total_samples,
        total_images,
        total_time_detection,
        total_time_detection / max(total_samples, 1),
        total_time_detection / max(total_images, 1),
        total_time_coarse,
        total_time_coarse / max(total_samples, 1),
        total_time_coarse / max(total_images, 1),
        total_time_refinement,
        total_time_refinement / max(total_samples, 1),
        total_time_refinement / max(total_images, 1),
        (total_time_detection + total_time_coarse + total_time_refinement) / max(total_images, 1),
    )
    
    logger.info("\n" + timing_msg)
    
    # Append the timing summary to log.txt
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + timing_msg + "\n")

    # Explicitly release model/extractor resources at the end of inference.
    dino_extractor = None
    dino_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return


def main():
    """Main entry point for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    # Keep tqdm bars readable by silencing per-sample INFO logs from feature fusion.
    logging.getLogger("novasplat.features.dense_features").setLevel(logging.WARNING)
    args = parse_args()
    input_dir = Path(args.input)
    if (input_dir / args.dataset / args.split).exists():
        dataset_dir = input_dir / args.dataset
    elif (input_dir / args.split).exists():
        dataset_dir = input_dir
    else:
        raise FileNotFoundError(
            "Could not resolve dataset directory from input and dataset name. "
            f"Checked: {input_dir / args.dataset / args.split} and {input_dir / args.split}"
        )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_cnos_detection:
        cnos_detection_file = (
            output_dir / f"cnos25/nms-None_{args.dataset}-{args.split}.json"
        )
        if not cnos_detection_file.exists():
            raise FileNotFoundError(
                f"CNOS detection file not found: {cnos_detection_file}"
            )
        logger.info("Using CNOS detection file: %s", cnos_detection_file)
    else:
        cnos_detection_file = None
        logger.info("Using gt masks")
    run(
        dataset_dir,
        output_dir,
        args.split,
        args.dataset,
        scene_id=args.scene_id,
        im_id=args.im_id,
        obj_id=args.obj_id,
        instance_id=args.instance_id,
        gedi_weight=args.gedi_weight,
        gedi_dim_override=args.gedi_dim,
        dino_chunk_size=args.dino_chunk_size,
        save_dino_features=args.save_dino_features,
        max_dense_points=args.max_dense_points,
        cnos_detection_file=cnos_detection_file,
        max_samples=args.max_samples,
        sample_workers=args.sample_workers,
        max_inflight_samples=args.max_inflight_samples,
        enable_refinement=args.enable_refinement,
        sample_index=args.sample_index,
        refinement_debug_dir=Path(args.refinement_debug_dir)
        if args.refinement_debug_dir
        else None,
        refinement_geo_weight=args.refinement_geo_weight,
        refinement_epsilon_z=args.refinement_epsilon_z,
        refinement_epsilon_xy=args.refinement_epsilon_xy,
        refinement_silhouette_weight=args.refinement_silhouette_weight,
        refinement_rotation_parametrization=args.refinement_rotation_parametrization,
        refinement_epsilon_rotation_deg=args.refinement_epsilon_rotation_deg,
        refiner_type=args.refiner,
        icp_max_correspondence_dist=args.icp_max_correspondence_dist,
        icp_voxel_size=args.icp_voxel_size,
        icp_method=args.icp_method,
    )


if __name__ == "__main__":
    main()
