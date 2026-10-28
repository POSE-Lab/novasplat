"""Extract Novasplat features for pre-rendered object templates.

This is a thin driver around
:func:`novasplat.features.dense_features.extract_and_fuse_features` for the case
where templates (RGB renders, masks, depth maps, camera params and ``model.ply``)
already exist -- e.g. produced by a separate onboarding -- and only the learned
features still need to be computed before running inference.

For every ``obj_*`` directory under ``--templates`` it writes
``features/dinov3.pt``, ``features/gedi.pt`` and ``features/fused.pt`` in place
(or under ``--output`` if given), which is exactly what ``run_inference.py``
expects at ``<template_root>/obj_XXXXXX/features/fused.pt``.

Example:
    ./run_template_features.sh -i ~/data/bop_h3/hopev2/templates
"""

import argparse
import logging
import sys
from pathlib import Path

from novasplat.features.dense_features import (
    PROD_DINO_AGG_DIST_THR,
    PROD_DINO_AGG_K,
    PROD_NUM_LAST_DINO_LAYERS,
    build_gedi_extractors,
    extract_and_fuse_features,
    load_dino_extractor,
)

logger = logging.getLogger(__name__)

REQUIRED_ENTRIES = (
    "rgb",
    "mask_visib",
    "depth",
    "rendered_camera_params.npz",
    "model.ply",
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract Novasplat features for pre-rendered templates.",
    )
    parser.add_argument(
        "-i",
        "--templates",
        required=True,
        type=str,
        help=(
            "Template root containing obj_* subdirectories, or a single "
            "obj_* template directory."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help=(
            "Optional output root. Features are written to "
            "<output>/obj_XXXXXX/features/. Defaults to writing in place under "
            "the template directory."
        ),
    )
    parser.add_argument(
        "--object",
        default=None,
        type=str,
        help="Process only this object directory name (e.g. obj_000001).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if features/fused.pt already exists.",
    )
    parser.add_argument(
        "--dinov3-repo",
        default="thirdparty/dinov3",
        type=str,
        help="Path to the local DINOv3 hub repository.",
    )
    parser.add_argument("--model-name", default="dinov3_vitl16", type=str)
    parser.add_argument(
        "--num-last-dino-layers",
        default=PROD_NUM_LAST_DINO_LAYERS,
        type=int,
    )
    parser.add_argument("--dino-agg-k", default=PROD_DINO_AGG_K, type=int)
    parser.add_argument(
        "--dino-agg-dist-thr",
        default=PROD_DINO_AGG_DIST_THR,
        type=float,
    )
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    return parser.parse_args()


def discover_object_dirs(templates_root: Path, object_name: str | None) -> list[Path]:
    """Return the list of template object directories to process."""
    if object_name is not None:
        candidate = templates_root / object_name
        if not candidate.is_dir():
            raise FileNotFoundError(f"Object directory not found: {candidate}")
        return [candidate]

    # Allow pointing directly at a single object directory.
    if (templates_root / "model.ply").exists() and (templates_root / "rgb").is_dir():
        return [templates_root]

    obj_dirs = sorted(p for p in templates_root.glob("obj_*") if p.is_dir())
    if not obj_dirs:
        raise FileNotFoundError(
            f"No obj_* template directories found under {templates_root}"
        )
    return obj_dirs


def missing_inputs(template_dir: Path) -> list[str]:
    """Return the names of any required template inputs that are absent."""
    return [name for name in REQUIRED_ENTRIES if not (template_dir / name).exists()]


def run(args: argparse.Namespace) -> int:
    """Extract features for every selected template object.

    Returns a process exit code (0 on full success).
    """
    templates_root = Path(args.templates).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve() if args.output else None

    object_dirs = discover_object_dirs(templates_root, args.object)
    logger.info("Found %d template object(s) to process", len(object_dirs))

    dino_extractor = load_dino_extractor(
        args.dinov3_repo,
        args.model_name,
        args.num_last_dino_layers,
    )
    gedi_extractors = build_gedi_extractors()

    failures: list[str] = []
    for template_dir in object_dirs:
        obj_name = template_dir.name
        if output_root is not None:
            features_dir = output_root / obj_name / "features"
        else:
            features_dir = template_dir / "features"

        if (features_dir / "fused.pt").exists() and not args.overwrite:
            logger.info(
                "[%s] features/fused.pt exists, skipping (use --overwrite)",
                obj_name,
            )
            continue

        absent = missing_inputs(template_dir)
        if absent:
            logger.error(
                "[%s] missing template inputs: %s", obj_name, ", ".join(absent)
            )
            failures.append(obj_name)
            continue

        logger.info("[%s] extracting features -> %s", obj_name, features_dir)
        try:
            extract_and_fuse_features(
                template_dir,
                features_dir,
                dinov3_repo=args.dinov3_repo,
                model_name=args.model_name,
                num_last_dino_layers=args.num_last_dino_layers,
                dino_agg_k=args.dino_agg_k,
                dino_agg_dist_thr=args.dino_agg_dist_thr,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dino_extractor=dino_extractor,
                gedi_extractors=gedi_extractors,
            )
            logger.info("[%s] done", obj_name)
        except Exception:
            logger.exception("[%s] feature extraction failed", obj_name)
            failures.append(obj_name)

    if failures:
        logger.error(
            "Feature extraction failed for %d object(s): %s",
            len(failures),
            ", ".join(failures),
        )
        return 1
    logger.info("Feature extraction completed for all objects.")
    return 0


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
