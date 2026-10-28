#!/usr/bin/env python3
"""
Render high-quality object mesh overlays for one BOP image from a CSV prediction file.

Expected CSV format:
scene_id,im_id,obj_id,score,R,t,time
where R is 9 space-separated values and t is 3 space-separated values in BOP/OpenCV camera coordinates.

Default paths are inferred from the provided visualize.py:
  csv:          ./output/novasplat_hopev2-val.csv
  onboarding:   ./output/onboarding/obj_000001/scene_dense.ply

Examples:
  python render_csv_mesh_overlay.py --scene-id 2 --im-id 3 \
      --csv ./output/novasplat_hopev2-val.csv \
      --out ./overlay_scene2_im3.png

  python render_csv_mesh_overlay.py --scene-id 2 --im-id 3 --top-k-per-object 1 --score-thresh 0.05

Notes:
  - Prefer true mesh files from DATASET_ROOT/models* if present.
  - If only onboarding scene_dense.ply point clouds exist, it renders them as colored point splats with anti-aliasing.
  - Open3D offscreen rendering may need a headless/EGL setup on servers.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import pandas as pd

try:
    import open3d.visualization.rendering as rendering
except Exception as exc:  # pragma: no cover
    rendering = None
    _RENDERING_IMPORT_ERROR = exc
else:
    _RENDERING_IMPORT_ERROR = None


# -----------------------------
# IO helpers
# -----------------------------


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def parse_vec(s: object, n: int) -> np.ndarray:
    vals = np.array([float(x) for x in str(s).split()], dtype=np.float64)
    if vals.size != n:
        raise ValueError(f"Expected {n} values, got {vals.size}: {s}")
    return vals


def load_K(scene_camera: dict, im_id: int) -> np.ndarray:
    return np.asarray(scene_camera[str(im_id)]["cam_K"], dtype=np.float64).reshape(3, 3)


def image_path_for(scene_dir: Path, im_id: int) -> Path:
    for sub in ["rgb", "gray"]:
        for ext in ["png", "jpg", "jpeg", "tif"]:
            p = scene_dir / sub / f"{im_id:06d}.{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(
        f"Could not find image for im_id={im_id} under {scene_dir}/rgb or /gray"
    )


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )


# -----------------------------
# Model loading
# -----------------------------


def model_candidates(
    dataset_root: Path, onboarding_root: Path, obj_id: int
) -> Iterable[Path]:
    name = f"obj_{obj_id:06d}.ply"
    yield dataset_root / "models" / name
    yield dataset_root / "models_eval" / name
    yield dataset_root / "models_cad" / name
    yield onboarding_root / f"obj_{obj_id:06d}" / "model.ply"
    yield onboarding_root / f"obj_{obj_id:06d}" / "mesh.ply"
    yield onboarding_root / f"obj_{obj_id:06d}" / "scene_dense.ply"


def load_geometry(
    dataset_root: Path, onboarding_root: Path, obj_id: int
) -> Tuple[o3d.geometry.Geometry, Path, str]:
    for p in model_candidates(dataset_root, onboarding_root, obj_id):
        if not p.exists():
            continue

        # Try triangle mesh first. If the file is only a point cloud, Open3D returns 0 triangles.
        mesh = o3d.io.read_triangle_mesh(str(p), enable_post_processing=True)
        if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()

            # Open3D doesn't automatically load texture_u/texture_v from PLY, so we use trimesh
            if p.suffix.lower() == ".ply":
                try:
                    import trimesh

                    tm = trimesh.load(str(p))
                    if hasattr(tm.visual, "uv") and tm.visual.uv is not None:
                        uv = tm.visual.uv
                        tris = np.asarray(mesh.triangles)
                        uv_flat = np.zeros((len(tris) * 3, 2), dtype=np.float64)
                        for i in range(3):
                            uv_flat[i::3, :] = uv[tris[:, i]]
                        mesh.triangle_uvs = o3d.utility.Vector2dVector(uv_flat)
                except Exception:
                    pass

            return mesh, p, "mesh"

        pcd = o3d.io.read_point_cloud(str(p))
        if pcd is not None and len(pcd.points) > 0:
            return pcd, p, "pointcloud"

    raise FileNotFoundError(f"No model found for obj_id={obj_id:06d}")


def transform_geometry_to_camera(
    geom: o3d.geometry.Geometry, R: np.ndarray, t: np.ndarray
) -> o3d.geometry.Geometry:
    g = copy.deepcopy(geom)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    g.transform(T)
    return g


# -----------------------------
# Rendering helpers
# -----------------------------


def make_material(
    kind: str,
    base_color: Tuple[float, float, float, float],
    point_size: float,
    texture: Optional[object] = None,
) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    if kind == "pointcloud":
        mat.shader = "defaultUnlit"
        mat.point_size = float(point_size)
        mat.base_color = base_color
    else:
        # Good-looking physically based shader. Uses mesh normals and scene lighting.
        mat.shader = "defaultLit"
        mat.base_color = base_color
        mat.base_roughness = 0.65
        mat.base_reflectance = 0.35
        if texture is not None:
            mat.albedo_img = texture
    return mat


def render_layer(
    geoms: list[
        Tuple[
            str,
            o3d.geometry.Geometry,
            str,
            Tuple[float, float, float, float],
            Optional[object],
        ]
    ],
    K: np.ndarray,
    width: int,
    height: int,
    super_sample: int,
    point_size: float,
    bg_rgba: Tuple[float, float, float, float] = (0, 0, 0, 0),
) -> np.ndarray:
    if rendering is None:
        raise RuntimeError(
            f"Open3D rendering module unavailable: {_RENDERING_IMPORT_ERROR}"
        )

    ss = int(max(1, super_sample))
    rw, rh = width * ss, height * ss
    K_ss = K.copy()
    K_ss[:2, :] *= ss

    renderer = rendering.OffscreenRenderer(rw, rh)
    renderer.scene.set_background(bg_rgba)
    renderer.scene.set_lighting(
        rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.0, 0.0, 0.0)
    )
    renderer.scene.scene.set_sun_light([0.3, -0.5, -1.0], [1.0, 1.0, 1.0], 60000)
    renderer.scene.scene.enable_sun_light(True)

    for name, geom, kind, color, texture in geoms:
        renderer.scene.add_geometry(
            name, geom, make_material(kind, color, point_size * ss, texture)
        )

    # Geometry has already been transformed into camera coordinates, so camera extrinsic is identity.
    renderer.setup_camera(K_ss.astype(np.float64), np.eye(4, dtype=np.float64), rw, rh)

    rgba = np.asarray(renderer.render_to_image())
    if rgba.shape[2] == 3:
        depth = np.asarray(renderer.render_to_depth_image())
        alpha = (np.isfinite(depth) & (depth > 0)).astype(np.uint8) * 255
        rgba = np.dstack([rgba, alpha])

    if ss > 1:
        rgba = cv2.resize(rgba, (width, height), interpolation=cv2.INTER_AREA)
    return rgba.astype(np.uint8)


def alpha_composite_rgb(
    base_rgb: np.ndarray, overlay_rgba: np.ndarray, alpha_scale: float = 1.0
) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    over = overlay_rgba[..., :3].astype(np.float32)
    a = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    a *= float(alpha_scale)
    return over * a + base * (1.0 - a)


def add_silhouette(
    rgb: np.ndarray, overlay_rgba: np.ndarray, thickness: int = 2
) -> np.ndarray:
    if thickness <= 0:
        return rgb
    mask = (overlay_rgba[..., 3] > 0).astype(np.uint8) * 255
    kernel = np.ones((2 * thickness + 1, 2 * thickness + 1), np.uint8)
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    out = rgb.copy()
    # Dark outline improves readability on bright backgrounds.
    out[edge > 0] = 0.25 * out[edge > 0]
    return out


def stable_color(obj_id: int, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    # Deterministic bright-ish color per object for models without vertex colors.
    rng = np.random.default_rng(obj_id * 1009)
    c = rng.uniform(0.35, 1.0, size=3)
    return float(c[0]), float(c[1]), float(c[2]), float(alpha)


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory of the BOP dataset (e.g. ~/data/bop_h3/hopev2)",
    )
    ap.add_argument("--split", default="val")
    ap.add_argument("--csv", default="./output/novasplat_hopev2-val.csv")
    ap.add_argument("--onboarding-root", default="./output/onboarding")
    ap.add_argument("--scene-id", type=int, required=True)
    ap.add_argument("--im-id", type=int, required=True)
    ap.add_argument("--out", default="./mesh_overlay.png")
    ap.add_argument("--score-thresh", type=float, default=-np.inf)
    ap.add_argument(
        "--top-k-per-object",
        type=int,
        default=1,
        help="Keep top K predictions per obj_id for this image. Use 0 to keep all.",
    )
    ap.add_argument("--alpha", type=float, default=0.72, help="Overlay transparency.")
    ap.add_argument(
        "--super-sample",
        type=int,
        default=2,
        help="Render at Nx resolution then downsample for anti-aliasing.",
    )
    ap.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Point size if only point clouds are available.",
    )
    ap.add_argument(
        "--outline",
        type=int,
        default=2,
        help="Silhouette thickness in pixels. Use 0 to disable.",
    )
    ap.add_argument(
        "--use-object-colors",
        action="store_true",
        help="Ignore mesh/point colors and tint each object by obj_id.",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    onboarding_root = Path(args.onboarding_root)
    scene_dir = dataset_root / args.split / f"{args.scene_id:06d}"

    image_path = image_path_for(scene_dir, args.im_id)
    rgb = read_rgb(image_path)
    h, w = rgb.shape[:2]

    scene_camera = load_json(scene_dir / "scene_camera.json")
    K = load_K(scene_camera, args.im_id)

    df = pd.read_csv(args.csv)
    rows = df[
        (df.scene_id.astype(int) == args.scene_id)
        & (df.im_id.astype(int) == args.im_id)
    ].copy()
    rows = rows[rows.score.astype(float) >= args.score_thresh]
    if rows.empty:
        raise RuntimeError(
            f"No predictions for scene={args.scene_id}, im={args.im_id}, score_thresh={args.score_thresh}"
        )

    rows["score"] = rows["score"].astype(float)
    rows = rows.sort_values("score", ascending=False)
    if args.top_k_per_object and args.top_k_per_object > 0:
        rows = rows.groupby("obj_id", group_keys=False).head(args.top_k_per_object)

    print(f"Image: {image_path}")
    print(f"Rendering {len(rows)} predictions")

    geoms: list[
        Tuple[
            str,
            o3d.geometry.Geometry,
            str,
            Tuple[float, float, float, float],
            Optional[object],
        ]
    ] = []
    geometry_cache: dict[
        int,
        Tuple[o3d.geometry.Geometry, Path, str, Optional[object]],
    ] = {}
    for i, row in rows.reset_index(drop=True).iterrows():
        obj_id = int(row.obj_id)
        R = parse_vec(row.R, 9).reshape(3, 3)
        t = parse_vec(row.t, 3)

        cached = geometry_cache.get(obj_id)
        if cached is None:
            geom, model_path, kind = load_geometry(dataset_root, onboarding_root, obj_id)

            texture = None
            if kind == "mesh":
                tex_path = model_path.with_suffix(".png")
                if tex_path.exists():
                    texture = o3d.io.read_image(str(tex_path))

            geometry_cache[obj_id] = (geom, model_path, kind, texture)
        else:
            geom, model_path, kind, texture = cached

        geom_cam = transform_geometry_to_camera(copy.deepcopy(geom), R, t)

        color = stable_color(obj_id, 1.0)
        if not args.use_object_colors and kind == "mesh":
            # If mesh has vertex colors or textures, defaultLit can use them; base_color remains neutral.
            color = (1.0, 1.0, 1.0, 1.0)
        elif not args.use_object_colors and kind == "pointcloud" and geom.has_colors():
            color = (1.0, 1.0, 1.0, 1.0)

        print(
            f"  obj={obj_id:06d} score={float(row.score):.4f} kind={kind} model={model_path}"
        )
        geoms.append((f"obj_{obj_id:06d}_{i}", geom_cam, kind, color, texture))

    overlay = render_layer(
        geoms=geoms,
        K=K,
        width=w,
        height=h,
        super_sample=args.super_sample,
        point_size=args.point_size,
    )

    # pure render
    render_rgb = overlay[..., :3]

    # composite
    composed = alpha_composite_rgb(rgb, overlay, alpha_scale=args.alpha)
    composed = add_silhouette(composed, overlay, thickness=args.outline)

    composed = np.clip(composed, 0, 255).astype(np.uint8)
    composed = np.ascontiguousarray(composed)

    for i, row in rows.reset_index(drop=True).iterrows():
        t = parse_vec(row.t, 3)
        score = float(row.score)

        p_img = K @ t
        if p_img[2] > 0:
            p_img = p_img / p_img[2]
            u, v = int(np.round(p_img[0])), int(np.round(p_img[1]))

            text = f"{score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7

            # Draw black outline
            cv2.putText(
                composed, text, (u, v), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA
            )
            # Draw white text
            cv2.putText(
                composed,
                text,
                (u, v),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    out_path = Path(args.out)

    # -------------------------------------------------
    # save original image
    # -------------------------------------------------
    orig_path = out_path.with_name(out_path.stem + "_orig.png")
    write_rgb(orig_path, rgb)

    # -------------------------------------------------
    # save pure render
    # -------------------------------------------------
    render_path = out_path.with_name(out_path.stem + "_render.png")
    write_rgb(render_path, render_rgb)

    # -------------------------------------------------
    # save overlay
    # -------------------------------------------------
    write_rgb(out_path, composed)

    print("Saved:")
    print(" ", orig_path)
    print(" ", render_path)
    print(" ", out_path)


if __name__ == "__main__":
    main()
