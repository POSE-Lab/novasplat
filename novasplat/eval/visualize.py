import os
import json
import cv2
import numpy as np
import pandas as pd
import open3d as o3d

# =====================================================
# CONFIG
# =====================================================

DATASET_ROOT = os.path.expanduser("~/data/bop_h3/hopev2")
SPLIT = "val"  # "val" or "test"

ONBOARDING_ROOT = "./output/onboarding"

CSV_PATH = f"./output/novasplat_hopev2-{SPLIT}.csv"

SCENE_ID = 1
IM_ID = 2

OUTPUT_PATH = "./novasplat/eval/projection_from_csv.png"

MAX_POINTS = 80000
POINT_RADIUS = 1

# =====================================================
# HELPERS
# =====================================================


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_K(scene_camera, im_id):
    return np.array(scene_camera[str(im_id)]["cam_K"]).reshape(3, 3)


def parse_vec(s):
    return np.array([float(x) for x in str(s).split()], dtype=np.float64)


def maybe_subsample(pts, colors, max_points):
    if max_points is None or len(pts) <= max_points:
        return pts, colors
    idx = np.random.choice(len(pts), max_points, replace=False)
    return pts[idx], colors[idx]


def project_points(pts3d, R, t, K):
    pts_cam = (R @ pts3d.T + t).T

    valid = pts_cam[:, 2] > 1e-6
    pts_cam = pts_cam[valid]

    if len(pts_cam) == 0:
        return np.zeros((0, 2)), valid

    pts_2d_h = (K @ pts_cam.T).T
    pts_2d = pts_2d_h[:, :2] / pts_2d_h[:, 2:3]

    return pts_2d, valid


def draw_colored_points(img, pts2d, colors, radius=1):
    h, w = img.shape[:2]
    pts = pts2d.astype(np.int32)

    mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)

    pts = pts[mask]
    colors = colors[mask]

    out = img.copy()

    for (x, y), c in zip(pts, colors):
        bgr = (int(c[2]), int(c[1]), int(c[0]))
        cv2.circle(out, (x, y), radius, bgr, -1)

    return out, len(pts)


# =====================================================
# LOAD IMAGE + CAMERA
# =====================================================

scene_folder = os.path.join(DATASET_ROOT, SPLIT, f"{SCENE_ID:06d}")

img_path = os.path.join(scene_folder, "rgb", f"{IM_ID:06d}.png")
img = cv2.imread(img_path)

if img is None:
    raise RuntimeError(f"Image not found: {img_path}")

scene_camera = load_json(os.path.join(scene_folder, "scene_camera.json"))

K = load_K(scene_camera, IM_ID)

# =====================================================
# LOAD CSV PREDICTIONS
# =====================================================

df = pd.read_csv(CSV_PATH)

# filter for this frame
rows = df[(df.scene_id.astype(int) == SCENE_ID) & (df.im_id.astype(int) == IM_ID)]

if len(rows) == 0:
    print(f"No predictions for scene={SCENE_ID}, im={IM_ID}")
    print("Using first available frame from CSV...")

    first = df.iloc[0]
    SCENE_ID = int(first.scene_id)
    IM_ID = int(first.im_id)

    rows = df[(df.scene_id.astype(int) == SCENE_ID) & (df.im_id.astype(int) == IM_ID)]

print("Using scene:", SCENE_ID, "image:", IM_ID)
print("Num predicted objects:", len(rows))

canvas = img.copy()

# =====================================================
# LOOP OVER OBJECTS
# =====================================================

for _, row in rows.iterrows():
    obj_id = int(row.obj_id)

    print("Rendering obj:", obj_id)

    R = parse_vec(row.R).reshape(3, 3)
    t = parse_vec(row.t).reshape(3, 1)

    # -------------------------------------------------
    # Load dense cloud
    # -------------------------------------------------
    dense_path = os.path.join(ONBOARDING_ROOT, f"obj_{obj_id:06d}", "scene_dense.ply")

    if not os.path.exists(dense_path):
        print("Missing dense cloud:", dense_path)
        continue

    pcd = o3d.io.read_point_cloud(dense_path)

    verts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255.0

    verts = verts.astype(np.float64)
    colors = colors.astype(np.uint8)

    verts, colors = maybe_subsample(verts, colors, MAX_POINTS)

    # -------------------------------------------------
    # Project + draw
    # -------------------------------------------------
    pts2d, valid = project_points(verts, R, t, K)
    colors_valid = colors[valid]

    canvas, n = draw_colored_points(canvas, pts2d, colors_valid, POINT_RADIUS)

    print("Drawn points:", n)

# =====================================================
# SAVE
# =====================================================
cv2.imwrite("real_image.png", img)

cv2.imwrite(OUTPUT_PATH, canvas)

print("Saved:", OUTPUT_PATH)
