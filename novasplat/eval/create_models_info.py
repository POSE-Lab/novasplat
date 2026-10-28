import trimesh
import json
import os
import numpy as np

model_dir = os.path.expanduser("~/data/bop_h3/hopev2/models_eval")
output_path = os.path.join(model_dir, "models_info.json")
models_info = {}

# Iterate through your 28 objects
for i in range(1, 29):
    obj_id = f"{i:06d}"
    ply_path = os.path.join(model_dir, f"obj_{obj_id}.ply")

    if os.path.exists(ply_path):
        print(f"Processing Object {i}...")
        mesh = trimesh.load(ply_path)

        # Calculate Bounding Box
        min_p, max_p = mesh.bounds
        size = max_p - min_p

        # Calculate Diameter (Distance between furthest points)
        # Using convex hull for speed on dense point clouds
        diameter = (
            mesh.convex_hull.extents.max()
            if hasattr(mesh, "convex_hull")
            else np.linalg.norm(size)
        )

        models_info[str(i)] = {
            "diameter": float(diameter),
            "min_x": float(min_p[0]),
            "min_y": float(min_p[1]),
            "min_z": float(min_p[2]),
            "size_x": float(size[0]),
            "size_y": float(size[1]),
            "size_z": float(size[2]),
        }

with open(output_path, "w") as f:
    json.dump(models_info, f, indent=2)

print(f"Successfully created {output_path}")
