import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class BOPValDataset(Dataset):
    """
    Clean BOP VAL dataset.

    One sample = one object instance.
    Uses scene_gt.json for correct mapping.
    """

    def __init__(self, root_dir, split="val", cnos_detection_file: str | None = None):
        self.root_dir = Path(root_dir)
        self.split = split
        if cnos_detection_file is not None:
            self.cnos_detection_file = Path(cnos_detection_file)
            self.use_gt = False
        else:
            self.use_gt = True

        self.samples = []
        if self.use_gt:
            self._build_samples()
        else:
            self.build_cnos_samples()

    # -----------------------------------------------------
    # Build all samples directly from scene_gt
    # -----------------------------------------------------

    def build_cnos_samples(self):
        det_path = self.cnos_detection_file
        with open(det_path, "r") as f:
            detections = json.load(f)

        scene_cams = {}
        self.samples = []
        for det in detections:
            scene_id = det["scene_id"]
            image_id = det["image_id"]
            object_id = det["category_id"]
            if scene_id not in scene_cams:
                scene_cam_path = (
                    self.root_dir / self.split / f"{scene_id:06d}" / "scene_camera.json"
                )
                with open(scene_cam_path, "r") as f:
                    scene_cams[scene_id] = json.load(f)
            cam_K = np.array(scene_cams[scene_id][str(image_id)]["cam_K"]).reshape(3, 3)
            if not (
                self.root_dir
                / self.split
                / f"{scene_id:06d}"
                / "rgb"
                / f"{image_id:06d}.png"
            ).exists():
                continue
            self.samples.append(
                {
                    "scene_id": scene_id,
                    "im_id": image_id,
                    "obj_id": object_id,
                    "image_path": self.root_dir
                    / self.split
                    / f"{scene_id:06d}"
                    / "rgb"
                    / f"{image_id:06d}.png",
                    "depth_path": self.root_dir
                    / self.split
                    / f"{scene_id:06d}"
                    / "depth"
                    / f"{image_id:06d}.png",
                    "mask_path": None,
                    "intrinsics": cam_K,
                    "score": det["score"],
                    "bbox": det["bbox"],
                    "segmentation": det.get("segmentation"),
                }
            )
        print(f"Loaded {len(self.samples)} BOP samples using CNOS25 detections")

    def _build_samples(self):
        split_dir = self.root_dir / self.split

        for scene_folder in sorted(split_dir.glob("*/")):
            if not scene_folder.name.isdigit():
                continue

            scene_id = int(scene_folder.name)

            scene_gt_path = scene_folder / "scene_gt.json"
            scene_cam_path = scene_folder / "scene_camera.json"

            if not scene_gt_path.exists():
                continue

            with open(scene_gt_path, "r") as f:
                scene_gt = json.load(f)

            with open(scene_cam_path, "r") as f:
                scene_cam = json.load(f)

            rgb_dir = scene_folder / "rgb"
            depth_dir = scene_folder / "depth"
            mask_visib_dir = scene_folder / "mask_visib"
            mask_dir = scene_folder / "mask"

            # -------------------------------------------------
            # iterate images
            # -------------------------------------------------
            for im_id_str in sorted(scene_gt.keys(), key=int):
                instances = scene_gt[im_id_str]

                im_id = int(im_id_str)

                rgb_path = rgb_dir / f"{im_id:06d}.png"
                depth_path = depth_dir / f"{im_id:06d}.png"

                cam_K = np.array(scene_cam[im_id_str]["cam_K"]).reshape(3, 3)

                # -------------------------------------------------
                # iterate instances (THIS IS THE KEY)
                # -------------------------------------------------
                for instance_id, inst in enumerate(instances):
                    obj_id = int(inst["obj_id"])

                    gt_id = instance_id
                    visib_mask_path = mask_visib_dir / f"{im_id:06d}_{gt_id:06d}.png"
                    full_mask_path = mask_dir / f"{im_id:06d}_{gt_id:06d}.png"

                    if visib_mask_path.exists():
                        mask_path = visib_mask_path
                        mask_kind = "mask_visib"
                    elif full_mask_path.exists():
                        mask_path = full_mask_path
                        mask_kind = "mask"
                    else:
                        continue

                    self.samples.append(
                        {
                            "scene_id": scene_id,
                            "im_id": im_id,
                            "obj_id": obj_id,
                            "instance_id": instance_id,
                            "gt_id": gt_id,
                            "image_path": rgb_path,
                            "depth_path": depth_path,
                            "mask_path": mask_path,
                            "mask_kind": mask_kind,
                            "intrinsics": cam_K,
                            "pose": inst,  # full GT pose
                        }
                    )

        print(f"Loaded {len(self.samples)} VAL samples")

    # -----------------------------------------------------
    # Dataset interface
    # -----------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -----------------------------------------------------
# DEBUG
# -----------------------------------------------------
if __name__ == "__main__":
    dataset = BOPValDataset(
        root_dir=Path("~/data/bop_h3/hopev2").expanduser(),
        split="val",
    )

    for i in range(20):
        print(dataset[i]["obj_id"], dataset[i]["image_path"], dataset[i]["mask_path"])
