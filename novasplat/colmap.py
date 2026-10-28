"""COLMAP reconstruction utilities for 3D scene processing."""

import logging
import subprocess
from pathlib import Path

import numpy as np
import pycolmap

from scipy.spatial.transform import Rotation

from novasplat.lie import invert_se3

logger = logging.getLogger(__name__)


def compute_camera_pose_from_object_pose(
    obj_r: np.ndarray,
    obj_t: np.ndarray,
    obj_center_wc: np.ndarray,
    obj_rotation_wc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute camera pose from object pose in world coordinates."""
    # Compute object pose in world coordinates
    obj_pose_wc = np.eye(4)
    obj_pose_wc[:3, :3] = obj_rotation_wc
    obj_pose_wc[:3, 3] = obj_center_wc

    # Compute object pose in camera coordinates
    obj_pose_cc = np.eye(4)
    obj_pose_cc[:3, :3] = obj_r
    obj_pose_cc[:3, 3] = obj_t

    # Compute camera pose in world coordinates
    camera_pose_wc = invert_se3(obj_pose_wc @ invert_se3(obj_pose_cc))

    return camera_pose_wc[:3, :3], camera_pose_wc[:3, 3].flatten()


def create_colmap_reconstruction(scene_data: dict) -> pycolmap.Reconstruction:
    logger.info("Creating COLMAP reconstruction from scene data")
    reconstruction = pycolmap.Reconstruction()

    camera = pycolmap.Camera.create(
        camera_id=1,
        model=pycolmap.CameraModelId.PINHOLE,
        focal_length=scene_data["cam_params"]["K"][0, 0],
        width=scene_data["cam_params"]["im_size"][0],
        height=scene_data["cam_params"]["im_size"][1],
    )
    camera.params[1] = scene_data["cam_params"]["K"][1, 1]
    camera.params[2] = scene_data["cam_params"]["K"][0, 2]
    camera.params[3] = scene_data["cam_params"]["K"][1, 2]
    reconstruction.add_camera(camera)

    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(reconstruction.camera(1).sensor_id)
    reconstruction.add_rig(rig)

    image_id = 1
    for frame_id, frame_data in sorted(scene_data["scene_gt"].items()):
        assert len(frame_data) == 1, "Expected only one object in the frame"
        cam_r_wc = frame_data[0]["cam_R_m2c"]
        cam_t_wc = frame_data[0]["cam_t_m2c"].flatten()

        rotation = Rotation.from_matrix(cam_r_wc)

        pose = pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(rotation.as_quat()),
            translation=cam_t_wc,
        )

        data = pycolmap.data_t(
            sensor_id=camera.sensor_id,
            id=image_id,
        )

        frame = pycolmap.Frame(
            rig_id=rig.rig_id,
            rig=reconstruction.rig(1),
            frame_id=image_id,
        )
        frame.add_data_id(data)
        frame.set_cam_from_world(1, pose)
        reconstruction.add_frame(frame)

        image_name = f"{int(frame_id):06d}.png"

        image = pycolmap.Image(
            name=image_name,
            camera_id=1,
            camera=reconstruction.camera(1),
            image_id=image_id,
            frame_id=frame.frame_id,
            frame=reconstruction.frame(frame.frame_id),
        )
        reconstruction.add_image(image)
        image_id += 1

    return reconstruction


def create_colmap_reconstruction_from_two_videos(
    scene_data: dict,
) -> pycolmap.Reconstruction:
    """Create COLMAP reconstruction from scene data."""
    logger.info("Creating COLMAP reconstruction from scene data")

    # Get reference poses and compute object's world pose
    # Place the object at the origin of the world coordinates with no rotation
    obj_center_wc = np.zeros(3)
    obj_rotation_wc = np.eye(3)

    # Initialize COLMAP reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Add camera models for each sequence
    camera_id = 1
    camera_ids = dict()  # sequence_name -> camera_id
    for sequence_name, sequence_data in scene_data.items():
        # Skip empty sequences. Some objects only have one onboarding sequence.
        if not sequence_data:
            continue

        # Get a sample camera from this sequence for the intrinsic parameters
        sample_camera = next(iter(sequence_data["scene_camera"].values()))

        # Create PINHOLE camera
        camera = pycolmap.Camera.create(
            camera_id=camera_id,
            model=pycolmap.CameraModelId.PINHOLE,
            focal_length=sample_camera["cam_K"][0, 0],
            width=sample_camera["width"],
            height=sample_camera["height"],
        )
        camera.params[1] = sample_camera["cam_K"][1, 1]
        camera.params[2] = sample_camera["cam_K"][0, 2]
        camera.params[3] = sample_camera["cam_K"][1, 2]
        reconstruction.add_camera(camera)

        # Create rig
        rig = pycolmap.Rig(rig_id=camera_id)
        rig.add_ref_sensor(reconstruction.camera(camera_id).sensor_id)
        reconstruction.add_rig(rig)

        camera_ids[sequence_name] = camera_id
        camera_id += 1

    # Process each sequence (sorted by sequence name, then by frame id)
    image_id = 1
    for sequence_name, sequence_data in scene_data.items():
        camera_id = camera_ids[sequence_name]
        camera = reconstruction.camera(camera_id)
        rig = reconstruction.rig(camera_id)

        for frame_id, frame_data in sorted(sequence_data["scene_gt"].items()):
            assert len(frame_data) == 1, "Expected only one object in the frame"
            obj_r = frame_data[0]["cam_R_m2c"]
            obj_t = frame_data[0]["cam_t_m2c"].flatten()

            cam_r_wc, cam_t_wc = compute_camera_pose_from_object_pose(
                obj_r, obj_t, obj_center_wc, obj_rotation_wc
            )

            # Convert rotation matrix to quaternion for COLMAP
            rotation = Rotation.from_matrix(cam_r_wc)

            # Create Rigid3d pose object
            pose = pycolmap.Rigid3d(
                rotation=pycolmap.Rotation3d(rotation.as_quat()),
                translation=cam_t_wc,
            )

            # Create data
            data = pycolmap.data_t(
                sensor_id=camera.sensor_id,
                id=image_id,
            )

            # Create frame and set pose
            frame = pycolmap.Frame(
                rig_id=rig.rig_id,
                rig=rig,
                frame_id=image_id,
            )
            frame.add_data_id(data)
            frame.set_cam_from_world(camera_id, pose)
            reconstruction.add_frame(frame)

            # Create image name to match the BOP naming convention
            image_name = f"{int(frame_id):06d}_{sequence_name}.jpg"

            # Create image
            image = pycolmap.Image(
                name=image_name,
                camera_id=camera_id,
                camera=camera,
                image_id=image_id,
                frame_id=frame.frame_id,
                frame=reconstruction.frame(frame.frame_id),
            )
            reconstruction.add_image(image)
            image_id += 1

    return reconstruction


def write_colmap_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    base_output_path: Path,
    object_id: str,
) -> None:
    """Write COLMAP reconstruction to directory."""
    logger.info("Writing COLMAP reconstruction to directory %s", base_output_path)
    # Create output directory structure
    base_output_path.mkdir(parents=True, exist_ok=True)
    colmap_output_path = Path(base_output_path) / object_id / "sparse" / "0"
    colmap_output_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(colmap_output_path))

    database_path = Path(base_output_path) / object_id / "database.db"
    database = pycolmap.Database.open(database_path)
    for camera_id in range(1, reconstruction.num_cameras() + 1):
        camera = reconstruction.camera(camera_id)
        database.write_camera(camera)
    for rig_id in range(1, reconstruction.num_rigs() + 1):
        rig = reconstruction.rig(rig_id)
        database.write_rig(rig)
    for frame_id in range(1, reconstruction.num_frames() + 1):
        frame = reconstruction.frame(frame_id)
        database.write_frame(frame)
    for image_id in range(1, reconstruction.num_images() + 1):
        image = reconstruction.image(image_id)
        database.write_image(image)
    database.close()


def create_sparse_reconstruction(
    colmap_path: Path,
):
    """Create sparse reconstruction using COLMAP feature extraction and matching."""
    logger.info(
        "Creating sparse reconstruction using COLMAP feature extraction and matching"
    )
    database_path = colmap_path / "database.db"
    images_path = colmap_path / "images"
    masks_path = colmap_path / "masks"
    logger.info("Extracting features using COLMAP feature extractor")
    feature_extractor_command = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_path),
        "--ImageReader.mask_path",
        str(masks_path),
    ]
    feature_extractor_result = subprocess.run(
        feature_extractor_command, capture_output=True, text=True, check=False
    )
    if feature_extractor_result.returncode != 0:
        logger.error(feature_extractor_result.stderr)
        raise RuntimeError(
            f"Feature extractor failed: {feature_extractor_result.stderr}"
        )

    logger.info("Matching sequential using COLMAP sequential matcher")
    match_sequential_command = [
        "colmap",
        "sequential_matcher",
        "--database_path",
        str(database_path),
    ]
    match_sequential_result = subprocess.run(
        match_sequential_command, capture_output=True, text=True, check=False
    )
    if match_sequential_result.returncode != 0:
        logger.error(match_sequential_result.stderr)
        raise RuntimeError(f"Match sequential failed: {match_sequential_result.stderr}")

    logger.info("Triangulating points using COLMAP point triangulator")
    manual_model_path = colmap_path / "sparse" / "0"
    sparse_model_path = colmap_path / "sparse" / "1"
    sparse_model_path.mkdir(parents=True, exist_ok=True)
    triangulate_points_command = [
        "colmap",
        "point_triangulator",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_path),
        "--input_path",
        str(manual_model_path),
        "--output_path",
        str(sparse_model_path),
    ]
    triangulate_points_result = subprocess.run(
        triangulate_points_command, capture_output=True, text=True, check=False
    )
    if triangulate_points_result.returncode != 0:
        logger.error(triangulate_points_result.stderr)
        raise RuntimeError(
            f"Triangulate points failed: {triangulate_points_result.stderr}"
        )


def get_colmap_camera_info(colmap_path: Path):
    """
    Get COLMAP camera intrinsics and image dimensions in one call.

    Args:
        colmap_path: Path to COLMAP directory

    Returns:
        tuple: (intrinsics_matrix, width, height)
    """
    database_path = colmap_path / "database.db"
    database = pycolmap.Database.open(database_path)
    camera = database.read_camera(1)

    intrinsics = np.array(
        [
            [camera.params[0], 0, camera.params[2]],
            [0, camera.params[1], camera.params[3]],
            [0, 0, 1],
        ]
    )

    return intrinsics, camera.width, camera.height
