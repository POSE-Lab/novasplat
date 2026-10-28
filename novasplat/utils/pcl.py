import numpy as np
import open3d as o3d


def poisson_disk_subsample(
    input_ply: str,
    output_ply: str,
    n_points: int = 5000,
    init_factor: int = 5,
):
    """
    Poisson-disk subsample a point cloud.

    Parameters
    ----------
    input_ply : str
        Input PLY point cloud.
    output_ply : str
        Output PLY point cloud.
    n_points : int
        Target number of points.
    init_factor : int
        Oversampling factor (higher = better blue-noise).
    """

    mesh = o3d.io.read_triangle_mesh(input_ply)
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=n_points, init_factor=init_factor
    )
    n_pts = np.asarray(pcd.points).shape[0]
    pcd.colors = o3d.utility.Vector3dVector(np.ones((n_pts, 3)))
    o3d.io.write_point_cloud(output_ply, pcd)


def load_point_cloud(ply_path, device="cuda"):
    """Load a point cloud from a PLY file."""
    pcd = o3d.io.read_point_cloud(ply_path)
    return pcd


def main_poisson_disk_subsample():
    import argparse

    parser = argparse.ArgumentParser(
        description="Poisson-disk subsample a point cloud using Open3D."
    )
    parser.add_argument("input_ply", type=str, help="Input PLY point cloud.")
    parser.add_argument("output_ply", type=str, help="Output PLY point cloud.")
    parser.add_argument(
        "--n_points",
        type=int,
        default=5000,
        help="Target number of points (default: 5000).",
    )
    parser.add_argument(
        "--init_factor",
        type=int,
        default=5,
        help="Oversampling factor (default: 5).",
    )
    args = parser.parse_args()

    poisson_disk_subsample(
        args.input_ply,
        args.output_ply,
        n_points=args.n_points,
        init_factor=args.init_factor,
    )


if __name__ == "__main__":
    main_poisson_disk_subsample()
