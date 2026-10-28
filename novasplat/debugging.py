"""
Debugging utilities for the novasplat project.
"""

import bpy
import bmesh

import numpy as np


def export_mesh_as_ply(mesh_obj, filepath):
    """Export a mesh object as PLY file using bmesh."""
    # Get the mesh data
    mesh = mesh_obj.data

    # Create a new bmesh instance
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Ensure we're in object mode
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode="OBJECT")

    # Write PLY file manually
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(mesh.vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(mesh.polygons)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Write vertices
        for vertex in mesh.vertices:
            f.write(f"{vertex.co.x} {vertex.co.y} {vertex.co.z}\n")

        # Write faces
        for poly in mesh.polygons:
            f.write(f"{len(poly.vertices)}")
            for vertex_idx in poly.vertices:
                f.write(f" {vertex_idx}")
            f.write("\n")

    bm.free()


def visualize_vectors_as_ply(
    positions, angles, output_path="vectors.ply", vector_length=0.1
):
    """
    Create a PLY file with arrow-like vectors representing the positions and angles.

    Args:
        positions: List of (x, y, z) positions
        angles: List of (elevation, azimuth) angles
        output_path: Output PLY file path
        vector_length: Length of the arrow vectors
    """
    vertices = []
    faces = []
    vertex_count = 0

    for pos, angle in zip(positions, angles):
        x, y, z = pos
        el, az = angle

        # Calculate vector direction from angles
        # Convert spherical to cartesian coordinates
        dir_x = np.cos(el) * np.sin(az)
        dir_y = np.cos(el) * np.cos(az)
        dir_z = np.sin(el)

        # Create arrow geometry (cone + cylinder)
        # Arrow tip (cone)
        tip_x = x + dir_x * vector_length
        tip_y = y + dir_y * vector_length
        tip_z = z + dir_z * vector_length

        # Arrow base (cylinder end)
        base_x = x + dir_x * vector_length * 0.3
        base_y = y + dir_y * vector_length * 0.3
        base_z = z + dir_z * vector_length * 0.3

        # Create a small cone for the arrow tip
        cone_radius = vector_length * 0.05

        # Cone vertices (4 vertices for a simple cone)
        cone_center_x = tip_x
        cone_center_y = tip_y
        cone_center_z = tip_z

        # Cone base vertices (circle)
        for j in range(4):
            angle_offset = j * np.pi / 2
            perp_x = np.cos(el + np.pi / 2) * np.cos(az + angle_offset)
            perp_y = np.cos(el + np.pi / 2) * np.sin(az + angle_offset)
            perp_z = np.sin(el + np.pi / 2)

            cone_base_x = base_x + perp_x * cone_radius
            cone_base_y = base_y + perp_y * cone_radius
            cone_base_z = base_z + perp_z * cone_radius

            vertices.append((cone_base_x, cone_base_y, cone_base_z))

        # Cone tip
        vertices.append((cone_center_x, cone_center_y, cone_center_z))

        # Create cone faces (4 triangular faces)
        for j in range(4):
            next_j = (j + 1) % 4
            # Each face: cone base vertex j, cone base vertex next_j, cone tip
            faces.append((vertex_count + j, vertex_count + next_j, vertex_count + 4))

        vertex_count += 5  # 4 base vertices + 1 tip

    # Write PLY file
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Write vertices
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Vector visualization saved to {output_path}")
    print(f"Created {len(vertices)} vertices and {len(faces)} faces")
