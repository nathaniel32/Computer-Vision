import trimesh
import numpy as np
from PIL import Image
import os
from typing import Tuple, List

# ============= BARYCENTRIC COORDINATES =============
def _barycentric_coords_batch(points, triangles):
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]
    
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = points - v0
    
    d00 = np.sum(v0v1 * v0v1, axis=1)
    d01 = np.sum(v0v1 * v0v2, axis=1)
    d11 = np.sum(v0v2 * v0v2, axis=1)
    d20 = np.sum(v0p * v0v1, axis=1)
    d21 = np.sum(v0p * v0v2, axis=1)
    
    denom = d00 * d11 - d01 * d01
    
    # Handle degenerate triangles
    valid = np.abs(denom) > 1e-10
    
    v_coord = np.zeros(len(points))
    w_coord = np.zeros(len(points))
    
    v_coord[valid] = (d11[valid] * d20[valid] - d01[valid] * d21[valid]) / denom[valid]
    w_coord[valid] = (d00[valid] * d21[valid] - d01[valid] * d20[valid]) / denom[valid]
    
    # For degenerate triangles, use centroid
    v_coord[~valid] = 1/3
    w_coord[~valid] = 1/3
    
    u_coord = 1 - v_coord - w_coord
    
    return np.stack([u_coord, v_coord, w_coord], axis=1)

# ============= RGB TO INTEGER =============
def _rgb_to_int_batch(colors_rgb):
    return (colors_rgb[:, 2].astype(np.uint32) + 
            256 * colors_rgb[:, 1].astype(np.uint32) + 
            65536 * colors_rgb[:, 0].astype(np.uint32))

def _visualize_point_cloud(points, colors_rgb):
    try:
        import open3d as o3d
        print("\nVisualizing with Open3D...")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64) / 255.0)
        
        o3d.visualization.draw_geometries(
            [pcd], 
            window_name="Point Cloud", 
            width=1000, 
            height=800
        )
    except ImportError:
        print("- Open3D not installed, skipping visualization")

def _get_mtl_filename(obj_path: str) -> str:
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('mtllib'):
                return line.split()[1]
    raise FileNotFoundError("No .mtl file reference ('mtllib') found in the .obj file.")

def _get_texture_filenames(dir_path, mtl_path: str) -> List[str]:
    textures_path: List[str] = []
    with open(mtl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("map_Kd"):
                parts = line.split()
                if len(parts) > 1:
                    texture_path: str = os.path.join(dir_path, parts[1])
                    textures_path.append(texture_path)

    if not textures_path:
        raise FileNotFoundError("No 'map_Kd' (texture file) found inside the .mtl file.")

    return textures_path

# ============= LOAD TEXTURE =============
def load_mesh_map(dir_path: str) -> Tuple[str, List[str]]:
    filenames: List[str] = os.listdir(dir_path)

    for f in filenames:
        if f.lower().endswith(".obj"):
            obj_path: str = os.path.join(dir_path, f)

            mtl_file: str = _get_mtl_filename(obj_path)
            mtl_path: str = os.path.join(dir_path, mtl_file)

            if not os.path.exists(mtl_path):
                raise FileNotFoundError(f"MTL file '{mtl_file}' not found in the folder.")

            textures_path: List[str] = _get_texture_filenames(dir_path, mtl_path)

            return obj_path, textures_path

    raise FileNotFoundError("No .obj file found in the folder.")

# ============= MESH TO POINT CLOUD =============


def convert_mesh_to_point_cloud_folder(dir_path, pcd_out_path, num_points, visualize=True):
    obj_path, textures_path = load_mesh_map(dir_path)

    if obj_path is None:
        raise FileNotFoundError("No .obj file found in this folder.")

    if textures_path is None:
        raise FileNotFoundError("No texture file found in this folder.")

    try:
        print("\nCreating point cloud...")
        points, colors_int, colors_rgb = convert_mesh_to_point_cloud(
            obj_path,
            textures_path,
            pcd_out_path,
            num_points=num_points
        )

        if visualize:
            _visualize_point_cloud(points, colors_rgb)

        return points, colors_int, obj_path

    except Exception as e:
        raise RuntimeError(f"Error processing folder: {dir_path}") from e