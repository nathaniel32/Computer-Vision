import trimesh
import numpy as np
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

def save_point_cloud_in_pcd(points, colors_int, out_dir, label=None, filename="point_cloud.pcd"):
    pcd_out_path = os.path.join(out_dir, filename)
    print("Writing PCD file (ASCII format)...")
    with open(pcd_out_path, 'w') as f:
        f.write('VERSION .7\n')
        f.write('FIELDS x y z rgb\n')
        f.write('SIZE 4 4 4 4\n')
        f.write('TYPE F F F U\n')
        f.write('COUNT 1 1 1 1\n')
        f.write(f'WIDTH {len(points)}\n')
        f.write('HEIGHT 1\n')
        f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.write(f'POINTS {len(points)}\n')
        f.write('DATA ascii\n')
        
        for i in range(len(points)):
            x, y, z = points[i]
            rgb = colors_int[i]
            f.write(f"{x} {y} {z} {rgb}\n")
    
    print(f"\n- Point cloud saved: {pcd_out_path}")

def convert_mesh_to_point_cloud(obj_path: str, total_num_points: int):
    print("Loading mesh...")
    # Load as scene to get all materials and textures
    scene = trimesh.load(obj_path, force='scene', process=False)
    
    # Get the mesh from scene
    if isinstance(scene, trimesh.Scene):
        # Combine all geometries in the scene
        mesh = trimesh.util.concatenate([
            geom for geom in scene.geometry.values() 
            if isinstance(geom, trimesh.Trimesh)
        ])
    else:
        mesh = scene
    
    if mesh.visual.uv is None:
        raise ValueError("Mesh does not have UV coordinates!")
    
    print(f"Mesh info:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Has UV: {mesh.visual.uv is not None}")
    
    # Get texture image from mesh visual
    print("Extracting texture from mesh...")
    
    if hasattr(mesh.visual, 'material'):
        # Try to get texture from material
        if hasattr(mesh.visual.material, 'image'):
            texture_img = mesh.visual.material.image
        elif hasattr(mesh.visual.material, 'baseColorTexture'):
            texture_img = mesh.visual.material.baseColorTexture
        else:
            # Try to convert visual to texture
            texture_img = mesh.visual.to_texture()
    else:
        raise ValueError("No texture information found in mesh!")
    
    # Convert PIL Image to numpy array
    if hasattr(texture_img, 'convert'):  # PIL Image
        texture = np.array(texture_img.convert('RGB'), dtype=np.uint8)
    else:
        texture = np.array(texture_img, dtype=np.uint8)
    
    print(f"Texture shape: {texture.shape}")
    h, w = texture.shape[:2]
    
    print(f"Sampling {total_num_points} points...")
    points, face_indices = mesh.sample(total_num_points, return_index=True)
    
    # Get UV coordinates for sampled points
    faces_uv = mesh.visual.uv[mesh.faces[face_indices]]
    triangles = mesh.vertices[mesh.faces[face_indices]]
    
    print("Computing barycentric coordinates...")
    bary = _barycentric_coords_batch(points, triangles)
    
    print("Interpolating UV coordinates...")
    uv_interpolated = (bary[:, 0:1] * faces_uv[:, 0] + 
                       bary[:, 1:2] * faces_uv[:, 1] + 
                       bary[:, 2:3] * faces_uv[:, 2])
    
    print("Sampling texture colors...")
    # Convert UV to pixel coordinates
    px = np.clip(uv_interpolated[:, 0] * (w - 1), 0, w - 1).astype(int)
    py = np.clip((1 - uv_interpolated[:, 1]) * (h - 1), 0, h - 1).astype(int)
    
    # Sample colors from texture
    colors_rgb = texture[py, px]
    
    # Convert to packed RGB integers
    colors_int = _rgb_to_int_batch(colors_rgb)

    print(f"- Points: {len(points)}")
    
    return points, colors_int, colors_rgb

def convert_mesh_to_point_cloud_folder(dir_path, total_num_points, visualize=False):
    obj_path, textures_path = load_mesh_map(dir_path)

    try:
        print("\nCreating point cloud...")
        points, colors_int, colors_rgb = convert_mesh_to_point_cloud(obj_path, total_num_points=total_num_points)

        if visualize:
            _visualize_point_cloud(points, colors_rgb)

        return points, colors_int, obj_path

    except Exception as e:
        raise RuntimeError(f"Error processing folder: {dir_path}") from e