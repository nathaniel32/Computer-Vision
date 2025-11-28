import trimesh
import numpy as np
from PIL import Image
import os

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

def _visualize_pointcloud(points, colors_rgb):
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

# ============= LOAD TEXTURE =============
def load_mesh_map(foldername):
    filenames = os.listdir(foldername)

    mesh_file = None
    tex_file = None
    ao_file = None
    norm_file = None
    
    for f in filenames:
        f_lower = f.lower()
        full_path = os.path.join(foldername, f)
        
        if f.endswith(".obj"):
            mesh_file = full_path
        elif f.endswith(".png"):
            if 'tex' in f_lower or 'diffuse' in f_lower or 'albedo' in f_lower:
                tex_file = full_path
            elif 'ao' in f_lower or 'occlusion' in f_lower:
                ao_file = full_path
            elif 'norm' in f_lower or 'normal' in f_lower:
                norm_file = full_path
    
    return mesh_file, tex_file, ao_file, norm_file

# ============= MESH TO POINT CLOUD =============
def mesh_to_point_cloud(mesh_path, texture_path, save_path, num_points):
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path, force='mesh')
    
    if mesh.visual.uv is None:
        raise ValueError("Mesh does not have UV coordinates!")
    
    print("Loading texture...")
    texture_img = Image.open(texture_path).convert('RGB')
    texture = np.array(texture_img, dtype=np.uint8)

    h, w = texture.shape[:2]
    
    print(f"Sampling {num_points} points...")
    points, face_indices = mesh.sample(num_points, return_index=True)
    
    # Get UV coordinates and triangle vertices for sampled points
    faces_uv = mesh.visual.uv[mesh.faces[face_indices]]
    triangles = mesh.vertices[mesh.faces[face_indices]]
    
    print("Computing barycentric coordinates...")
    bary = _barycentric_coords_batch(points, triangles)
    
    print("Interpolating UV coordinates...")
    # Interpolate UV using barycentric coordinates
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
    rgb_ints = _rgb_to_int_batch(colors_rgb)
    
    print("Writing PCD file (ASCII format)...")
    with open(save_path, 'w') as f:
        # Write header
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
        
        # Write point data
        for i in range(len(points)):
            x, y, z = points[i]
            rgb = rgb_ints[i]
            f.write(f"{x} {y} {z} {rgb}\n")
    
    print(f"- Point cloud saved to: {save_path}")
    print(f"- Points: {len(points)}")
    
    return points, rgb_ints, colors_rgb

def convert_mesh_folder_to_pcd(input_dir, save_pcd_path, num_points, visualize=False):
    mesh_file, tex_file, ao_file, norm_file = load_mesh_map(input_dir)

    if mesh_file is None:
        raise FileNotFoundError("No .obj file found in this folder.")

    if tex_file is None:
        raise FileNotFoundError("No texture file found in this folder.")

    try:
        print("\nCreating point cloud...")
        points, rgb_ints, colors_rgb = mesh_to_point_cloud(
            mesh_file,
            tex_file,
            save_pcd_path,
            num_points=num_points
        )

        if visualize:
            _visualize_pointcloud(points, colors_rgb)

        return points, rgb_ints, colors_rgb, mesh_file, tex_file, ao_file, norm_file

    except Exception as e:
        raise RuntimeError(f"Error processing folder: {input_dir}") from e

def get_chunks_indices(n_data, chunk_size):
    rand_indices = np.random.permutation(n_data)
    chunks_indices = [rand_indices[i:i + chunk_size] for i in range(0, n_data, chunk_size)]
    return chunks_indices

def make_dataset():
    print("""Expected folder structure:
    input_dir/
        data_1/
            - mesh.obj
            - mesh.mtl
            - mesh_tex0.png
        data_2/
            - mesh.obj
            - mesh.mtl
            - mesh_tex0.png
        ...
    """)
    input_dir = input('Input directory: ')
    out_dir = input('Output directory: ')

    os.makedirs(out_dir, exist_ok=True)

    processed_count = 0
    
    for foldername, subfolders, filenames in os.walk(input_dir):
        if not filenames:
            continue

        clean_name = (
            os.path.basename(foldername)
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        save_pcd_path = os.path.join(out_dir, clean_name + ".pcd")

        try:
            convert_mesh_folder_to_pcd(foldername, save_pcd_path=save_pcd_path, num_points=100000)
            processed_count += 1

        except Exception as e:
            print(f"✗ Failed to process {foldername}: {e}")

    print(f"\n{'='*60}")
    print(f"- ALL DONE! Processed {processed_count} meshes")
    print(f"{'='*60}")