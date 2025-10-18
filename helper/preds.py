import trimesh
import numpy as np
from PIL import Image
import trimesh
import numpy as np
from scipy.spatial import cKDTree

# ============= BARYCENTRIC COORDINATES =============
def barycentric_coords(p, tri):
    """Hitung koordinat barycentric untuk interpolasi UV"""
    v0, v1, v2 = tri
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)
    denom = d00 * d11 - d01 * d01
    
    if abs(denom) < 1e-10:
        return np.array([1/3, 1/3, 1/3])
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return np.array([u, v, w])

# ============= RGB TO INTEGER (Compatible dengan THREE.PCDLoader) =============
def rgb_to_int(r, g, b):
    """
    Konversi RGB (0-255) ke integer format THREE.PCDLoader
    Formula: b + 256*g + 256*256*r
    """
    return int(b) + 256 * int(g) + 256 * 256 * int(r)

def int_to_rgb(rgb_int):
    """
    Konversi integer RGB kembali ke R, G, B
    """
    b = rgb_int & 0xFF
    g = (rgb_int >> 8) & 0xFF
    r = (rgb_int >> 16) & 0xFF
    return r, g, b

# ============= SAVE POINT CLOUD DENGAN RGB INTEGER =============
def mesh_to_point_cloud(mesh_path, texture_path, save_path, num_points=100000):
    """
    Simpan point cloud dari mesh dengan texture ke PCD format (ASCII)
    RGB disimpan sebagai single integer field (kompatibel THREE.PCDLoader)
    """
    print("Loading mesh...")
    mesh = trimesh.load(mesh_path, force='mesh')
    
    if mesh.visual.uv is None:
        raise ValueError("Mesh tidak memiliki UV coordinates!")
    
    print("Loading texture...")
    texture_image = Image.open(texture_path).convert('RGB')
    texture = np.array(texture_image)  # uint8, range 0-255
    
    print(f"Sampling {num_points} points...")
    points, face_indices = mesh.sample(num_points, return_index=True)
    
    faces_uv = mesh.visual.uv[mesh.faces[face_indices]]
    triangles = mesh.vertices[mesh.faces[face_indices]]
    
    print("Mapping UV ke texture...")
    rgb_ints = []
    colors_rgb = []
    
    for i, p in enumerate(points):
        if (i + 1) % 20000 == 0:
            print(f"  Progress: {i+1}/{num_points}")
        
        tri = triangles[i]
        uv_tri = faces_uv[i]
        bary = barycentric_coords(p, tri)
        uv = bary[0]*uv_tri[0] + bary[1]*uv_tri[1] + bary[2]*uv_tri[2]
        
        h, w, _ = texture.shape
        px = int(np.clip(uv[0], 0, 1) * (w - 1))
        py = int(np.clip(1 - uv[1], 0, 1) * (h - 1))
        color = texture[py, px]
        
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        rgb_int = rgb_to_int(r, g, b)
        
        colors_rgb.append([r, g, b])
        rgb_ints.append(rgb_int)
    
    rgb_ints = np.array(rgb_ints, dtype=np.uint32)
    colors_rgb = np.array(colors_rgb, dtype=np.uint8)
    
    # Tulis ke PCD file (ASCII format, kompatibel THREE.PCDLoader)
    print("Writing PCD file (ASCII format)...")
    with open(save_path, 'w') as f:
        # Header
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
        
        # ASCII data: x y z rgb
        for i in range(len(points)):
            x, y, z = points[i]
            rgb = rgb_ints[i]
            f.write(f"{x} {y} {z} {rgb}\n")
    
    print(f"✓ Point cloud disimpan ke: {save_path}")
    print(f"  Points: {len(points)}")
    print(f"  Format: PCD ASCII dengan RGB integer (kompatibel THREE.PCDLoader)")
    print(f"  RGB conversion: b + 256*g + 256*256*r")
    
    return points, rgb_ints, colors_rgb

# ============= READ POINT CLOUD =============
def read_pointcloud_pcd(pcd_path):
    """
    Baca PCD file (ASCII) dan return points + RGB integer + RGB decomposed
    """
    print(f"\nReading point cloud dari: {pcd_path}")
    
    with open(pcd_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            data_start = i + 1
            break
    
    # Parse ASCII data
    points = []
    rgb_ints = []
    colors_rgb = []
    
    for line in lines[data_start:]:
        values = line.strip().split()
        if len(values) >= 4:
            x, y, z = float(values[0]), float(values[1]), float(values[2])
            rgb_int = int(values[3])
            r, g, b = int_to_rgb(rgb_int)
            
            points.append([x, y, z])
            rgb_ints.append(rgb_int)
            colors_rgb.append([r, g, b])
    
    points = np.array(points, dtype=np.float32)
    rgb_ints = np.array(rgb_ints, dtype=np.uint32)
    colors_rgb = np.array(colors_rgb, dtype=np.uint8)
    
    print(f"✓ Loaded {len(points)} points")
    print(f"  Has RGB: True")
    print(f"Sample RGB integer: {rgb_ints[:5]}")
    print(f"Sample RGB decomposed:\n{colors_rgb[:5]}")
    
    return points, rgb_ints, colors_rgb

def smooth_labels(points, pred_label, k=10):
    tree = cKDTree(points)
    new_label = np.copy(pred_label)
    
    for i, p in enumerate(points):
        dists, idx = tree.query(p, k=k)
        neighbor_labels = pred_label[idx]
        # ambil mayoritas
        counts = np.bincount(neighbor_labels)
        new_label[i] = np.argmax(counts)
        
    return new_label

def remove_object_part(points, pred_label, mesh_file_path, save_obj_trim_path, remove_item_id):
    # --- Load mesh
    mesh = trimesh.load(mesh_file_path, process=False)
    original_faces = len(mesh.faces)

    pred_label = smooth_labels(points=points, pred_label=pred_label)

    # --- Ambil points yang akan dihapus
    points_to_remove = points[pred_label == remove_item_id]

    if len(points_to_remove) > 0:
        # --- KDTree untuk filter vertex
        kdtree = cKDTree(points_to_remove)

        # Jarak terdekat tiap vertex ke points_to_remove
        distances, _ = kdtree.query(mesh.vertices, k=1)

        # Rata-rata jarak antar points_to_remove sendiri
        distances_nn, _ = kdtree.query(points_to_remove, k=2)
        mean_dist = np.mean(distances_nn[:, 1])
        threshold = mean_dist * 2

        # --- Mask untuk vertex yang ingin dipertahankan
        mask_keep = distances > threshold
    else:
        # Jika tidak ada label yang sesuai, pertahankan semua vertex
        mask_keep = np.ones(len(mesh.vertices), dtype=bool)

    # --- Hapus faces yang memiliki vertex yang akan dihapus
    faces_keep = mask_keep[mesh.faces].all(axis=1)
    mesh.update_faces(faces_keep)
    mesh.remove_unreferenced_vertices()

    # --- Simpan mesh
    mesh.export(save_obj_trim_path)

    print(f"Mesh trimmed saved to: {save_obj_trim_path}")
    print(f"Original faces: {original_faces} -> Remaining faces: {len(mesh.faces)}")