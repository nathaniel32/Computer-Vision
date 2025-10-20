from scipy.spatial import cKDTree
import trimesh
import numpy as np

def smooth_labels(points, pred_label, k=20):
    """Smoothing lebih aggressive untuk hasil lebih clean"""
    tree = cKDTree(points)
    new_label = np.copy(pred_label)
    
    for i, p in enumerate(points):
        dists, idx = tree.query(p, k=k)
        neighbor_labels = pred_label[idx]
        counts = np.bincount(neighbor_labels)
        new_label[i] = np.argmax(counts)
        
    return new_label

def remove_object_part_v1(points, pred_label, mesh_file_path, save_obj_trim_path, keep_label_id):
    # --- Load mesh
    mesh = trimesh.load(mesh_file_path, process=False)
    original_faces = len(mesh.faces)

    pred_label = smooth_labels(points=points, pred_label=pred_label)

    # --- Ambil points yang akan dihapus
    points_to_remove = points[pred_label != keep_label_id]

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

def remove_object_part_v2(points, pred_label, mesh_file_path, save_obj_trim_path, keep_label_id, k_smooth=20, dilation_ratio=1.3):
    """
    Potong object dengan hasil rapi dan minimal noise
    
    Args:
        k_smooth: jumlah neighbors untuk smoothing (lebih besar = lebih smooth)
        dilation_ratio: multiplier untuk threshold (lebih kecil = potongan lebih clean)
    """
    
    mesh = trimesh.load(mesh_file_path, process=False)
    original_faces = len(mesh.faces)
    
    # Step 1: Aggressive smoothing untuk eliminate noise
    print("Step 1: Smoothing labels...")
    pred_label = smooth_labels(points=points, pred_label=pred_label, k=k_smooth)
    
    # Step 2: Identifikasi region yang akan dihapus
    print("Step 2: Calculating removal region...")
    points_to_remove = points[pred_label != keep_label_id]
    
    if len(points_to_remove) < 10:
        print("Warning: Terlalu sedikit points untuk dihapus")
        return None
    
    # Step 3: Hitung threshold secara smart
    kdtree = cKDTree(points_to_remove)
    k_neighbors = min(5, len(points_to_remove) - 1)
    distances_nn, _ = kdtree.query(points_to_remove, k=k_neighbors+1)
    
    # Ambil rata-rata dari k-nearest neighbor (skip index 0 yang adalah diri sendiri)
    mean_dist = np.mean(distances_nn[:, 1:])
    threshold = mean_dist * dilation_ratio
    
    print(f"Mean distance: {mean_dist:.4f}, Threshold: {threshold:.4f}")
    
    # Step 4: Filter vertices
    distances, _ = kdtree.query(mesh.vertices, k=1)
    mask_keep = distances > threshold
    
    # Step 5: Filter faces dan remove unreferenced vertices
    print("Step 3: Filtering faces...")
    faces_keep = mask_keep[mesh.faces].all(axis=1)
    mesh.update_faces(faces_keep)
    mesh.remove_unreferenced_vertices()
    
    # Step 6: Post-processing untuk hasil lebih rapi
    print("Step 4: Post-processing...")
    
    # Fill small holes
    if len(mesh.faces) > 100:
        original_mesh_faces = len(mesh.faces)
        mesh.fill_holes()
        print(f"  Filled holes: {len(mesh.faces) - original_mesh_faces} faces added")
    
    # Remove isolated small components (noise)
    mesh.remove_degenerate_faces()
    
    # Split dan ambil komponen terbesar (buang noise kecil)
    #components = mesh.split(only_watertight=False)
    #if len(components) > 1:
    #    print(f"  Found {len(components)} components, keeping largest...")
    #    mesh = max(components, key=lambda x: len(x.faces))
    
    mesh.remove_unreferenced_vertices()
    
    # Step 7: Simpan
    print("Step 5: Saving mesh...")
    mesh.export(save_obj_trim_path)
    
    print(f"\n✓ Result:")
    print(f"  Original faces: {original_faces}")
    print(f"  Remaining faces: {len(mesh.faces)}")
    print(f"  Reduction: {(1 - len(mesh.faces)/original_faces)*100:.1f}%")
    print(f"  Saved to: {save_obj_trim_path}")
    
    return mesh