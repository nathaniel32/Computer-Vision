import trimesh
import numpy as np
from pc_color_seg.helper.mesh.mesh_utils import smooth_labels
from scipy.spatial import cKDTree

def remove_object_part(points, pred_label, mesh_file_path, save_obj_trim_path, keep_label, k_smooth=20, dilation_ratio=1.3):
    """ Cut objects with neat results and minimal noise """
    
    mesh = trimesh.load(mesh_file_path, process=False)
    original_faces = len(mesh.faces)
    
    print("Step 1: Smoothing labels...")
    pred_label = smooth_labels(points=points, pred_label=pred_label, k=k_smooth)
    
    print("Step 2: Calculating removal region...")
    points_to_remove = points[pred_label != keep_label]
    
    if len(points_to_remove) < 10:
        print("Warning: Too few points to delete")
        return None
    
    # Calculate threshold smartly
    kdtree = cKDTree(points_to_remove)
    k_neighbors = min(5, len(points_to_remove) - 1)
    distances_nn, _ = kdtree.query(points_to_remove, k=k_neighbors+1)
    
    # Take the average of the k-nearest neighbors (skip index 0 which is yourself)
    mean_dist = np.mean(distances_nn[:, 1:])
    threshold = mean_dist * dilation_ratio
    
    print(f"Mean distance: {mean_dist:.4f}, Threshold: {threshold:.4f}")
    
    # Filter vertices
    distances, _ = kdtree.query(mesh.vertices, k=1)
    mask_keep = distances > threshold
    
    # Filter faces and remove unreferenced vertices
    print("Step 3: Filtering faces...")
    faces_keep = mask_keep[mesh.faces].all(axis=1)
    mesh.update_faces(faces_keep)
    mesh.remove_unreferenced_vertices()
    
    print("Step 4: Post-processing...")
    
    # Fill small holes
    if len(mesh.faces) > 100:
        original_mesh_faces = len(mesh.faces)
        mesh.fill_holes()
        print(f"  Filled holes: {len(mesh.faces) - original_mesh_faces} faces added")
    
    # Remove isolated small components (noise)
    mesh.remove_degenerate_faces()
    
    # Split and take the largest component (discard small noise)
    #components = mesh.split(only_watertight=False)
    #if len(components) > 1:
    #    print(f"  Found {len(components)} components, keeping largest...")
    #    mesh = max(components, key=lambda x: len(x.faces))
    
    mesh.remove_unreferenced_vertices()
    
    print("Step 5: Saving mesh...")
    mesh.export(save_obj_trim_path)
    
    print(f"\nResult:")
    print(f"  Original faces: {original_faces}")
    print(f"  Remaining faces: {len(mesh.faces)}")
    print(f"  Reduction: {(1 - len(mesh.faces)/original_faces)*100:.1f}%")
    print(f"  Saved to: {save_obj_trim_path}")
    
    return mesh