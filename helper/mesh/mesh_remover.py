from scipy.spatial import cKDTree
import trimesh
import numpy as np
from helper.mesh.mesh_utils import get_cluster_labels

def filter_largest_cluster(xyz):    
    cluster_labels = get_cluster_labels(xyz)
    
    if (cluster_labels < 0).all():
        return xyz
    
    largest_cluster_label = np.argmax(np.bincount(cluster_labels[cluster_labels >= 0]))
    indices = np.where(cluster_labels == largest_cluster_label)[0]
    
    return xyz[indices]

def smooth_labels(points, pred_label, k=50):
    tree = cKDTree(points)
    new_label = np.copy(pred_label)
    
    for i, p in enumerate(points):
        dists, idx = tree.query(p, k=k)
        neighbor_labels = pred_label[idx]
        counts = np.bincount(neighbor_labels)
        new_label[i] = np.argmax(counts)
        
    return new_label

def remove_object_part_v1(points, pred_label, mesh_file_path, save_obj_trim_path, keep_label):
    # Load mesh
    mesh = trimesh.load(mesh_file_path, process=False)
    original_faces = len(mesh.faces)

    pred_label = smooth_labels(points=points, pred_label=pred_label)

    # Take points to be deleted
    points_to_remove = points[pred_label != keep_label]

    if len(points_to_remove) > 0:
        # KDTree for vertex filter
        kdtree = cKDTree(points_to_remove)

        # The closest distance of each vertex to points_to_remove
        distances, _ = kdtree.query(mesh.vertices, k=1)

        # Average distance between points_to_remove itself
        distances_nn, _ = kdtree.query(points_to_remove, k=2)
        mean_dist = np.mean(distances_nn[:, 1])
        threshold = mean_dist * 2

        # Mask for the vertices you want to keep
        mask_keep = distances > threshold
    else:
        # If no label matches, keep all vertices.
        mask_keep = np.ones(len(mesh.vertices), dtype=bool)

    # Delete faces that contain vertices to be deleted.
    faces_keep = mask_keep[mesh.faces].all(axis=1)
    mesh.update_faces(faces_keep)
    mesh.remove_unreferenced_vertices()

    # Save mesh
    mesh.export(save_obj_trim_path)

    print(f"Mesh trimmed saved to: {save_obj_trim_path}")
    print(f"Original faces: {original_faces} -> Remaining faces: {len(mesh.faces)}")

def remove_object_part_v2(points, pred_label, mesh_file_path, save_obj_trim_path, keep_label, k_smooth=20, dilation_ratio=1.3):
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

def remove_object_part_v3(points, pred_label, mesh_file_path, save_obj_trim_path, keep_label, k_smooth=20, padding=0.05):
    """ Cut the object with a bounding box - only keep the region labeled keep_label """
    
    mesh = trimesh.load(mesh_file_path, process=False)
    original_faces = len(mesh.faces)
    
    print("Step 1: Smoothing labels...")
    pred_label = smooth_labels(points=points, pred_label=pred_label, k=k_smooth)
    
    print("Step 2: Computing bounding box for keep region...")
    points_to_keep = points[pred_label == keep_label]
    
    if len(points_to_keep) < 10:
        print("Warning: Too few points to keep")
        return None
    
    # Calculate the bounding box of the points you want to keep.
    bbox_min = points_to_keep.min(axis=0) - padding
    bbox_max = points_to_keep.max(axis=0) + padding
    
    print(f"Bounding box: min={bbox_min}, max={bbox_max}")
    
    # Step 4: Define 6 planes dari bounding box
    # Format: (plane_origin, plane_normal)
    planes = [
        # X-min plane (normal menunjuk ke kanan/+X)
        (bbox_min, np.array([1.0, 0.0, 0.0])),
        # X-max plane (normal menunjuk ke kiri/-X)
        (bbox_max, np.array([-1.0, 0.0, 0.0])),
        # Y-min plane (normal menunjuk ke atas/+Y)
        (bbox_min, np.array([0.0, 1.0, 0.0])),
        # Y-max plane (normal menunjuk ke bawah/-Y)
        (bbox_max, np.array([0.0, -1.0, 0.0])),
        # Z-min plane (normal menunjuk ke depan/+Z)
        (bbox_min, np.array([0.0, 0.0, 1.0])),
        # Z-max plane (normal menunjuk ke belakang/-Z)
        (bbox_max, np.array([0.0, 0.0, -1.0])),
    ]
    
    # Slice mesh with all 6 planes in sequence
    print("Step 3: Slicing mesh with bounding box planes...")
    result_mesh = mesh
    for i, (plane_origin, plane_normal) in enumerate(planes):
        if result_mesh is None or len(result_mesh.faces) == 0:
            print(f"Warning: Mesh became empty after plane {i}")
            break
        
        result_mesh = result_mesh.slice_plane(
            plane_origin=plane_origin,
            plane_normal=plane_normal
        )
        
        if result_mesh is not None:
            print(f"  Plane {i+1}/6: {len(result_mesh.faces)} faces remaining")
    
    if result_mesh is None or len(result_mesh.faces) == 0:
        print("Error: No mesh remaining after slicing")
        return None
    
    print("Step 4: Post-processing...")
    
    # Remove unreferenced vertices
    result_mesh.remove_unreferenced_vertices()
    
    # Remove degenerate faces
    result_mesh.remove_degenerate_faces()
    
    # Optional: Fill small holes
    if len(result_mesh.faces) > 100:
        original_mesh_faces = len(result_mesh.faces)
        result_mesh.fill_holes()
        print(f"  Filled holes: {len(result_mesh.faces) - original_mesh_faces} faces added")
    
    # Clean up again
    result_mesh.remove_unreferenced_vertices()
    
    print("Step 5: Saving mesh...")
    result_mesh.export(save_obj_trim_path)
    
    print(f"\nResult:")
    print(f"  Original faces: {original_faces}")
    print(f"  Remaining faces: {len(result_mesh.faces)}")
    print(f"  Reduction: {(1 - len(result_mesh.faces)/original_faces)*100:.1f}%")
    print(f"  Saved to: {save_obj_trim_path}")
    
    return result_mesh