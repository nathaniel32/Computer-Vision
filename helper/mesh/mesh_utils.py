import open3d as o3d
import numpy as np
import trimesh
import os
from scipy.spatial import cKDTree

def get_cluster_labels(xyz, eps=0.02, min_points=10):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    cluster_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return cluster_labels

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

def filter_clusters(xyz):
    cluster_labels = get_cluster_labels(xyz)

    clusters = []
    for label in np.unique(cluster_labels):
        if label == -1:
            continue  # skip noise
        clusters.append(xyz[cluster_labels == label])

    return clusters

def scale_mesh(scale_factor, input_path, out_dir_path):
    mesh = trimesh.load(input_path)
    mesh.apply_scale(scale_factor)
    scale_factor_percent = int(scale_factor * 100)
    mesh.export(os.path.join(out_dir_path, f"scaled_{scale_factor_percent}_percent.obj"))

def read_pcd_label(filename, target_label):
    """
    Read ASCII PCD and extract points with a specific label.
    Automatically detect column positions from FIELDS header.
    """
    points = []
    total_points = 0
    label_counts = {}

    # Parse header to get field order
    field_names = []
    x_idx = y_idx = z_idx = label_idx = None

    with open(filename, 'r') as f:
        header_done = False
        for line in f:
            line = line.strip()

            # Parse FIELDS header
            if line.startswith("FIELDS"):
                field_names = line.split()[1:]  # Skip "FIELDS" keyword
                #print(f"Detected fields: {field_names}")

                # Find indices for x, y, z, label
                for i, field in enumerate(field_names):
                    if field.lower() == 'x':
                        x_idx = i
                    elif field.lower() == 'y':
                        y_idx = i
                    elif field.lower() == 'z':
                        z_idx = i
                    elif field.lower() == 'label':
                        label_idx = i

                #print(f"Column indices - x:{x_idx}, y:{y_idx}, z:{z_idx}, label:{label_idx}")

                # Validation
                if x_idx is None or y_idx is None or z_idx is None or label_idx is None:
                    raise ValueError("PCD file must have x, y, z, and label fields!")

            # Start reading data
            if line.startswith("DATA"):
                header_done = True
                continue

            if header_done:
                vals = line.split()
                if len(vals) < len(field_names):
                    continue

                total_points += 1

                # Extract values based on indices
                x = float(vals[x_idx])
                y = float(vals[y_idx])
                z = float(vals[z_idx])
                label = int(vals[label_idx])

                label_counts[label] = label_counts.get(label, 0) + 1

                if label == target_label:
                    points.append([x, y, z])

    #print(f"\nTotal points in PCD: {total_points}")
    #print(f"Label counts: {label_counts}")
    #print(f"Points with label={target_label}: {len(points)}")
    return np.array(points)