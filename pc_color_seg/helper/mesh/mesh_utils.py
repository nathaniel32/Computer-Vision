import open3d as o3d
import numpy as np
import os
from scipy.spatial import cKDTree

def get_eps(xyz, k=10):
    from sklearn.neighbors import NearestNeighbors
    # K-distance -> eps optimal
    nbrs = NearestNeighbors(n_neighbors=k).fit(xyz)
    distances, _ = nbrs.kneighbors(xyz)
    k_distances = np.sort(distances[:, k-1])[::-1]
    
    # Elbow detection
    eps = k_distances[np.argmax(np.diff(np.diff(k_distances)) + 1)]
    return eps

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
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    scale_factor_percent = int(scale_factor * 100)
    output_path = os.path.join(out_dir_path, f"scaled_{scale_factor_percent}_percent.obj")
    
    with open(output_path, 'w') as f:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                f.write(f'v {x * scale_factor} {y * scale_factor} {z * scale_factor}\n')
            else:
                f.write(line)