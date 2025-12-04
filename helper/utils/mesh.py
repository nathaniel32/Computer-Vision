import open3d as o3d
import numpy as np
from dataclasses import dataclass
import numpy as np
from typing import Tuple
import trimesh
import os

def get_cluster_labels(xyz, eps=0.02, min_points=10):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    cluster_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return cluster_labels

@dataclass
class AxisMetrics:
    length: float
    pc: np.ndarray
    min_proj: float
    max_proj: float
    p_min: np.ndarray
    p_max: np.ndarray
    
    def get_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.p_min, self.p_max

@dataclass
class PointsMetrics:
    pc1: AxisMetrics  # Diameter 1
    pc2: AxisMetrics  # Diameter 2
    pc3: AxisMetrics  # Thickness
    center: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    
    def get_circularity(self) -> float:
        d1, d2 = self.pc1.length, self.pc2.length
        return min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
    
    def is_valid_marker(self, circularity_threshold: float = 0.85, thickness_ratio_max: float = 0.3) -> bool:
        circ = self.get_circularity()
        thickness_ratio = self.pc3.length / self.pc1.length if self.pc1.length > 0 else float('inf')
        return circ >= circularity_threshold and thickness_ratio <= thickness_ratio_max
    
    def get_diameter(self) -> float:
        return (self.pc1.length + self.pc2.length) / 2.0
    
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

def plot_marker_all_axes(points, center, marker_axes_metrics, file_base_name="plot", save_dir=None, headless=False):
    import matplotlib.pyplot as plt
    """
    Plot 3D marker with all 3 principal PCA axes.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:,0], points[:,1], points[:,2], s=1, color='green', alpha=0.3, label='Marker points')

    ax.scatter(center[0], center[1], center[2], color='red', s=100, marker='o', label='Centroid')

    colors = ['blue', 'orange', 'purple']
    labels = ['PC1 (Diameter 1)', 'PC2 (Diameter 2)', 'PC3 (Thickness)']

    for name, color, label in zip(['PC1', 'PC2', 'PC3'], colors, labels):
        res = marker_axes_metrics[name]
        line = np.vstack([res['p_min'], res['p_max']])
        ax.plot(line[:,0], line[:,1], line[:,2], color=color, linewidth=3, label=f"{label}: {res['length']:.4f}")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Marker PCA - All Principal Axes')
    ax.legend()

    # Set equal aspect ratio
    max_range = np.array([
        points[:,0].max()-points[:,0].min(),
        points[:,1].max()-points[:,1].min(),
        points[:,2].max()-points[:,2].min()
    ]).max() / 2.0

    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_base_name}_marker.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if not headless:
        plt.show()

    plt.close()

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
                print(f"Detected fields: {field_names}")

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

                print(f"Column indices - x:{x_idx}, y:{y_idx}, z:{z_idx}, label:{label_idx}")

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

    print(f"\nTotal points in PCD: {total_points}")
    print(f"Label counts: {label_counts}")
    print(f"Points with label={target_label}: {len(points)}")
    return np.array(points)