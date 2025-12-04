import open3d as o3d
import numpy as np
from dataclasses import dataclass, field
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
    
@dataclass
class PointsMetrics:
    points: np.ndarray
    center: np.ndarray = field(init=False)
    singular_values: np.ndarray = field(init=False)
    explained_variance_ratio: np.ndarray = field(init=False)
    pc1: AxisMetrics = field(init=False)
    pc2: AxisMetrics = field(init=False)
    pc3: AxisMetrics = field(init=False)

    def __post_init__(self):
        if self.points.shape[0] < 3:
            raise ValueError("Not enough points for PCA (minimum 3)")
        
        # compute PCA
        self.center = self.points.mean(axis=0)
        pts_centered = self.points - self.center

        # SVD
        U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)

        self.singular_values = S
        self.explained_variance_ratio = S**2 / np.sum(S**2)

        self.pc1 = self._create_axis_metrics(pts_centered, Vt[0], self.center)
        self.pc2 = self._create_axis_metrics(pts_centered, Vt[1], self.center)
        self.pc3 = self._create_axis_metrics(pts_centered, Vt[2], self.center)

    def _create_axis_metrics(self, pts_centered, pc, center):
        projections = pts_centered @ pc
        min_proj = projections.min()
        max_proj = projections.max()
        length = max_proj - min_proj
        
        return AxisMetrics(
            length=float(length),
            pc=pc,
            min_proj=float(min_proj),
            max_proj=float(max_proj),
            p_min=center + min_proj * pc,
            p_max=center + max_proj * pc
        )
    
    def calculate_circularity(self):
        pc1_length = self.pc1.length
        pc2_length = self.pc2.length
        avg_diameter = (pc1_length + pc2_length) / 2.0
        diameter_diff = abs(pc1_length - pc2_length)
        diameter_ratio = diameter_diff / avg_diameter if avg_diameter > 0 else 1.0
        circularity = 1.0 - diameter_ratio
        quality_score = circularity * (1.0 - diameter_ratio)
        return circularity, avg_diameter, diameter_ratio, quality_score
    
    def plot_points_axes(self, all_points, file_base_name="plot", save_dir=None, headless=False):
        import matplotlib.pyplot as plt
        import os
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot points dan center
        ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2], 
                s=1, color='green', alpha=0.3, label='Points')
        ax.scatter(self.center[0], self.center[1], self.center[2], 
                color='red', s=100, marker='o', label='Centroid')

        # Plot PC axes
        colors = ['blue', 'orange', 'purple']
        labels = ['PC1 (Diameter 1)', 'PC2 (Diameter 2)', 'PC3 (Thickness)']
        axes = [self.pc1, self.pc2, self.pc3]

        for axis_metrics, color, label in zip(axes, colors, labels):
            line = np.vstack([axis_metrics.p_min, axis_metrics.p_max])
            ax.plot(line[:,0], line[:,1], line[:,2], 
                    color=color, linewidth=3, 
                    label=f"{label}: {axis_metrics.length:.4f}")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('PCA - All Principal Axes')
        ax.legend()

        # Set equal aspect ratio
        max_range = np.array([
            all_points[:,0].max()-all_points[:,0].min(),
            all_points[:,1].max()-all_points[:,1].min(),
            all_points[:,2].max()-all_points[:,2].min()
        ]).max() / 2.0

        mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
        mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
        mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{file_base_name}_axes.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        if not headless:
            plt.show()

        plt.close()

@dataclass
class MarkerPair:
    marker1: PointsMetrics
    marker2: PointsMetrics


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