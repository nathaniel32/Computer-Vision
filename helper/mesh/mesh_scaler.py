import numpy as np
from helper.mesh.mesh_utils import filter_clusters
from helper.train.dataset import load_pcd_with_point_labels, transform_color
from itertools import permutations, combinations
from dataclasses import dataclass, field
import configs
from typing import Optional

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
    
    def get_avg_diameter(self) -> float:
        """Get average diameter dari PC1 dan PC2."""
        return (self.pc1.length + self.pc2.length) / 2.0

    def calculate_circularity(self):
        pc1_length = self.pc1.length
        pc2_length = self.pc2.length
        avg_diameter = self.get_avg_diameter()
        diameter_diff = abs(pc1_length - pc2_length)
        diameter_ratio = diameter_diff / avg_diameter if avg_diameter > 0 else 1.0
        circularity = 1.0 - diameter_ratio
        quality_score = circularity * (1.0 - diameter_ratio)
        return circularity, avg_diameter, diameter_ratio, quality_score
    
    def plot_points_axes(self, full_points=None, full_points_color='green', title="Principal Axes", file_base_name="plot", save_dir=None, headless=False):
        import matplotlib.pyplot as plt
        import os

        if full_points is None:
            full_points = self.points
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot points dan center
        ax.scatter(full_points[:,0], full_points[:,1], full_points[:,2], 
                s=1, color=full_points_color, alpha=0.3, label='Points')
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
        ax.set_title(title)
        ax.legend()

        # Set equal aspect ratio
        max_range = np.array([
            full_points[:,0].max()-full_points[:,0].min(),
            full_points[:,1].max()-full_points[:,1].min(),
            full_points[:,2].max()-full_points[:,2].min()
        ]).max() / 2.0

        mid_x = (full_points[:,0].max()+full_points[:,0].min()) * 0.5
        mid_y = (full_points[:,1].max()+full_points[:,1].min()) * 0.5
        mid_z = (full_points[:,2].max()+full_points[:,2].min()) * 0.5

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
    label_class: str
    merged_marker: PointsMetrics = field(init=False)

    def __post_init__(self):
        merged_points = self._get_merged_points()
        self.merged_marker = PointsMetrics(merged_points)

    def get_diameter_similarity(self) -> float:
        """ 1.0 = identical, 0.0 = very different """
        d1 = self.marker1.get_avg_diameter()
        d2 = self.marker2.get_avg_diameter()
        
        if d1 == 0 or d2 == 0:
            return 0.0
        
        # Smaller / larger diameter ratio
        similarity = min(d1, d2) / max(d1, d2)
        return similarity
    
    def _get_merged_points(self) -> np.ndarray:
        return np.vstack([self.marker1.points, self.marker2.points])
    
    def _get_merged_length(self) -> float:
        return self.merged_marker.pc1.length
    
    def _get_center_distance(self) -> float:
        return np.linalg.norm(self.marker2.center - self.marker1.center)
    
    def _get_predicted_merged_length(self) -> float:
        center_dist = self._get_center_distance()
        radius1 = self.marker1.get_avg_diameter() / 2.0
        radius2 = self.marker2.get_avg_diameter() / 2.0
        return center_dist + radius1 + radius2
    
    def _get_merged_length_accuracy(self) -> float:
        actual = self._get_merged_length()
        predicted = self._get_predicted_merged_length()
        
        if actual == 0 or predicted == 0:
            return 0.0
        
        return min(actual, predicted) / max(actual, predicted)
    
    def _get_distance_ratio_accuracy(self, expected_ratio: float) -> float:
        center_dist = self._get_center_distance()
        merged_len = self._get_merged_length()
        
        if center_dist == 0:
            return 0.0
        
        actual_ratio = merged_len / center_dist
        
        # Similarity antara actual ratio vs expected ratio
        similarity = min(actual_ratio, expected_ratio) / max(actual_ratio, expected_ratio)
        
        return similarity
    
    def get_prediction_accuracy(self, expected_ratio: float) -> float:
        merged_acc = self._get_merged_length_accuracy()
        distance_acc = self._get_distance_ratio_accuracy(expected_ratio)
        
        combined = (merged_acc * 0.6 + distance_acc * 0.4)
        
        return combined
    
    def get_scale_factor(self, real_center_distance) -> float:
        center_dist = self._get_center_distance()
        scale_factor = real_center_distance / center_dist
        return scale_factor

    def plot_marker_pair(self, title="Marker Pair", file_base_name="marker_pair", save_dir=None, headless=False):
        """Plot both markers with their PCA axes and connection line."""
        import matplotlib.pyplot as plt
        import os
        
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot marker 1 points
        ax.scatter(self.marker1.points[:,0], self.marker1.points[:,1], self.marker1.points[:,2], 
                   s=2, color='blue', alpha=0.4, label='Marker 1 Points')
        ax.scatter(self.marker1.center[0], self.marker1.center[1], self.marker1.center[2], 
                   color='darkblue', s=150, marker='o', label='Marker 1 Center')

        # Plot marker 2 points
        ax.scatter(self.marker2.points[:,0], self.marker2.points[:,1], self.marker2.points[:,2], 
                   s=2, color='red', alpha=0.4, label='Marker 2 Points')
        ax.scatter(self.marker2.center[0], self.marker2.center[1], self.marker2.center[2], 
                   color='darkred', s=150, marker='o', label='Marker 2 Center')

        # Plot connection line between centers
        center_line = np.vstack([self.marker1.center, self.marker2.center])
        ax.plot(center_line[:,0], center_line[:,1], center_line[:,2], 
                'k--', linewidth=2, alpha=0.6, 
                label=f'Distance: {self._get_center_distance():.4f}')
        
        # Plot Marker 1
        line_m1_pc1 = np.vstack([self.marker1.pc1.p_min, self.marker1.pc1.p_max])
        ax.plot(line_m1_pc1[:,0], line_m1_pc1[:,1], line_m1_pc1[:,2], 
                color='cyan', linewidth=2.5, label=f'M1 Diameter 1: {self.marker1.pc1.length:.4f}')

        line_m1_pc2 = np.vstack([self.marker1.pc2.p_min, self.marker1.pc2.p_max])
        ax.plot(line_m1_pc2[:,0], line_m1_pc2[:,1], line_m1_pc2[:,2], 
                color='cyan', linewidth=2.5, linestyle='--', alpha=0.6, 
                label=f'M1 Diameter 2: {self.marker1.pc2.length:.4f}')

        # Plot Marker 2
        line_m2_pc1 = np.vstack([self.marker2.pc1.p_min, self.marker2.pc1.p_max])
        ax.plot(line_m2_pc1[:,0], line_m2_pc1[:,1], line_m2_pc1[:,2], 
                color='orange', linewidth=2.5, label=f'M2 Diameter 1: {self.marker2.pc1.length:.4f}')

        line_m2_pc2 = np.vstack([self.marker2.pc2.p_min, self.marker2.pc2.p_max])
        ax.plot(line_m2_pc2[:,0], line_m2_pc2[:,1], line_m2_pc2[:,2], 
                color='orange', linewidth=2.5, linestyle='--', alpha=0.6,
                label=f'M2 Diameter 2: {self.marker2.pc2.length:.4f}')
        
        # Plot merged
        line_merged = np.vstack([self.merged_marker.pc1.p_min, self.merged_marker.pc1.p_max])
        ax.plot(line_merged[:,0], line_merged[:,1], line_merged[:,2], 
                color='green', linewidth=1, linestyle=':', alpha=0.6,
                label=f'Pair Length: {self.merged_marker.pc1.length:.4f}')

        # Add metrics to title
        ax.set_title(title, fontsize=14)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper right', fontsize=9)

        # Set equal aspect ratio based on both markers
        merged_points = self._get_merged_points()
        max_range = np.array([
            merged_points[:,0].max() - merged_points[:,0].min(),
            merged_points[:,1].max() - merged_points[:,1].min(),
            merged_points[:,2].max() - merged_points[:,2].min()
        ]).max() / 2.0

        mid_x = (merged_points[:,0].max() + merged_points[:,0].min()) * 0.5
        mid_y = (merged_points[:,1].max() + merged_points[:,1].min()) * 0.5
        mid_z = (merged_points[:,2].max() + merged_points[:,2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{file_base_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Marker pair plot saved to {save_path}")

        if not headless:
            plt.show()

        plt.close()

class MeshScaler:
    def __init__(self, config:configs.BaseConfig):
        self.config = config

    def calculate_scale_factor(self, points, colors, labels, real_marker_pair_length, real_marker_pair_center_distance, circularity_threshold=0.75, diameter_tolerance=0.25, pair_similarity_threshold=0.85, pair_prediction_threshold=0.98, quality_check=True, plot_points_max=5000, plot=True, plot_dir=None, headless=False):
        best_marker_pairs: Optional[MarkerPair] = None
        distance_expected_ratio = real_marker_pair_length/real_marker_pair_center_distance

        for scale_label in self.config.scale_labels:
            label_class = self.config.classes[scale_label]['label']
            print(f"\n== Marker {label_class} ==")
            try:
                marker_indices = labels == scale_label
                marker_points = points[marker_indices]
                
                if len(marker_points) == 0:
                    print(f"No points found for label {label_class}")
                    continue
                    
                marker_clusters = filter_clusters(marker_points)

                markers = []
                for marker_cluster in marker_clusters:
                    marker_metrics = PointsMetrics(marker_cluster)

                    circularity, avg_diameter, diameter_ratio, quality_score = marker_metrics.calculate_circularity()
                    print(f"- Circularity: {circularity:.3f} - Diameter Ratio: {diameter_ratio*100:.1f}% - Avg Diameter: {avg_diameter}% - Quality Score: {quality_score}")

                    if circularity < circularity_threshold and quality_check:
                        print(f"  SKIPPED - Low circularity ({circularity:.3f} < {circularity_threshold})")
                        continue
                    if diameter_ratio > diameter_tolerance and quality_check:
                        print(f"  SKIPPED - Diameter mismatch ({diameter_ratio*100:.1f}% > {diameter_tolerance*100:.1f}%)")
                        continue
                    
                    markers.append(marker_metrics)

                if len(markers) < 2:
                    print(f"Marker {label_class}: Need at least 2 markers, found {len(markers)}")
                    continue

                for marker1, marker2 in combinations(markers, 2):
                    pair = MarkerPair(marker1, marker2, label_class)

                    pair_similarity = pair.get_diameter_similarity()
                    prediction_acc = pair.get_prediction_accuracy(distance_expected_ratio)
                    print(f"- Pair accuracy: {prediction_acc:.3f}, similarity: {pair_similarity:.3f}")
                    
                    if pair_similarity < pair_similarity_threshold and quality_check:
                        print(f"  SKIPPED - Low similarity ({pair_similarity:.3f} < {pair_similarity_threshold})")
                        continue

                    if prediction_acc < pair_prediction_threshold and quality_check:
                        print(f"  SKIPPED - Low prediction accuracy ({prediction_acc:.3f} < {pair_prediction_threshold})")
                        continue

                    if plot:
                        full_points = np.concatenate((points[:plot_points_max], points[marker_indices][:plot_points_max]))
                        full_points_color = np.concatenate((colors[:plot_points_max], colors[marker_indices][:plot_points_max]))
                        
                        scale_factor = pair.get_scale_factor(real_marker_pair_center_distance)
                        prediction_acc_percent = int(prediction_acc * 100)
                        pair.merged_marker.plot_points_axes(full_points=full_points, full_points_color=full_points_color, title=f"Marker: {label_class} - Prediction Accuracy: {prediction_acc:.3f} - Scale Factor: {scale_factor:.3f}", file_base_name=f"marker_{label_class}_{prediction_acc_percent}", save_dir=plot_dir, headless=headless)

                    if best_marker_pairs is None or best_marker_pairs.get_prediction_accuracy(distance_expected_ratio) < prediction_acc:
                        best_marker_pairs = pair
                        
            except Exception as e:
                print(f"Error processing label {label_class}: {e}")

        if best_marker_pairs is None:
            raise ValueError("No valid marker pairs found!")
        
        scale_factor = best_marker_pairs.get_scale_factor(real_marker_pair_center_distance)
        prediction_accuracy = best_marker_pairs.get_prediction_accuracy(distance_expected_ratio)

        print(f"\nBest pair found:")
        print(f"- Real center distance: {real_marker_pair_center_distance:.4f}")
        print(f"- Predicted Marker Pair length: {scale_factor*best_marker_pairs._get_merged_length():.4f}")
        print(f"- Scale factor: {scale_factor:.6f}")
        
        if plot:
            best_marker_pairs.plot_marker_pair(title=f'Marker: {best_marker_pairs.label_class} - Prediction Accuracy: {prediction_accuracy:.3f} - Scale Factor: {scale_factor:.3f}', file_base_name=f"best_marker", save_dir=plot_dir, headless=headless)

        return scale_factor, best_marker_pairs

if __name__ == "__main__":
    def main():
        pcd_file = input("pcd dir path: ").strip().strip('"').strip("'")
        real_marker_pair_length = 4
        real_marker_pair_center_distance = 3
        points, colors_int, labels = load_pcd_with_point_labels(pcd_file)

        MeshScaler(configs.marker_config).calculate_scale_factor(points[0], transform_color(colors_int[0]), labels[0], real_marker_pair_length, real_marker_pair_center_distance)

    main() # py -m helper.mesh.mesh_scaler