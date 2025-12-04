import numpy as np
from helper.mesh.mesh_utils import filter_clusters, read_pcd_label
from itertools import permutations
from dataclasses import dataclass, field

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
    
    def plot_points_axes(self, full_points=None, file_base_name="plot", save_dir=None, headless=False):
        import matplotlib.pyplot as plt
        import os

        if full_points is None:
            full_points = self.points
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot points dan center
        ax.scatter(full_points[:,0], full_points[:,1], full_points[:,2], 
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

    def get_diameter_similarity(self) -> float:
        """ 1.0 = identical, 0.0 = very different """
        d1 = self.marker1.get_avg_diameter()
        d2 = self.marker2.get_avg_diameter()
        
        if d1 == 0 or d2 == 0:
            return 0.0
        
        # Smaller / larger diameter ratio
        similarity = min(d1, d2) / max(d1, d2)
        return similarity
    
    def get_merged_points(self) -> np.ndarray:
        return np.vstack([self.marker1.points, self.marker2.points])
    
    def _get_center_distance(self) -> float:
        return np.linalg.norm(self.marker2.center - self.marker1.center)

    def plot_marker_pair(self, file_base_name="marker_pair", save_dir=None, headless=False):
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

        # Plot PC1 axes for marker 1 (diameter)
        line_m1_pc1 = np.vstack([self.marker1.pc1.p_min, self.marker1.pc1.p_max])
        ax.plot(line_m1_pc1[:,0], line_m1_pc1[:,1], line_m1_pc1[:,2], 
                color='cyan', linewidth=2.5, label=f'M1 Diameter: {self.marker1.pc1.length:.4f}')

        # Plot PC1 axes for marker 2 (diameter)
        line_m2_pc1 = np.vstack([self.marker2.pc1.p_min, self.marker2.pc1.p_max])
        ax.plot(line_m2_pc1[:,0], line_m2_pc1[:,1], line_m2_pc1[:,2], 
                color='orange', linewidth=2.5, label=f'M2 Diameter: {self.marker2.pc1.length:.4f}')

        # Add metrics to title
        similarity = self.get_diameter_similarity()
        ax.set_title(f'Marker Pair - Diameter Similarity: {similarity:.3f}', fontsize=14)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(loc='upper right', fontsize=9)

        # Set equal aspect ratio based on both markers
        merged_points = self.get_merged_points()
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

def main():
    from configs import marker_config

    # TODO
    # per label
    # - filter cluster circle                                                                                   OK
    # - peer setiap cluster circle (tanpa peer dengan diri), ukuran circle harus mirip atau abaikan             OK
    # - hitung peer marker caranya marker1 kira kira 1/4 dari length marker pair -> kasi nilai quality
    #   - NOTE: (Marker 1cm)  -- distance 2cm -- (Marker 1cm)   total 4cm
    # - append rangking nilai quality tertinggi
    
    # per object
    # - rangking label dengan nilai tertinggi
    # - hitung scale factor

    pcd_file = r"C:\Users\natha\Desktop\test_preds\obj_marker\bone\1\out\point_cloud.pcd" #input("pcd path: ").strip().strip('"').strip("'")
    marker_diameter_cm = 1 #float(input("real marker size: "))
    pair_marker_length = 4
    circularity_threshold = 0.85
    diameter_tolerance = 0.15
    pair_similarity_threshold = 0.85

    for label in marker_config.scale_labels:
        try:
            points_marker = read_pcd_label(pcd_file, target_label=label)
            marker_clusters = filter_clusters(points_marker)

            markers = []
            for marker_cluster in marker_clusters:
                marker_metrics = PointsMetrics(marker_cluster)

                # plot
                #marker_metrics.plot_points_axes(points_marker)
                #marker_metrics.plot_points_axes()

                circularity, avg_diameter, diameter_ratio, quality_score = marker_metrics.calculate_circularity()

                if circularity < circularity_threshold:
                    print(f"- SKIPPED - Low circularity ({circularity:.3f} < {circularity_threshold})")
                    continue
                if diameter_ratio > diameter_tolerance:
                    print(f"- SKIPPED - Diameter mismatch too large ({diameter_ratio*100:.1f}% > {diameter_tolerance*100:.1f}%)")
                    continue
                
                marker_metrics.plot_points_axes()

                markers.append(marker_metrics)

            marker_pairs = []
            for marker1, marker2 in permutations(markers, 2):
                pair = MarkerPair(marker1, marker2)
                pair_similarity = pair.get_diameter_similarity()
                print(pair_similarity)
                pair.plot_marker_pair()
                PointsMetrics(pair.get_merged_points()).plot_points_axes()
                if pair_similarity < pair_similarity_threshold:
                    print(f"- SKIPPED - Low Similarity")
                    continue

                marker_pairs.append(pair)
            
            print(f"== Total pairs: {len(marker_pairs)}") # n*(n-1)
        except Exception as e:
            print("Error: ", e)

    #scale_factor = ?
    #print(f"Scale factor (cm/unit): {scale_factor:.4f}")

if __name__ == "__main__":
    main() # py -m helper.mesh.mesh_scaler