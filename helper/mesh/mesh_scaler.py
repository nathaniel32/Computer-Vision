import numpy as np
import trimesh
import os
from helper.utils.mesh import get_cluster_labels

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

def measure_points_axes(points):
    """
    Compute marker size along all 3 PCA principal axes.
    For a circular marker, PC1 and PC2 should be similar (diameter).
    """
    if points.shape[0] < 3:
        raise ValueError("Not enough points for PCA (minimum 3)")

    print("\n=== PCA Analysis ===")
    print("Sample first 5 points:\n", points[:5])
    center = points.mean(axis=0)
    print("Center of points:", center)

    pts_centered = points - center
    print("Sample first 5 centered points:\n", pts_centered[:5])

    # SVD for PCA
    U, S, Vt = np.linalg.svd(pts_centered, full_matrices=False)

    # Principal components
    pc1 = Vt[0]
    pc2 = Vt[1]
    pc3 = Vt[2]

    print("\nPrincipal Components:")
    print(f"PC1 (largest variance): {pc1}")
    print(f"PC2 (second variance):  {pc2}")
    print(f"PC3 (smallest variance): {pc3}")

    # Singular values (square roots of eigenvalues)
    print(f"\nSingular values: {S}")
    print(f"Explained variance ratio: {S**2 / np.sum(S**2)}")

    marker_axes_metrics = {}

    for i, (pc, name) in enumerate([(pc1, 'PC1'), (pc2, 'PC2'), (pc3, 'PC3')]):
        projections = pts_centered @ pc
        min_proj = projections.min()
        max_proj = projections.max()
        length = max_proj - min_proj

        p_min_3d = center + min_proj * pc
        p_max_3d = center + max_proj * pc

        marker_axes_metrics[name] = {
            'length': float(length),
            'pc': pc,
            'min_proj': min_proj,
            'max_proj': max_proj,
            'p_min': p_min_3d,
            'p_max': p_max_3d
        }

        print(f"\n{name} ({['Diameter 1', 'Diameter 2', 'Thickness'][i]}):")
        print(f"  Projection range: [{min_proj:.4f}, {max_proj:.4f}]")
        print(f"  Length: {length:.4f}")
        print(f"  Min point: {p_min_3d}")
        print(f"  Max point: {p_max_3d}")

    return marker_axes_metrics, center


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

def calculate_scale_factor(all_markers_metrics, real_diameter_cm, circularity_threshold=0.85, diameter_tolerance=0.15):
    """
    Compute scale factor based on real marker diameter.
    Only uses circular markers (high circularity).
    Compares all markers and returns most reliable scale factor.
    """
    valid_markers = []
    
    print("\n=== Marker Quality Check ===")
    
    for idx, markers_metrics in enumerate(all_markers_metrics):
        length_pc1 = markers_metrics['PC1']['length']
        length_pc2 = markers_metrics['PC2']['length']
        
        # Calculate circularity (PC1 vs PC2 should be similar for circles)
        avg_diameter = (length_pc1 + length_pc2) / 2.0
        diameter_diff = abs(length_pc1 - length_pc2)
        diameter_ratio = diameter_diff / avg_diameter if avg_diameter > 0 else 1.0
        circularity = 1.0 - diameter_ratio  # 1.0 = perfect circle, lower = more elongated
        
        print(f"\nMarker {idx}:")
        print(f"  PC1 length: {length_pc1:.4f}")
        print(f"  PC2 length: {length_pc2:.4f}")
        print(f"  Average diameter: {avg_diameter:.4f}")
        print(f"  Diameter difference: {diameter_diff:.4f} ({diameter_ratio*100:.1f}%)")
        print(f"  Circularity: {circularity:.3f}")
        
        # Check if marker is circular enough
        if circularity < circularity_threshold:
            print(f"  - SKIPPED - Low circularity ({circularity:.3f} < {circularity_threshold})")
            continue
        
        # Check if PC1 and PC2 are within tolerance
        if diameter_ratio > diameter_tolerance:
            print(f"  - SKIPPED - Diameter mismatch too large ({diameter_ratio*100:.1f}% > {diameter_tolerance*100:.1f}%)")
            continue
        
        # Calculate scale factor for this marker
        scale_factor = real_diameter_cm / avg_diameter
        
        valid_markers.append({
            'index': idx,
            'pc1': length_pc1,
            'pc2': length_pc2,
            'avg_diameter': avg_diameter,
            'circularity': circularity,
            'diameter_ratio': diameter_ratio,
            'scale_factor': scale_factor,
            'quality_score': circularity * (1.0 - diameter_ratio)  # Combined quality metric
        })
        
        print(f"  - VALID - Scale factor: {scale_factor:.4f}")
    
    # If no valid markers
    if len(valid_markers) == 0:
        raise ValueError("- ERROR: No valid circular markers found!")
    
    print("\n=== Scale Factor Selection ===")
    print(f"Valid markers: {len(valid_markers)}/{len(all_markers_metrics)}")
    
    # Calculate statistics
    scale_factors = [m['scale_factor'] for m in valid_markers]
    mean_scale = np.mean(scale_factors)
    std_scale = np.std(scale_factors)
    
    print(f"Mean scale factor: {mean_scale:.4f}")
    print(f"Std deviation: {std_scale:.4f}")
    
    # Remove outliers (> 2 std deviations from mean)
    filtered_markers = []
    for marker in valid_markers:
        deviation = abs(marker['scale_factor'] - mean_scale)
        if deviation < 2 * std_scale:
            filtered_markers.append(marker)
            print(f"  Marker {marker['index']}: scale={marker['scale_factor']:.4f}, quality={marker['quality_score']:.3f} ✓")
        else:
            print(f"  Marker {marker['index']}: scale={marker['scale_factor']:.4f} - OUTLIER (deviation={deviation:.4f})")
    
    if len(filtered_markers) == 0:
        print("- WARNING: All markers are outliers, using best quality marker anyway")
        filtered_markers = valid_markers
    
    # Sort by quality score (circularity * consistency)
    filtered_markers.sort(key=lambda x: x['quality_score'], reverse=True)
    
    best_marker = filtered_markers[0]
    
    print("\n=== Final Result ===")
    print(f"Selected marker: {best_marker['index']}")
    print(f"  Circularity: {best_marker['circularity']:.3f}")
    print(f"  PC1: {best_marker['pc1']:.4f}, PC2: {best_marker['pc2']:.4f}")
    print(f"  Avg diameter: {best_marker['avg_diameter']:.4f}")
    print(f"  Quality score: {best_marker['quality_score']:.3f}")
    print(f"  Scale factor: {best_marker['scale_factor']:.4f}")
    
    # If multiple good markers, use weighted average
    if len(filtered_markers) >= 3:
        weights = [m['quality_score'] for m in filtered_markers]
        weighted_scale = np.average(
            [m['scale_factor'] for m in filtered_markers],
            weights=weights
        )
        print(f"- Weighted average from {len(filtered_markers)} markers: {weighted_scale:.4f}")
        return weighted_scale
    
    return best_marker['scale_factor']

def calculate_circularity(pc1_length, pc2_length):
    avg_diameter = (pc1_length + pc2_length) / 2.0
    diameter_diff = abs(pc1_length - pc2_length)
    diameter_ratio = diameter_diff / avg_diameter if avg_diameter > 0 else 1.0
    circularity = 1.0 - diameter_ratio
    quality_score = circularity * (1.0 - diameter_ratio)
    return circularity, avg_diameter, diameter_ratio, quality_score

if __name__ == "__main__":
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
    
    def main():
        from configs import marker_config

        # TODO
        # per label
        # - filter cluster circle
        # - peer setiap cluster circle (tanpa peer dengan diri), ukuran circle harus mirip atau abaikan
        # - hitung circle harus 1/4 dari panjang peer -> kasi nilai
        # - rangking nilai
        # per object
        # - rangking label dengan nilai tertinggi

        pcd_file = r"C:\Users\natha\Desktop\test_preds\obj_marker\bone\1\out\point_cloud.pcd" #input("pcd path: ").strip().strip('"').strip("'")
        real_marker_diameter_cm = 1 #float(input("real marker size: "))

        all_markers_metrics = []
        for label in marker_config.scale_labels:
            try:
                points_marker = read_pcd_label(pcd_file, target_label=label)
                marker_clusters = filter_clusters(points_marker)

                for marker_cluster in marker_clusters:
                    marker_axes_metrics, center = measure_points_axes(marker_cluster)
                    circularity, avg_diameter, diameter_ratio, quality_score = calculate_circularity(marker_axes_metrics['PC1']['length'], marker_axes_metrics['PC2']['length'])

                    print(circularity, avg_diameter, diameter_ratio, quality_score)

                    all_markers_metrics.append(marker_axes_metrics)
                    
                    plot_marker_all_axes(points_marker, center, marker_axes_metrics)
                    plot_marker_all_axes(marker_cluster, center, marker_axes_metrics)
            except Exception as e:
                print(e)

        scale_factor = calculate_scale_factor(all_markers_metrics, real_marker_diameter_cm)
        print(f"Scale factor (cm/unit): {scale_factor:.4f}")
    
    main() # py -m helper.mesh.mesh_scaler