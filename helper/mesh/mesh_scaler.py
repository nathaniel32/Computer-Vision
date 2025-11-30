import numpy as np
import matplotlib.pyplot as plt
import trimesh
import os

def scale_mesh(scale_factor, input_path, out_dir_path):
    mesh = trimesh.load(input_path)
    mesh.apply_scale(scale_factor)
    mesh.export(os.path.join(out_dir_path, "scaled.obj"))

def measure_marker_all_axes(points):
    """
    Compute marker size along all 3 PCA principal axes.
    For a circular marker, PC1 and PC2 should be similar (diameter).
    """
    if points.shape[0] < 3:
        print("Not enough points for PCA (minimum 3)")
        return None

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


def plot_marker_all_axes(points, center, marker_axes_metrics):
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
    plt.show()


def calculate_scale_factor(all_markers_metrics, real_diameter_cm):
    """
    Compute scale factor based on real marker diameter.
    Uses average of PC1 & PC2 (diameters).
    """
    for markers_metrics in all_markers_metrics:
        length_pc1 = markers_metrics['PC1']['length']
        length_pc2 = markers_metrics['PC2']['length']

        avg_diameter = (length_pc1 + length_pc2) / 2.0

        print("\n=== Scale Factor Calculation ===")
        print(f"PC1 length (Diameter 1): {length_pc1:.4f}")
        print(f"PC2 length (Diameter 2): {length_pc2:.4f}")
        print(f"Average diameter: {avg_diameter:.4f}")
        print(f"Real marker diameter: {real_diameter_cm} cm")

        scale_factor = real_diameter_cm / avg_diameter
        print(f"Scale factor: {scale_factor:.4f}")

        return scale_factor

if __name__ == "__main__":
    def read_pcd_label(filename, target_label=1):
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
        pcd_file = input("pcd path: ").strip().strip('"').strip("'")
        real_marker_diameter_cm = 3.85

        for i in range(1, 5):
            points_marker = read_pcd_label(pcd_file, target_label=i)

            if len(points_marker) == 0:
                print(f"No marker points with label = {i}")
            else:
                results, center = measure_marker_all_axes(points_marker)

                if results:
                    scale_factor, avg_diameter = calculate_scale_factor(results, real_marker_diameter_cm)

                    print("\n=== Summary ===")
                    print(f"Marker measurements in model units:")
                    print(f"  Diameter 1 (PC1): {results['PC1']['length']:.4f}")
                    print(f"  Diameter 2 (PC2): {results['PC2']['length']:.4f}")
                    print(f"  Thickness (PC3):  {results['PC3']['length']:.4f}")
                    print(f"  Average diameter: {avg_diameter:.4f}")
                    print(f"\nScale factor (cm/unit): {scale_factor:.4f}")
                    print(f"\nScaled measurements in cm:")
                    print(f"  Diameter 1: {results['PC1']['length'] * scale_factor:.4f} cm")
                    print(f"  Diameter 2: {results['PC2']['length'] * scale_factor:.4f} cm")
                    print(f"  Thickness:  {results['PC3']['length'] * scale_factor:.4f} cm")

                    plot_marker_all_axes(points_marker, center, results)