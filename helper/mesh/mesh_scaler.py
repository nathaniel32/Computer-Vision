import numpy as np
from helper.utils.mesh import get_cluster_labels, AxisMetrics, PointsMetrics, filter_clusters, read_pcd_label, MarkerPair
from itertools import permutations

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
    circularity_threshold=0.85
    diameter_tolerance=0.15

    all_markers_metrics = []
    for label in marker_config.scale_labels:
        try:
            points_marker = read_pcd_label(pcd_file, target_label=label)
            marker_clusters = filter_clusters(points_marker)

            markers = []
            for marker_cluster in marker_clusters:
                marker_metrics = PointsMetrics(marker_cluster)

                #marker_metrics.plot_points_axes(points_marker)
                #marker_metrics.plot_points_axes(marker_cluster)

                circularity, avg_diameter, diameter_ratio, quality_score = marker_metrics.calculate_circularity()

                print(circularity, avg_diameter, diameter_ratio, quality_score)

                if circularity < circularity_threshold:
                    print(f"  - SKIPPED - Low circularity ({circularity:.3f} < {circularity_threshold})")
                    continue
                if diameter_ratio > diameter_tolerance:
                    print(f"  - SKIPPED - Diameter mismatch too large ({diameter_ratio*100:.1f}% > {diameter_tolerance*100:.1f}%)")
                    continue

                markers.append(marker_metrics)

            marker_pairs = []
            for marker1, marker2 in permutations(markers, 2):
                pair = MarkerPair(marker1, marker2)
                marker_pairs.append(pair)
            
            print(f"== Total pairs: {len(marker_pairs)}") # n*(n-1)
        except Exception as e:
            print(e)

    #scale_factor = calculate_scale_factor(all_markers_metrics, real_marker_diameter_cm)
    #print(f"Scale factor (cm/unit): {scale_factor:.4f}")

if __name__ == "__main__":
    main() # py -m helper.mesh.mesh_scaler