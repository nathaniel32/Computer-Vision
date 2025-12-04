import numpy as np
from helper.mesh.mesh_utils import get_cluster_labels, AxisMetrics, PointsMetrics, filter_clusters, read_pcd_label, MarkerPair
from itertools import permutations



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