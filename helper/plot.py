import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from enum import Enum

def plot_training_stats(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None, headless=False):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    if not headless:
        plt.show()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

class PlotTool(str, Enum):
    MATPLOTLIB = "matplotlib"
    OPEN3D = "open3d"

def plot_point_cloud(point_cloud, color_plot, classes, pred_label=None, true_label=None, plot_tool=PlotTool.MATPLOTLIB, save_dir=None, file_base_name="plot", headless=False):
    if plot_tool == PlotTool.MATPLOTLIB:
        # Determine the number of subplots
        num_subplots = 1
        if true_label is not None:
            num_subplots += 1
        if pred_label is not None:
            num_subplots += 1
        
        fig = plt.figure(figsize=(7 * num_subplots, 6))
        subplot_idx = 1
        
        # Original Point Cloud with Color
        ax = fig.add_subplot(1, num_subplots, subplot_idx, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                    c=color_plot, s=10, alpha=0.6)
        ax.set_title('Original Point Cloud (RGB)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        subplot_idx += 1
        
        # True Labels
        if true_label is not None:
            df_true = pd.DataFrame({
                'x': point_cloud[:, 0], 
                'y': point_cloud[:, 1], 
                'z': point_cloud[:, 2], 
                'label': true_label
            })
            
            ax = fig.add_subplot(1, num_subplots, subplot_idx, projection='3d')
            for idx, _class in enumerate(classes):
                c_df = df_true[df_true['label'] == idx]
                if len(c_df) > 0:
                    ax.scatter(c_df['x'], c_df['y'], c_df['z'], c=[_class['color']] * len(c_df), 
                            label=_class['label'], alpha=0.5, s=10)
            ax.set_title('True Labels')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend(loc='upper right') #ax.legend()
            subplot_idx += 1
        
        # Predicted Labels
        if pred_label is not None:
            df_pred = pd.DataFrame({
                'x': point_cloud[:, 0], 
                'y': point_cloud[:, 1], 
                'z': point_cloud[:, 2], 
                'label': pred_label
            })
            
            ax = fig.add_subplot(1, num_subplots, subplot_idx, projection='3d')
            for idx, _class in enumerate(classes):
                c_df = df_pred[df_pred['label'] == idx]
                if len(c_df) > 0:
                    ax.scatter(c_df['x'], c_df['y'], c_df['z'], c=[_class['color']] * len(c_df), 
                            label=_class['label'], alpha=0.5, s=10)
            ax.set_title('Predicted Labels')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend(loc='upper right') # ax.legend()
        
        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{file_base_name}_plt.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if not headless:
            plt.show()
        plt.close()

    elif plot_tool == PlotTool.OPEN3D:
        import open3d as o3d
        
        def create_pointcloud_with_colors(point_cloud, label, classes):           
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            
            colors = np.zeros_like(point_cloud, dtype=np.float64)
            for idx, _class in enumerate(classes):
                mask = (label == idx)
                if mask.any():
                    color = hex_to_rgb(_class['color']) if isinstance(_class['color'], str) else _class['color']
                    colors[mask] = color
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            return pcd

        def create_visualizer(window_name, geometry, point_size=3, background_color=None):
            if background_color is None:
                background_color = np.array([0.5, 0.5, 0.5])
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name, width=600, height=600, visible=not headless)
            vis.add_geometry(geometry)
            vis.get_render_option().point_size = point_size
            vis.get_render_option().background_color = background_color
            return vis

        def capture_and_save(vis, save_path):
            """Capture screen from visualizer and save to file"""
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_path, do_render=True)
            print(f"Screenshot saved to {save_path}")

        def run_multiple_visualizers(visualizers, save_dir=None, file_base_name="plot", window_names=None):
            # Initial update
            for vis in visualizers:
                vis.poll_events()
                vis.update_renderer()
            
            # Save screenshots if save_dir is provided
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                for idx, vis in enumerate(visualizers):
                    window_suffix = window_names[idx] if window_names else f"view_{idx}"
                    save_path = os.path.join(save_dir, f"{file_base_name}_o3d_{window_suffix}.png")
                    capture_and_save(vis, save_path)
            
            if not headless:
                # Main loop
                while True:
                    all_active = True
                    for vis in visualizers:
                        if not vis.poll_events():
                            all_active = False
                        vis.update_renderer()
                    
                    if not all_active:
                        break
            
            # Cleanup
            for vis in visualizers:
                vis.destroy_window()

        visualizers = []
        window_names = []
        
        # Original Point Cloud with RGB colors
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_original.colors = o3d.utility.Vector3dVector(color_plot)
        vis_original = create_visualizer("Original Point Cloud (RGB)", pcd_original)
        visualizers.append(vis_original)
        window_names.append("original")
        
        # True Labels
        if true_label is not None:
            pcd_true = create_pointcloud_with_colors(point_cloud, true_label, classes)
            vis_true = create_visualizer("True Labels", pcd_true)
            visualizers.append(vis_true)
            window_names.append("true_labels")
        
        # Predicted Labels
        if pred_label is not None:
            pcd_pred = create_pointcloud_with_colors(point_cloud, pred_label, classes)
            vis_pred = create_visualizer("Predicted Labels", pcd_pred)
            visualizers.append(vis_pred)
            window_names.append("pred_labels")
        
        # Run all visualizers
        run_multiple_visualizers(visualizers, save_dir=save_dir, file_base_name=file_base_name, window_names=window_names)

    """ elif plot_tool == PlotTool.OPEN3D:
        def create_pointcloud_with_colors(point_cloud, label, classes):           
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            
            colors = np.zeros_like(point_cloud, dtype=np.float64)
            for idx, _class in enumerate(classes):
                mask = (label == idx)
                if mask.any():
                    color = hex_to_rgb(_class['color']) if isinstance(_class['color'], str) else _class['color']
                    colors[mask] = color
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            return pcd

        def create_visualizer(window_name, geometry, point_size=3, background_color=None):
            if background_color is None:
                background_color = np.array([0.5, 0.5, 0.5])
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name, width=600, height=600)
            vis.add_geometry(geometry)
            vis.get_render_option().point_size = point_size
            vis.get_render_option().background_color = background_color
            return vis

        def run_multiple_visualizers(visualizers):
            # Initial update
            for vis in visualizers:
                vis.poll_events()
                vis.update_renderer()
            
            # Main loop
            while True:
                all_active = True
                for vis in visualizers:
                    if not vis.poll_events():
                        all_active = False
                    vis.update_renderer()
                
                if not all_active:
                    break
            
            # Cleanup
            for vis in visualizers:
                vis.destroy_window()


        visualizers = []
        
        # Original Point Cloud with RGB colors
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_original.colors = o3d.utility.Vector3dVector(color_plot)
        vis_original = create_visualizer("Original Point Cloud (RGB)", pcd_original)
        visualizers.append(vis_original)
        
        # True Labels
        if true_label is not None:
            pcd_true = create_pointcloud_with_colors(point_cloud, true_label, classes)
            vis_true = create_visualizer("True Labels", pcd_true)
            visualizers.append(vis_true)
        
        # Predicted Labels
        if pred_label is not None:
            pcd_pred = create_pointcloud_with_colors(point_cloud, pred_label, classes)
            vis_pred = create_visualizer("Predicted Labels", pcd_pred)
            visualizers.append(vis_pred)
        
        # Run all visualizers
        run_multiple_visualizers(visualizers) """