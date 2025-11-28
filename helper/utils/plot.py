import matplotlib.pyplot as plt
import pandas as pd
import configs
import open3d as o3d
import numpy as np

def plot_training_stats(train_losses, val_losses, train_accuracies, val_accuracies):
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
    plt.show()

def plot_point_cloud(point_cloud, color_plot, pred_label=None, true_label=None, plot_tool="matplotlib"):
    if plot_tool == "matplotlib":
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
            for idx, _class in enumerate(configs.CLASSES):
                c_df = df_true[df_true['label'] == idx]
                if len(c_df) > 0:
                    ax.scatter(c_df['x'], c_df['y'], c_df['z'], c=[_class['color']] * len(c_df), 
                            label=_class['label'], alpha=0.5, s=10)
            ax.set_title('True Labels')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
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
            for idx, _class in enumerate(configs.CLASSES):
                c_df = df_pred[df_pred['label'] == idx]
                if len(c_df) > 0:
                    ax.scatter(c_df['x'], c_df['y'], c_df['z'], c=[_class['color']] * len(c_df), 
                            label=_class['label'], alpha=0.5, s=10)
            ax.set_title('Predicted Labels')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
        
        plt.tight_layout()
        plt.show()

    elif plot_tool == "open3d":
        def hex_to_rgb(hex_color):
            """Convert hex color string to RGB normalized (0-1)"""
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        
        visualizers = []
        
        # Original Point Cloud with Color
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_original.colors = o3d.utility.Vector3dVector(color_plot)
        
        vis_original = o3d.visualization.VisualizerWithVertexSelection()
        vis_original.create_window(window_name="Original Point Cloud (RGB)", width=600, height=600)
        vis_original.add_geometry(pcd_original)
        vis_original.get_render_option().point_size = 3
        vis_original.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
        visualizers.append(vis_original)
        
        # True Labels
        if true_label is not None:
            pcd_true = o3d.geometry.PointCloud()
            pcd_true.points = o3d.utility.Vector3dVector(point_cloud)
            
            true_colors = np.zeros_like(point_cloud, dtype=np.float64)
            for idx, _class in enumerate(configs.CLASSES):
                mask = (true_label == idx)
                if mask.any():
                    color = hex_to_rgb(_class['color']) if isinstance(_class['color'], str) else _class['color']
                    true_colors[mask] = color
            
            pcd_true.colors = o3d.utility.Vector3dVector(true_colors)
            
            vis_true = o3d.visualization.VisualizerWithVertexSelection()
            vis_true.create_window(window_name="True Labels", width=600, height=600)
            vis_true.add_geometry(pcd_true)
            vis_true.get_render_option().point_size = 3
            vis_true.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
            visualizers.append(vis_true)
        
        # Predicted Labels
        if pred_label is not None:
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(point_cloud)
            
            pred_colors = np.zeros_like(point_cloud, dtype=np.float64)
            for idx, _class in enumerate(configs.CLASSES):
                mask = (pred_label == idx)
                if mask.any():
                    color = hex_to_rgb(_class['color']) if isinstance(_class['color'], str) else _class['color']
                    pred_colors[mask] = color
            
            pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)
            
            vis_pred = o3d.visualization.VisualizerWithVertexSelection()
            vis_pred.create_window(window_name="Predicted Labels", width=600, height=600)
            vis_pred.add_geometry(pcd_pred)
            vis_pred.get_render_option().point_size = 3
            vis_pred.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
            visualizers.append(vis_pred)
        
        # Update semua visualizer
        for vis in visualizers:
            vis.poll_events()
            vis.update_renderer()
        
        # Tampilkan semua windows
        while True:
            all_active = True
            for vis in visualizers:
                if not vis.poll_events():
                    all_active = False
                vis.update_renderer()
            
            if not all_active:
                break
        
        for vis in visualizers:
            vis.destroy_window()