import matplotlib.pyplot as plt
import pandas as pd
import config
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

def plot_test_prediction(point_cloud, color_plot, true_labels, pred_labels, plot_tool="matplotlib"):
    if plot_tool == "matplotlib":
        df_true = pd.DataFrame({
            'x': point_cloud[:, 0], 
            'y': point_cloud[:, 1], 
            'z': point_cloud[:, 2], 
            'label': true_labels
        })
        
        df_pred = pd.DataFrame({
            'x': point_cloud[:, 0], 
            'y': point_cloud[:, 1], 
            'z': point_cloud[:, 2], 
            'label': pred_labels
        })
        
        fig = plt.figure(figsize=(20, 6))
        
        # Original Point Cloud dengan Color (paling kiri)
        ax0 = fig.add_subplot(131, projection='3d')
        # color_plot should be in shape (N, 3) with values 0-1
        ax0.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                    c=color_plot, s=10, alpha=0.6)
        ax0.set_title('Original Point Cloud (RGB)')
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        
        # True Labels
        ax1 = fig.add_subplot(132, projection='3d')
        for idx, _class in enumerate(config.CLASSES):
            c_df = df_true[df_true['label'] == idx]
            if len(c_df) > 0:
                ax1.scatter(c_df['x'], c_df['y'], c_df['z'], c=[_class['color']] * len(c_df), 
                        label=_class['label'], alpha=0.5, s=10)
        ax1.set_title('True Labels')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # Predicted Labels
        ax2 = fig.add_subplot(133, projection='3d')
        for idx, _class in enumerate(config.CLASSES):
            c_df = df_pred[df_pred['label'] == idx]
            if len(c_df) > 0:
                ax2.scatter(c_df['x'], c_df['y'], c_df['z'], c=[_class['color']] * len(c_df), 
                        label=_class['label'], alpha=0.5, s=10)
        ax2.set_title('Predicted Labels')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    elif plot_tool == "open3d":
        def hex_to_rgb(hex_color):
            """Convert hex color string to RGB normalized (0-1)"""
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        
        # Original Point Cloud dengan Color
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_original.colors = o3d.utility.Vector3dVector(color_plot)
        
        # True Labels
        pcd_true = o3d.geometry.PointCloud()
        pcd_true.points = o3d.utility.Vector3dVector(point_cloud)
        
        true_colors = np.zeros_like(point_cloud, dtype=np.float64)
        for idx, _class in enumerate(config.CLASSES):
            mask = (true_labels == idx)
            if mask.any():
                color = hex_to_rgb(_class['color']) if isinstance(_class['color'], str) else _class['color']
                true_colors[mask] = color
        
        pcd_true.colors = o3d.utility.Vector3dVector(true_colors)
        
        # Predicted Labels
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(point_cloud)
        
        pred_colors = np.zeros_like(point_cloud, dtype=np.float64)
        for idx, _class in enumerate(config.CLASSES):
            mask = (pred_labels == idx)
            if mask.any():
                color = hex_to_rgb(_class['color']) if isinstance(_class['color'], str) else _class['color']
                pred_colors[mask] = color
        
        pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)
        
        # Visualize dengan 3 windows
        vis_original = o3d.visualization.VisualizerWithVertexSelection()
        vis_original.create_window(window_name="Original Point Cloud (RGB)", width=600, height=600)
        vis_original.add_geometry(pcd_original)
        vis_original.get_render_option().point_size = 3
        vis_original.get_render_option().background_color = np.array([0.5, 0.5, 0.5])  # Gray background
        
        vis_true = o3d.visualization.VisualizerWithVertexSelection()
        vis_true.create_window(window_name="True Labels", width=600, height=600)
        vis_true.add_geometry(pcd_true)
        vis_true.get_render_option().point_size = 3
        vis_true.get_render_option().background_color = np.array([0.5, 0.5, 0.5])  # Gray background
        
        vis_pred = o3d.visualization.VisualizerWithVertexSelection()
        vis_pred.create_window(window_name="Predicted Labels", width=600, height=600)
        vis_pred.add_geometry(pcd_pred)
        vis_pred.get_render_option().point_size = 3
        vis_pred.get_render_option().background_color = np.array([0.5, 0.5, 0.5])  # Gray background
        
        # Update semua visualizer
        vis_original.poll_events()
        vis_original.update_renderer()
        
        vis_true.poll_events()
        vis_true.update_renderer()
        
        vis_pred.poll_events()
        vis_pred.update_renderer()
        
        # Tampilkan semua windows
        while True:
            vis_original.poll_events()
            vis_original.update_renderer()
            
            vis_true.poll_events()
            vis_true.update_renderer()
            
            vis_pred.poll_events()
            vis_pred.update_renderer()
            
            if not vis_original.poll_events() or not vis_true.poll_events() or not vis_pred.poll_events():
                break
        
        vis_original.destroy_window()
        vis_true.destroy_window()
        vis_pred.destroy_window()