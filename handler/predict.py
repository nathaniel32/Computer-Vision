import torch
import os
import configs
from helper.train.dataset import PointCloudSegmentationDataset, get_chunks_indices
from helper.train.model import get_predict_model
from helper.plot import plot_point_cloud, PlotTool
import helper.mesh.mesh_remover
from helper.mesh.mesh_converter import convert_mesh_to_point_cloud_folder, save_point_cloud_in_pcd
import numpy as np
from helper.mesh.mesh_scaler import MeshScaler

class Predict:
    def __init__(self, config:configs.BaseConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.model_num_points, self.classes = get_predict_model(self.config.save_model_path, self.device)
        self.mesh_scaler = MeshScaler(self.config)

    def _predicting(self, input_dir_path, output_dir_path, smoothing=False):
        os.makedirs(output_dir_path, exist_ok=True)

        total_num_points = self.config.total_num_points
        
        self.model.eval()
        with torch.no_grad():
            # divide into chunks
            chunks_indices = get_chunks_indices(total_num_points, self.model_num_points)

            # obj to point cloud
            points, colors_int, mesh_file_path = convert_mesh_to_point_cloud_folder(input_dir_path, total_num_points=total_num_points)

            comb_points = []
            comb_color_norm = []
            comb_pred_label = []
            comb_color_int = []

            for i, chunk_indices in enumerate(chunks_indices, start=1):
                print(f"- Chunk {i}/{len(chunks_indices)}")
                points_chunk = points[chunk_indices]
                colors_int_chunk = colors_int[chunk_indices]
                pred_dataset = PointCloudSegmentationDataset([points_chunk], [colors_int_chunk])
                
                for (t_point, t_color) in pred_dataset:
                    t_point = t_point.unsqueeze(0).to(self.device)
                    t_color = t_color.unsqueeze(0).to(self.device)
                    outputs = self.model(t_point, t_color)

                    point_plot = t_point.squeeze(0).transpose(0, 1).cpu().numpy()
                    color_plot = t_color.squeeze(0).transpose(0, 1).cpu().numpy()
                    pred_label = outputs.squeeze(0).argmax(dim=1).cpu().numpy()
                    if smoothing:
                        pred_label = helper.mesh.mesh_remover.smooth_labels(points=point_plot, pred_label=pred_label) #extra smoothing

                    comb_points.extend(points_chunk)
                    comb_color_norm.extend(color_plot)
                    comb_pred_label.extend(pred_label)
                    comb_color_int.extend(colors_int_chunk)
                    
            comb_points = np.array(comb_points)
            comb_color_norm = np.array(comb_color_norm)
            comb_pred_label = np.array(comb_pred_label)

            save_point_cloud_in_pcd(comb_points, comb_color_int, output_dir_path, label=comb_pred_label)

            return comb_points, comb_color_norm, comb_pred_label, mesh_file_path

    def cleaning_object(self, input_dir_path, output_dir_path, keep_label, plot=True, plot_dir=None, headless=False):
        points, color, pred_label, mesh_file_path = self._predicting(input_dir_path, output_dir_path)

        if plot:
            plot_point_cloud(points, color, self.classes, pred_label=pred_label, plot_tool=PlotTool.OPEN3D, save_dir=plot_dir, headless=headless, file_base_name="cleaning")
        
        # trim mesh
        trim_out_path = os.path.join(output_dir_path, "trim_mesh.obj")
        helper.mesh.mesh_remover.remove_object_part_v2(points, pred_label, mesh_file_path, trim_out_path, keep_label)
        

        keep_indecies = pred_label == keep_label # keep
        remove_indecies = pred_label != keep_label # remove
        
        if plot:
            plot_point_cloud(points[keep_indecies], color[keep_indecies], self.classes, pred_label=pred_label[keep_indecies], plot_tool=PlotTool.OPEN3D, save_dir=plot_dir, file_base_name="cleaning_keep", headless=headless)
            plot_point_cloud(points[remove_indecies], color[remove_indecies], self.classes, pred_label=pred_label[remove_indecies], plot_tool=PlotTool.OPEN3D, save_dir=plot_dir, file_base_name="cleaning_remove", headless=headless)

    def scaling_object(self, input_dir_path, output_dir_path, real_marker_pair_length, real_marker_pair_center_distance, plot=True, plot_dir=None, headless=False):
        points, color, pred_label, mesh_file_path = self._predicting(input_dir_path, output_dir_path)

        if plot:
            plot_point_cloud(points, color, self.classes, pred_label=pred_label, plot_tool=PlotTool.OPEN3D, save_dir=plot_dir, headless=headless, file_base_name="scaling")

        scale_factor, best_marker_pairs = self.mesh_scaler.calculate_scale_factor(points, color, pred_label, real_marker_pair_length, real_marker_pair_center_distance, quality_check=False, plot=plot, plot_dir=plot_dir, headless=headless)

        print(scale_factor)
        