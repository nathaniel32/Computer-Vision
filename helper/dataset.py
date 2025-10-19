import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import open3d as o3d
import numpy as np
from glob import glob
import os
import config
from helper.augment import Augmenter

def sample_points(xyz, rgb_raw, labels=None):
    if len(xyz) < config.NUM_SAMPLE_POINTS:
        raise ValueError(f"Jumlah titik ({len(xyz)}) lebih sedikit daripada NUM_SAMPLE_POINTS ({config.NUM_SAMPLE_POINTS})")
    
    idx = np.random.choice(len(xyz), config.NUM_SAMPLE_POINTS, replace=False)
    sampled_points = xyz[idx]
    sampled_colors = rgb_raw[idx]

    if labels is None:
        return sampled_points, sampled_colors
    else:
        sampled_labels = labels[idx]
        return sampled_points, sampled_colors, sampled_labels

def transform_color(color_int):
    # Konversi RGB dari uint32 ke 3 channel (0-1 normalized)
    colors = np.zeros((len(color_int), 3), dtype=np.float32)
    colors[:, 0] = ((color_int >> 16) & 0xFF) / 255.0  # R
    colors[:, 1] = ((color_int >> 8) & 0xFF) / 255.0   # G
    colors[:, 2] = (color_int & 0xFF) / 255.0          # B
    return colors

def transform_cloud_point(points):
    # Normalisasi point cloud
    norm_points = points.copy()
    norm_points -= np.mean(norm_points, axis=0)
    norm_points /= np.max(np.linalg.norm(norm_points, axis=1))
    return norm_points

def load_pcd_with_point_labels(directory):
    point_clouds, label_clouds, color_clouds = [], [], []
    pcd_files = glob(os.path.join(directory, "*.pcd"))
    
    print(f"\nLoading from {directory}, found {len(pcd_files)} files")
    
    for pcd_file in tqdm(pcd_files, desc=f"Loading {os.path.basename(directory)}"):
        pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=True)
        if not pcd.has_points():
            continue
        
        with open(pcd_file, 'r') as f:
            lines = f.readlines()
        
        start_index = [i for i, line in enumerate(lines) if line.startswith('DATA')][0] + 1
        data = np.loadtxt(lines[start_index:])
        
        if data.shape[1] < 5:  # x, y, z, rgb, label
            continue
        
        xyz = data[:, :3]
        rgb_raw = data[:, 3].astype(np.uint32)
        labels = data[:, 4].astype(int)

        sampled_points, sampled_colors, sampled_labels = sample_points(xyz, rgb_raw, labels=labels)
        
        # Original sample
        point_clouds.append(sampled_points)
        color_clouds.append(sampled_colors)
        label_clouds.append(sampled_labels)
    
    print(f"Total samples: {len(point_clouds)}")
    return point_clouds, color_clouds, label_clouds

class PointCloudSegmentationDataset(Dataset):
    def __init__(self, point_clouds, color_clouds, labels=None, augment=False):
        self.point_clouds = point_clouds
        self.color_clouds = [transform_color(c) for c in color_clouds]
        self.labels = labels
        self.augment = augment
        self.augmenter = Augmenter()

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        points = self.point_clouds[idx]
        colors = self.color_clouds[idx]
        
        if self.labels is None:
            points = transform_cloud_point(points)
            tensor_points = torch.FloatTensor(points)
            tensor_colors = torch.FloatTensor(colors)

            # Transpose points: (N, 3) -> (3, N)
            tensor_points = tensor_points.transpose(0, 1)
            tensor_colors = tensor_colors.transpose(0, 1)

            return tensor_points, tensor_colors
        else:
            labels = self.labels[idx]

            if self.augment:
                points, colors, labels = self.augmenter.augment(points, colors, labels)
            
            points = transform_cloud_point(points)
            tensor_points = torch.FloatTensor(points)
            tensor_colors = torch.FloatTensor(colors)
            tensor_labels = torch.LongTensor(labels)

            tensor_points = tensor_points.transpose(0, 1)
            tensor_colors = tensor_colors.transpose(0, 1)

            return tensor_points, tensor_colors, tensor_labels