import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import open3d as o3d
import numpy as np
from glob import glob
import os
import config
from helper.plot import plot_point_cloud
from helper.augment import PointCloudColorAugmenter, PointCloudAugmenter

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

def transform_data(points, rgb_raw):
    # Konversi RGB dari uint32 ke 3 channel (0-1 normalized)
    colors = np.zeros((len(rgb_raw), 3), dtype=np.float32)
    colors[:, 0] = ((rgb_raw >> 16) & 0xFF) / 255.0  # R
    colors[:, 1] = ((rgb_raw >> 8) & 0xFF) / 255.0   # G
    colors[:, 2] = (rgb_raw & 0xFF) / 255.0          # B

    # Normalisasi point cloud
    norm_points = points.copy()
    norm_points -= np.mean(norm_points, axis=0)
    norm_points /= np.max(np.linalg.norm(norm_points, axis=1))

    return norm_points, colors

def load_pcd_with_point_labels(directory, augment=False, num_augmentations=2):
    color_augmenter = PointCloudColorAugmenter()
    augmenter = PointCloudAugmenter(p_aug=0.8)

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

        norm_points, norm_colors = transform_data(sampled_points, sampled_colors)
                
        # Original sample
        point_clouds.append(norm_points)
        color_clouds.append(norm_colors)
        label_clouds.append(sampled_labels)
        
        # Augmented samples
        if augment:
            
            colors_aug = color_augmenter.original(norm_colors, sampled_labels)
            point_clouds.append(norm_points)
            color_clouds.append(colors_aug)
            label_clouds.append(sampled_labels)
            #plot_point_cloud(norm_points, colors_aug, true_label=sampled_labels)

            colors_aug = color_augmenter.negative(norm_colors, sampled_labels)
            point_clouds.append(norm_points)
            color_clouds.append(colors_aug)
            label_clouds.append(sampled_labels)
            #plot_point_cloud(norm_points, colors_aug, true_label=sampled_labels)

            colors_aug = color_augmenter.original_to_color(norm_colors, sampled_labels, '#FF6600')
            point_clouds.append(norm_points)
            color_clouds.append(colors_aug)
            label_clouds.append(sampled_labels)
            #plot_point_cloud(norm_points, colors_aug, true_label=sampled_labels)

            colors_aug = color_augmenter.negative_to_color(norm_colors, sampled_labels, '#FF6600')
            point_clouds.append(norm_points)
            color_clouds.append(colors_aug)
            label_clouds.append(sampled_labels)
            #plot_point_cloud(norm_points, colors_aug, true_label=sampled_labels)

            for _ in range(num_augmentations):
                aug_points, aug_colors, aug_labels = augmenter.augment(norm_points, norm_colors, sampled_labels)
                point_clouds.append(aug_points)
                color_clouds.append(aug_colors)
                label_clouds.append(aug_labels)

                # plot augment
                #plot_point_cloud(aug_points, aug_colors, true_label=aug_labels)
    
    print(f"Total samples (with augmentation): {len(point_clouds)}")
    return point_clouds, color_clouds, label_clouds

class PointCloudSegmentationDataset(Dataset):
    def __init__(self, point_clouds, color_clouds, labels=None):
        self.point_clouds = point_clouds
        self.color_clouds = color_clouds
        self.labels = labels

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        points = torch.FloatTensor(self.point_clouds[idx])
        colors = torch.FloatTensor(self.color_clouds[idx])

        # Transpose points: (N, 3) -> (3, N)
        points = points.transpose(0, 1)
        colors = colors.transpose(0, 1)

        if self.labels is None:
            return points, colors
        else:
            labels = torch.LongTensor(self.labels[idx])
            return points, colors, labels