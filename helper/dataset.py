import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import open3d as o3d
import numpy as np
from glob import glob
import os
import config
from scipy.spatial.transform import Rotation as R

class PointCloudAugmenter:
    def __init__(self, p_aug=0.7):
        self.p_aug = p_aug
    
    def random_rotation(self, points, labels, colors=None, axis=None):
        """Rotasi random 3D"""
        if np.random.random() > self.p_aug:
            return points, labels, colors
        
        if axis is None:
            angles = np.random.uniform(0, 2*np.pi, 3)
            rotation = R.from_euler('xyz', angles)
        else:
            angle = np.random.uniform(0, 2*np.pi)
            rotation = R.from_euler(axis, angle)
        
        rotated_points = rotation.apply(points)
        return rotated_points, labels, colors
    
    def random_scaling(self, points, labels, colors=None, scale_range=(0.8, 1.2)):
        """Scaling random isotropic"""
        if np.random.random() > self.p_aug:
            return points, labels, colors
        
        scale = np.random.uniform(scale_range[0], scale_range[1])
        scaled_points = points * scale
        return scaled_points, labels, colors
    
    def random_jitter(self, points, labels, colors=None, sigma=0.01, clip=0.05):
        """Tambah noise Gaussian kecil"""
        if np.random.random() > self.p_aug:
            return points, labels, colors
        
        noise = np.random.normal(0, sigma, points.shape)
        noise = np.clip(noise, -clip, clip)
        jittered_points = points + noise
        return jittered_points, labels, colors
    
    def random_dropout(self, points, labels, colors=None, dropout_rate=0.2):
        """Hapus point random (occlusion simulation)"""
        if np.random.random() > self.p_aug:
            return points, labels, colors
        
        num_points = len(points)
        num_drop = int(num_points * dropout_rate)
        keep_idx = np.random.choice(num_points, num_points - num_drop, replace=False)
        
        dropped_points = points[keep_idx]
        dropped_labels = labels[keep_idx]
        dropped_colors = colors[keep_idx] if colors is not None else None
        
        return dropped_points, dropped_labels, dropped_colors
    
    def random_translation(self, points, labels, colors=None, trans_range=0.2):
        """Translasi random"""
        if np.random.random() > self.p_aug:
            return points, labels, colors
        
        translation = np.random.uniform(-trans_range, trans_range, 3)
        translated_points = points + translation
        return translated_points, labels, colors
    
    def random_axis_rotation(self, points, labels, colors=None):
        """Rotasi hanya pada satu axis"""
        if np.random.random() > self.p_aug:
            return points, labels, colors
        
        axis = np.random.choice(['x', 'y', 'z'])
        angle = np.random.uniform(0, 2*np.pi)
        rotation = R.from_euler(axis, angle)
        
        rotated_points = rotation.apply(points)
        return rotated_points, labels, colors
    
    def random_flip(self, points, labels, colors=None, axes=[0, 1, 2]):
        """Flip random pada sumbu tertentu"""
        if np.random.random() > self.p_aug:
            return points, labels, colors
        
        axis = np.random.choice(axes)
        flipped_points = points.copy()
        flipped_points[:, axis] *= -1
        return flipped_points, labels, colors
    
    def augment(self, points, labels, colors=None, augmentation_list=None):
        if augmentation_list is None:
            augmentation_list = [
                ('rotation', {}),
                ('scaling', {'scale_range': (0.85, 1.15)}),
                ('jitter', {'sigma': 0.01}),
                ('translation', {'trans_range': 0.1}),
            ]
        
        aug_points = points.copy()
        aug_labels = labels.copy()
        aug_colors = colors.copy() if colors is not None else None
        
        for aug_name, aug_params in augmentation_list:
            if aug_name == 'rotation':
                aug_points, aug_labels, aug_colors = self.random_rotation(
                    aug_points, aug_labels, aug_colors, **aug_params)
            elif aug_name == 'axis_rotation':
                aug_points, aug_labels, aug_colors = self.random_axis_rotation(
                    aug_points, aug_labels, aug_colors)
            elif aug_name == 'scaling':
                aug_points, aug_labels, aug_colors = self.random_scaling(
                    aug_points, aug_labels, aug_colors, **aug_params)
            elif aug_name == 'jitter':
                aug_points, aug_labels, aug_colors = self.random_jitter(
                    aug_points, aug_labels, aug_colors, **aug_params)
            elif aug_name == 'dropout':
                aug_points, aug_labels, aug_colors = self.random_dropout(
                    aug_points, aug_labels, aug_colors, **aug_params)
            elif aug_name == 'translation':
                aug_points, aug_labels, aug_colors = self.random_translation(
                    aug_points, aug_labels, aug_colors, **aug_params)
            elif aug_name == 'flip':
                aug_points, aug_labels, aug_colors = self.random_flip(
                    aug_points, aug_labels, aug_colors, **aug_params)
        
        return aug_points, aug_labels, aug_colors

def transform_data(xyz, rgb_raw, labels=None):
    # Konversi RGB dari uint32 ke 3 channel (0-1 normalized)
    rgb = np.zeros((len(rgb_raw), 3), dtype=np.float32)
    rgb[:, 0] = ((rgb_raw >> 16) & 0xFF) / 255.0  # R
    rgb[:, 1] = ((rgb_raw >> 8) & 0xFF) / 255.0   # G
    rgb[:, 2] = (rgb_raw & 0xFF) / 255.0          # B
    
    if len(xyz) < config.NUM_SAMPLE_POINTS:
        return None, None, None
    
    # Sampling & normalisasi
    idx = np.random.choice(len(xyz), config.NUM_SAMPLE_POINTS, replace=False)
    sampled_points = xyz[idx]
    sampled_points -= np.mean(sampled_points, axis=0)
    sampled_points /= np.max(np.linalg.norm(sampled_points, axis=1))
    sampled_colors = rgb[idx]

    if labels is None:
        return sampled_points, sampled_colors
    else:
        sampled_labels = labels[idx]
        return sampled_points, sampled_colors, sampled_labels

def load_pcd_with_point_labels(directory, augment=False, num_augmentations=2):
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
        
        sampled_points, sampled_colors, sampled_labels = transform_data(xyz, rgb_raw, labels=labels)

        if sampled_points is None or sampled_colors is None or sampled_labels is None:
            print(f"Data Error!")
            continue
        
        # Original sample
        point_clouds.append(sampled_points)
        label_clouds.append(sampled_labels)
        color_clouds.append(sampled_colors)
        
        # Augmented samples
        if augment:
            for _ in range(num_augmentations):
                aug_points, aug_labels, aug_colors = augmenter.augment(
                    sampled_points, sampled_labels, sampled_colors
                )
                point_clouds.append(aug_points)
                label_clouds.append(aug_labels)
                color_clouds.append(aug_colors)
    
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