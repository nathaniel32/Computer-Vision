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

    def random_rotation(self, points, colors, labels, axis=None):
        """Rotasi random 3D"""
        if np.random.random() > self.p_aug:
            return points, colors, labels

        if axis is None:
            angles = np.random.uniform(0, 2*np.pi, 3)
            rotation = R.from_euler('xyz', angles)
        else:
            angle = np.random.uniform(0, 2*np.pi)
            rotation = R.from_euler(axis, angle)

        rotated_points = rotation.apply(points)
        return rotated_points, colors, labels

    def random_scaling(self, points, colors, labels, scale_range=(0.8, 1.2)):
        """Scaling random isotropic"""
        if np.random.random() > self.p_aug:
            return points, colors, labels

        scale = np.random.uniform(scale_range[0], scale_range[1])
        scaled_points = points * scale
        return scaled_points, colors, labels

    def random_jitter(self, points, colors, labels, sigma=0.01, clip=0.05):
        """Tambah noise Gaussian kecil"""
        if np.random.random() > self.p_aug:
            return points, colors, labels

        noise = np.random.normal(0, sigma, points.shape)
        noise = np.clip(noise, -clip, clip)
        jittered_points = points + noise
        return jittered_points, colors, labels

    def random_dropout(self, points, colors, labels, dropout_rate=0.2):
        """Hapus point random (occlusion simulation)"""
        if np.random.random() > self.p_aug:
            return points, colors, labels

        num_points = len(points)
        num_drop = int(num_points * dropout_rate)
        keep_idx = np.random.choice(num_points, num_points - num_drop, replace=False)

        dropped_points = points[keep_idx]
        dropped_colors = colors[keep_idx]
        dropped_labels = labels[keep_idx]

        return dropped_points, dropped_colors, dropped_labels

    def random_translation(self, points, colors, labels, trans_range=0.2):
        """Translasi random"""
        if np.random.random() > self.p_aug:
            return points, colors, labels

        translation = np.random.uniform(-trans_range, trans_range, 3)
        translated_points = points + translation
        return translated_points, colors, labels

    def random_axis_rotation(self, points, colors, labels):
        """Rotasi hanya pada satu axis"""
        if np.random.random() > self.p_aug:
            return points, colors, labels

        axis = np.random.choice(['x', 'y', 'z'])
        angle = np.random.uniform(0, 2*np.pi)
        rotation = R.from_euler(axis, angle)

        rotated_points = rotation.apply(points)
        return rotated_points, colors, labels

    def random_flip(self, points, colors, labels, axes=[0, 1, 2]):
        """Flip random pada sumbu tertentu"""
        if np.random.random() > self.p_aug:
            return points, colors, labels

        axis = np.random.choice(axes)
        flipped_points = points.copy()
        flipped_points[:, axis] *= -1
        return flipped_points, colors, labels

    def augment(self, points, colors, labels, augmentation_list=None):
        """Augmentasi point cloud sesuai daftar augmentation"""
        if augmentation_list is None:
            augmentation_list = [
                ('rotation', {}),
                ('scaling', {'scale_range': (0.85, 1.15)}),
                ('jitter', {'sigma': 0.01}),
                ('translation', {'trans_range': 0.1}),
            ]

        aug_points = points.copy()
        aug_colors = colors.copy()
        aug_labels = labels.copy()

        for aug_name, aug_params in augmentation_list:
            if aug_name == 'rotation':
                aug_points, aug_colors, aug_labels = self.random_rotation(
                    aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'axis_rotation':
                aug_points, aug_colors, aug_labels = self.random_axis_rotation(
                    aug_points, aug_colors, aug_labels)
            elif aug_name == 'scaling':
                aug_points, aug_colors, aug_labels = self.random_scaling(
                    aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'jitter':
                aug_points, aug_colors, aug_labels = self.random_jitter(
                    aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'dropout':
                aug_points, aug_colors, aug_labels = self.random_dropout(
                    aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'translation':
                aug_points, aug_colors, aug_labels = self.random_translation(
                    aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'flip':
                aug_points, aug_colors, aug_labels = self.random_flip(
                    aug_points, aug_colors, aug_labels, **aug_params)

        return aug_points, aug_colors, aug_labels

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

""" def transform_data(xyz, rgb_raw, labels=None):
    # Konversi RGB dari uint32 ke 3 channel (0-1 normalized)
    rgb = np.zeros((len(rgb_raw), 3), dtype=np.float32)
    rgb[:, 0] = ((rgb_raw >> 16) & 0xFF) / 255.0  # R
    rgb[:, 1] = ((rgb_raw >> 8) & 0xFF) / 255.0   # G
    rgb[:, 2] = (rgb_raw & 0xFF) / 255.0          # B

    # Normalisasi posisi titik
    xyz_centered = xyz - np.mean(xyz, axis=0)
    xyz_normalized = xyz_centered / np.max(np.linalg.norm(xyz_centered, axis=1))

    # Warna disesuaikan
    sampled_points = xyz_normalized
    sampled_colors = rgb

    if labels is None:
        return sampled_points, sampled_colors
    else:
        sampled_labels = labels
        return sampled_points, sampled_colors, sampled_labels """

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

        sampled_points, sampled_colors, sampled_labels = sample_points(xyz, rgb_raw, labels=labels)

        norm_points, norm_colors = transform_data(sampled_points, sampled_colors)
                
        # Original sample
        point_clouds.append(norm_points)
        color_clouds.append(norm_colors)
        label_clouds.append(sampled_labels)
        
        # Augmented samples
        if augment:
            for _ in range(num_augmentations):
                aug_points, aug_colors, aug_labels = augmenter.augment(
                    norm_points, norm_colors, sampled_labels
                )
                point_clouds.append(aug_points)
                color_clouds.append(aug_colors)
                label_clouds.append(aug_labels)
    
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