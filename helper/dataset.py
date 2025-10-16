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
    """Augmentasi point cloud untuk tugas segmentation"""
    
    def __init__(self, p_aug=0.7):
        """
        Args:
            p_aug: probabilitas menerapkan augmentasi (0-1)
        """
        self.p_aug = p_aug
    
    def random_rotation(self, points, labels, axis=None):
        """Rotasi random 3D"""
        if np.random.random() > self.p_aug:
            return points, labels
        
        if axis is None:
            # Rotasi penuh 3D
            angles = np.random.uniform(0, 2*np.pi, 3)
            rotation = R.from_euler('xyz', angles)
        else:
            # Rotasi pada axis tertentu (misal Z untuk top-down)
            angle = np.random.uniform(0, 2*np.pi)
            rotation = R.from_euler(axis, angle)
        
        rotated_points = rotation.apply(points)
        return rotated_points, labels
    
    def random_scaling(self, points, labels, scale_range=(0.8, 1.2)):
        """Scaling random isotropic"""
        if np.random.random() > self.p_aug:
            return points, labels
        
        scale = np.random.uniform(scale_range[0], scale_range[1])
        scaled_points = points * scale
        return scaled_points, labels
    
    def random_jitter(self, points, labels, sigma=0.01, clip=0.05):
        """Tambah noise Gaussian kecil"""
        if np.random.random() > self.p_aug:
            return points, labels
        
        noise = np.random.normal(0, sigma, points.shape)
        noise = np.clip(noise, -clip, clip)
        jittered_points = points + noise
        return jittered_points, labels
    
    def random_dropout(self, points, labels, dropout_rate=0.2):
        """Hapus point random (occlusion simulation)"""
        if np.random.random() > self.p_aug:
            return points, labels
        
        num_points = len(points)
        num_drop = int(num_points * dropout_rate)
        keep_idx = np.random.choice(num_points, num_points - num_drop, replace=False)
        
        return points[keep_idx], labels[keep_idx]
    
    def random_translation(self, points, labels, trans_range=0.2):
        """Translasi random"""
        if np.random.random() > self.p_aug:
            return points, labels
        
        translation = np.random.uniform(-trans_range, trans_range, 3)
        translated_points = points + translation
        return translated_points, labels
    
    def random_axis_rotation(self, points, labels):
        """Rotasi hanya pada satu axis (lebih naturalistik)"""
        if np.random.random() > self.p_aug:
            return points, labels
        
        axis = np.random.choice(['x', 'y', 'z'])
        angle = np.random.uniform(0, 2*np.pi)
        rotation = R.from_euler(axis, angle)
        
        rotated_points = rotation.apply(points)
        return rotated_points, labels
    
    def random_flip(self, points, labels, axes=[0, 1, 2]):
        """Flip random pada sumbu tertentu"""
        if np.random.random() > self.p_aug:
            return points, labels
        
        axis = np.random.choice(axes)
        flipped_points = points.copy()
        flipped_points[:, axis] *= -1
        return flipped_points, labels
    
    def augment(self, points, labels, augmentation_list=None):
        """
        Terapkan augmentasi secara sequential
        
        Args:
            points: (N, 3) point cloud
            labels: (N,) label per point
            augmentation_list: list of augmentation methods
        
        Returns:
            aug_points, aug_labels
        """
        if augmentation_list is None:
            augmentation_list = [
                ('rotation', {}),
                ('scaling', {'scale_range': (0.85, 1.15)}),
                ('jitter', {'sigma': 0.01}),
                ('translation', {'trans_range': 0.1}),
            ]
        
        aug_points, aug_labels = points.copy(), labels.copy()
        
        for aug_name, aug_params in augmentation_list:
            if aug_name == 'rotation':
                aug_points, aug_labels = self.random_rotation(aug_points, aug_labels, **aug_params)
            elif aug_name == 'axis_rotation':
                aug_points, aug_labels = self.random_axis_rotation(aug_points, aug_labels)
            elif aug_name == 'scaling':
                aug_points, aug_labels = self.random_scaling(aug_points, aug_labels, **aug_params)
            elif aug_name == 'jitter':
                aug_points, aug_labels = self.random_jitter(aug_points, aug_labels, **aug_params)
            elif aug_name == 'dropout':
                aug_points, aug_labels = self.random_dropout(aug_points, aug_labels, **aug_params)
            elif aug_name == 'translation':
                aug_points, aug_labels = self.random_translation(aug_points, aug_labels, **aug_params)
            elif aug_name == 'flip':
                aug_points, aug_labels = self.random_flip(aug_points, aug_labels, **aug_params)
        
        return aug_points, aug_labels

def load_pcd_with_point_labels(directory, augment=False, num_augmentations=2):
    """Load point cloud dengan augmentasi"""
    augmenter = PointCloudAugmenter(p_aug=0.8)
    point_clouds, label_clouds = [], []
    pcd_files = glob(os.path.join(directory, "*.pcd"))
    
    print(f"\nLoading from {directory}, found {len(pcd_files)} files")
    
    for pcd_file in tqdm(pcd_files, desc=f"Loading {os.path.basename(directory)}"):
        pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=True)
        if not pcd.has_points():
            continue
        
        try:
            with open(pcd_file, 'r') as f:
                lines = f.readlines()
            
            start_index = [i for i, line in enumerate(lines) if line.startswith('DATA')][0] + 1
            data = np.loadtxt(lines[start_index:])
            
            if data.shape[1] < 4:
                continue
            
            xyz = data[:, :3]
            labels = data[:, 3].astype(int)
            
            if len(xyz) < config.NUM_SAMPLE_POINTS:
                continue
            
            # Sampling & normalisasi
            idx = np.random.choice(len(xyz), config.NUM_SAMPLE_POINTS, replace=False)
            sampled_points = xyz[idx]
            sampled_labels = labels[idx]
            
            sampled_points -= np.mean(sampled_points, axis=0)
            sampled_points /= np.max(np.linalg.norm(sampled_points, axis=1))
            
            # Original sample
            point_clouds.append(sampled_points)
            label_clouds.append(sampled_labels)
            
            # Augmented samples
            if augment:
                for _ in range(num_augmentations):
                    aug_points, aug_labels = augmenter.augment(
                        sampled_points, sampled_labels
                    )
                    point_clouds.append(aug_points)
                    label_clouds.append(aug_labels)
        
        except Exception as e:
            print(f"Error reading {pcd_file}: {e}")
            continue
    
    print(f"Total samples (with augmentation): {len(point_clouds)}")
    return point_clouds, label_clouds

""" def load_pcd_with_point_labels(directory):
    point_clouds, label_clouds = [], []
    pcd_files = glob(os.path.join(directory, "*.pcd"))
    print(f"\nLoading from {directory}, found {len(pcd_files)} files")

    for pcd_file in tqdm(pcd_files, desc=f"Loading {os.path.basename(directory)}"):
        pcd = o3d.io.read_point_cloud(pcd_file, remove_nan_points=True)
        if not pcd.has_points():
            continue

        # Open3D tidak membaca kolom 'label', jadi kita ambil manual via numpy
        try:
            # Baca file pcd manual
            with open(pcd_file, 'r') as f:
                lines = f.readlines()

            # Cari baris awal data
            start_index = [i for i, line in enumerate(lines) if line.startswith('DATA')][0] + 1
            data = np.loadtxt(lines[start_index:])
            
            # Ambil kolom x, y, z, label
            if data.shape[1] < 4:
                continue
            xyz = data[:, :3]
            labels = data[:, 3].astype(int)

            if len(xyz) < config.NUM_SAMPLE_POINTS:
                continue

            # Sampling & normalisasi
            idx = np.random.choice(len(xyz), config.NUM_SAMPLE_POINTS, replace=False)
            sampled_points = xyz[idx]
            sampled_labels = labels[idx]

            sampled_points -= np.mean(sampled_points, axis=0)
            sampled_points /= np.max(np.linalg.norm(sampled_points, axis=1))

            point_clouds.append(sampled_points)
            label_clouds.append(sampled_labels)
        except Exception as e:
            print(f"Error reading {pcd_file}: {e}")
            continue

    return point_clouds, label_clouds """

class PointCloudSegmentationDataset(Dataset):
    def __init__(self, point_clouds, labels, augment=False):
        self.point_clouds = point_clouds
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        points = torch.FloatTensor(self.point_clouds[idx])
        labels = torch.LongTensor(self.labels[idx])

        """ if self.augment:
            noise = torch.randn_like(points) * 0.005
            points += noise """

        # (N, 3) -> (3, N)
        points = points.transpose(0, 1)
        
        return points, labels
    
    """ def __getitem__(self, idx):
        points = torch.FloatTensor(self.point_clouds[idx])
        labels = torch.LongTensor(self.labels[idx])

        if self.augment:
            # 1. Random Rotation (Z-axis)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-np.pi, np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotation_matrix = torch.tensor([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ], dtype=points.dtype)
                points = torch.matmul(points, rotation_matrix)
            
            # 2. Random Scaling
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.8, 1.2)
                points = points * scale
            
            # 3. Random Jitter
            if np.random.rand() > 0.5:
                noise = torch.randn_like(points) * 0.01
                points = points + noise
            
            # 4. Random Translation
            if np.random.rand() > 0.5:
                translation = torch.FloatTensor(3).uniform_(-0.2, 0.2)
                points = points + translation

        # (N, 3) -> (3, N)
        points = points.transpose(0, 1)
        
        return points, labels """