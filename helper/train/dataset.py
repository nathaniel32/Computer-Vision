import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from glob import glob
import os
import config
from helper.train.augment import Augmenter

def sample_points(points, colors, labels=None):
    if len(points) < config.NUM_SAMPLE_POINTS:
        raise ValueError(f"the number of points is less than NUM_SAMPLE_POINTS {len(points)}/{config.NUM_SAMPLE_POINTS}")
    
    idx = np.random.choice(len(points), config.NUM_SAMPLE_POINTS, replace=False)
    sampled_points = points[idx]
    sampled_colors = colors[idx]

    if labels is None:
        return sampled_points, sampled_colors
    else:
        sampled_labels = labels[idx]
        return sampled_points, sampled_colors, sampled_labels

def transform_color(color_int):
    # RGB conversion from uint32 to 3 channels (0-1 normalized)
    colors = np.zeros((len(color_int), 3), dtype=np.float32)
    colors[:, 0] = ((color_int >> 16) & 0xFF) / 255.0  # R
    colors[:, 1] = ((color_int >> 8) & 0xFF) / 255.0   # G
    colors[:, 2] = (color_int & 0xFF) / 255.0          # B
    return colors

def transform_cloud_point(points):
    # Point cloud normalization
    norm_points = points.copy()
    norm_points -= np.mean(norm_points, axis=0)
    norm_points /= np.max(np.linalg.norm(norm_points, axis=1))
    return norm_points

def load_pcd_with_point_labels(directory, sampling=False):
    point_clouds, label_clouds, color_clouds = [], [], []
    pcd_files = glob(os.path.join(directory, "*.pcd"))
    
    print(f"\nLoading from {directory}, found {len(pcd_files)} files")
    
    for pcd_file in tqdm(pcd_files, desc=f"Loading {os.path.basename(directory)}"):        
        with open(pcd_file, 'r') as f:
            lines = f.readlines()
        
        start_index = [i for i, line in enumerate(lines) if line.startswith('DATA')][0] + 1
        data = np.loadtxt(lines[start_index:])
        
        if data.shape[1] < 5:  # x, y, z, rgb, label
            continue
        
        points = data[:, :3]
        color_ints = data[:, 3].astype(np.uint32)
        labels = data[:, 4].astype(int)

        if sampling:
            points, color_ints, labels = sample_points(points, color_ints, labels=labels)
        
        # Original sample
        point_clouds.append(points)
        color_clouds.append(color_ints)
        label_clouds.append(labels)
    
    print(f"Total samples: {len(point_clouds)}")
    return point_clouds, color_clouds, label_clouds

class PointCloudSegmentationDataset(Dataset):
    def __init__(self, point_clouds, color_clouds, labels=None, augment=False, sampling=False):
        self.point_clouds = point_clouds
        self.color_clouds = [transform_color(c) for c in color_clouds]
        self.labels = labels
        self.augment = augment
        self.sampling = sampling
        self.augmenter = Augmenter()

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        points = self.point_clouds[idx]
        colors = self.color_clouds[idx]
        
        if self.labels is None:
            if self.sampling:
                points, colors = sample_points(points, colors)
            
            points = transform_cloud_point(points)
            tensor_points = torch.FloatTensor(points)
            tensor_colors = torch.FloatTensor(colors)

            # Transpose points: (N, 3) -> (3, N)
            tensor_points = tensor_points.transpose(0, 1)
            tensor_colors = tensor_colors.transpose(0, 1)

            return tensor_points, tensor_colors
        else:
            labels = self.labels[idx]

            if self.sampling:
                points, colors, labels = sample_points(points, colors, labels=labels)

            if self.augment:
                #plot_point_cloud(points, colors, true_label=labels)
                points, colors, labels = self.augmenter.augment(points, colors, labels)
                #plot_point_cloud(points, colors, true_label=labels)
            
            points = transform_cloud_point(points)
            #plot_point_cloud(points, colors, true_label=labels)

            tensor_points = torch.FloatTensor(points)
            tensor_colors = torch.FloatTensor(colors)
            tensor_labels = torch.LongTensor(labels)

            tensor_points = tensor_points.transpose(0, 1)
            tensor_colors = tensor_colors.transpose(0, 1)

            return tensor_points, tensor_colors, tensor_labels
        
def get_chunks_indices(n_data, chunk_size):
    rand_indices = np.random.permutation(n_data)
    chunks_indices = [rand_indices[i:i + chunk_size] for i in range(0, n_data, chunk_size)]
    return chunks_indices