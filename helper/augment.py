import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

class PointCloudColorAugmenter:
    """Main class untuk augmentasi warna point cloud"""
    
    def __init__(self, target_label=1):
        self.target_label = target_label
        self.hex_colors_list = ['#FF6600', '#00CCFF', '#FF00FF', '#FFFF00']
    
    def _get_mask(self, labels):
        """Get mask untuk target label"""
        return labels == self.target_label
    
    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert hex color ke RGB normalized (0-1)"""
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
    
    @staticmethod
    def _get_hue_from_hex(hex_color):
        """Extract hue value dari hex color"""
        rgb = PointCloudColorAugmenter.hex_to_rgb(hex_color)
        rgb = rgb.reshape(1, -1)
        hsv = rgb_to_hsv(rgb)
        return hsv[0, 0]
    
    def _apply_negative(self, colors, labels):
        """Helper: apply negative ke target label"""
        colors_aug = colors.copy()
        mask = self._get_mask(labels)
        colors_aug[mask] = 1.0 - colors_aug[mask]
        return colors_aug
    
    def _apply_hue_shift(self, colors, labels, hue_value, use_negative=False):
        """Helper: shift ke target hue"""
        colors_aug = colors.copy()
        mask = self._get_mask(labels)
        
        if use_negative:
            colors_aug[mask] = 1.0 - colors_aug[mask]
        
        hsv = rgb_to_hsv(colors_aug[mask])
        hsv[:, 0] = hue_value % 1.0
        hsv[:, 1] = np.clip(hsv[:, 1] * 1.2, 0, 1)
        colors_aug[mask] = hsv_to_rgb(hsv)
        
        return colors_aug
    
    # =====================
    # Original Augmentations
    # =====================
    
    def original(self, colors, labels):
        """Original warna"""
        return colors.copy()
    
    def original_to_color(self, colors, labels, hex_color):
        """Original + shift ke custom color (hex)"""
        hue = self._get_hue_from_hex(hex_color)
        return self._apply_hue_shift(colors, labels, hue, use_negative=False)
    
    # =====================
    # Negative Augmentations
    # =====================
    
    def negative(self, colors, labels):
        """Negative (invert RGB)"""
        return self._apply_negative(colors, labels)
    
    def negative_to_color(self, colors, labels, hex_color):
        """Negative + shift ke custom color (hex)"""
        hue = self._get_hue_from_hex(hex_color)
        return self._apply_hue_shift(colors, labels, hue, use_negative=True)
    
    # =====================
    # Get Augmentations
    # =====================
    
    def augment(self):
        """Return list augmentasi dengan custom colors untuk training loop
        
        Args:
            hex_colors_list: list hex colors ['#FF0000', '#00FF00', '#0000FF', ...]
        """
        aug_list = [self.original, self.negative]
        
        for hex_color in self.hex_colors_list:
            aug_list.append(lambda c, l, hc=hex_color: self.negative_to_color(c, l, hc))
        
        for hex_color in self.hex_colors_list:
            aug_list.append(lambda c, l, hc=hex_color: self.original_to_color(c, l, hc))
        
        return aug_list

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
                #('scaling', {'scale_range': (0.85, 1.15)}),
                #('jitter', {'sigma': 0.01}),
                ('translation', {'trans_range': 0.1}),

                ('axis_rotation', {}),
                ('flip', {'axes': [0, 1]}),
                #('dropout', {'dropout_rate': 0.15})
            ]

        aug_points = points.copy()
        aug_colors = colors.copy()
        aug_labels = labels.copy()

        for aug_name, aug_params in augmentation_list:
            if aug_name == 'rotation':
                aug_points, aug_colors, aug_labels = self.random_rotation(aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'axis_rotation':
                aug_points, aug_colors, aug_labels = self.random_axis_rotation(aug_points, aug_colors, aug_labels)
            elif aug_name == 'scaling':
                aug_points, aug_colors, aug_labels = self.random_scaling(aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'jitter':
                aug_points, aug_colors, aug_labels = self.random_jitter(aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'dropout':
                aug_points, aug_colors, aug_labels = self.random_dropout(aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'translation':
                aug_points, aug_colors, aug_labels = self.random_translation(aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'flip':
                aug_points, aug_colors, aug_labels = self.random_flip(aug_points, aug_colors, aug_labels, **aug_params)

        return aug_points, aug_colors, aug_labels