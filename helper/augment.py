import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import config
import random

class ColorPartAugmenter: #tidak merusak norm
    """Augmentasi warna point cloud"""
    
    def __init__(self, target_label=config.TARGET_CLASS_ID, p_aug=0.7):
        self.target_label = target_label
        self.p_aug = p_aug
        self.color_augment_list = self.get_augment_list()
    
    def _get_mask(self, labels):
        return labels == self.target_label
    
    def _apply_negative(self, colors, labels):
        colors_aug = colors.copy()
        mask = self._get_mask(labels)
        colors_aug[mask] = 1.0 - colors_aug[mask]
        return colors_aug
    
    def _apply_hue_shift(self, colors, labels, use_negative=False):
        colors_aug = colors.copy()
        mask = self._get_mask(labels)
        
        if use_negative:
            colors_aug[mask] = 1.0 - colors_aug[mask]
        
        hue = random.uniform(0, 1.0)  # Hue dalam range [0, 1]
        
        hsv = rgb_to_hsv(colors_aug[mask])
        hsv[:, 0] = hue
        hsv[:, 1] = np.clip(hsv[:, 1] * 1.2, 0, 1)
        colors_aug[mask] = hsv_to_rgb(hsv)
        
        return colors_aug
        
    def original_to_color(self, colors, labels):
        return self._apply_hue_shift(colors, labels, use_negative=False)
    
    def negative(self, colors, labels):
        return self._apply_negative(colors, labels)
    
    def negative_to_color(self, colors, labels):
        return self._apply_hue_shift(colors, labels, use_negative=True)
    
    def get_augment_list(self):
        """Return list augmentasi"""
        return [
            self.negative,
            self.negative_to_color,
            self.original_to_color
        ]
    
    def _should_augment(self):
        return np.random.random() < self.p_aug
    
    def augment(self, colors, labels):
        if not self._should_augment():
            return colors
        
        # Random color augmentasi
        color_idx = np.random.randint(0, len(self.color_augment_list))
        color_aug_func = self.color_augment_list[color_idx]
        aug_colors = color_aug_func(colors, labels)
        return aug_colors

class ObjectAugmenter: # merusak norm
    """Augmentasi geometri dan warna point cloud"""
    
    def __init__(self, p_aug=0.7):
        self.p_aug = p_aug
    
    def _should_augment(self):
        return np.random.random() < self.p_aug

    # ===== GEOMETRIC AUGMENTATIONS =====
    def random_rotation(self, points, colors, labels, axis=None):
        if not self._should_augment():
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
        if not self._should_augment():
            return points, colors, labels

        scale = np.random.uniform(scale_range[0], scale_range[1])
        scaled_points = points * scale
        return scaled_points, colors, labels

    def random_jitter(self, points, colors, labels, sigma=0.01, clip=0.05):
        if not self._should_augment():
            return points, colors, labels

        noise = np.random.normal(0, sigma, points.shape)
        noise = np.clip(noise, -clip, clip)
        jittered_points = points + noise
        return jittered_points, colors, labels

    def random_dropout(self, points, colors, labels, dropout_rate=0.2):
        if not self._should_augment():
            return points, colors, labels

        num_points = len(points)
        num_drop = int(num_points * dropout_rate)
        keep_idx = np.random.choice(num_points, num_points - num_drop, replace=False)

        return points[keep_idx], colors[keep_idx], labels[keep_idx]

    def random_translation(self, points, colors, labels, trans_range=0.2):
        if not self._should_augment():
            return points, colors, labels

        translation = np.random.uniform(-trans_range, trans_range, 3)
        translated_points = points + translation
        return translated_points, colors, labels

    def random_axis_rotation(self, points, colors, labels):
        if not self._should_augment():
            return points, colors, labels

        axis = np.random.choice(['x', 'y', 'z'])
        angle = np.random.uniform(0, 2*np.pi)
        rotation = R.from_euler(axis, angle)

        rotated_points = rotation.apply(points)
        return rotated_points, colors, labels

    def random_flip(self, points, colors, labels, axes=[0, 1, 2]):
        if not self._should_augment():
            return points, colors, labels

        axis = np.random.choice(axes)
        flipped_points = points.copy()
        flipped_points[:, axis] *= -1
        return flipped_points, colors, labels

    # ===== COLOR AUGMENTATIONS =====
    def random_brightness(self, points, colors, labels, brightness_range=(0.7, 1.3)):
        """Adjust brightness dengan multiplier"""
        if not self._should_augment():
            return points, colors, labels

        brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
        aug_colors = np.clip(colors * brightness_factor, 0, 1)
        return points, aug_colors, labels

    def random_contrast(self, points, colors, labels, contrast_range=(0.8, 1.2)):
        """Adjust contrast dengan mean sebagai anchor"""
        if not self._should_augment():
            return points, colors, labels

        contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
        mean_color = np.mean(colors, axis=0, keepdims=True)
        aug_colors = (colors - mean_color) * contrast_factor + mean_color
        aug_colors = np.clip(aug_colors, 0, 1)
        return points, aug_colors, labels

    def random_saturation(self, points, colors, labels, saturation_range=(0.7, 1.3)):
        """Adjust saturation di HSV space"""
        if not self._should_augment():
            return points, colors, labels

        saturation_factor = np.random.uniform(saturation_range[0], saturation_range[1])
        hsv = rgb_to_hsv(colors)
        hsv[:, 1] = np.clip(hsv[:, 1] * saturation_factor, 0, 1)
        aug_colors = hsv_to_rgb(hsv)
        return points, aug_colors, labels

    def random_hue_shift(self, points, colors, labels, hue_range=(-0.1, 0.1)):
        """Shift hue di HSV space"""
        if not self._should_augment():
            return points, colors, labels

        hue_shift = np.random.uniform(hue_range[0], hue_range[1])
        hsv = rgb_to_hsv(colors)
        hsv[:, 0] = (hsv[:, 0] + hue_shift) % 1.0
        aug_colors = hsv_to_rgb(hsv)
        return points, aug_colors, labels

    def random_color_jitter(self, points, colors, labels, jitter_std=0.02):
        """Add random noise ke RGB channels"""
        if not self._should_augment():
            return points, colors, labels

        noise = np.random.normal(0, jitter_std, colors.shape)
        aug_colors = np.clip(colors + noise, 0, 1)
        return points, aug_colors, labels

    def random_gamma_correction(self, points, colors, labels, gamma_range=(0.8, 1.2)):
        """Apply gamma correction untuk simulate lighting changes"""
        if not self._should_augment():
            return points, colors, labels

        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        aug_colors = np.power(colors, gamma)
        aug_colors = np.clip(aug_colors, 0, 1)
        return points, aug_colors, labels

    def augment(self, points, colors, labels, augmentation_list=None):
        """Apply augmentasi geometri dan warna sesuai list"""
        if augmentation_list is None:
            augmentation_list = [
                # Geometric augmentations
                #('rotation', {}),
                #('flip', {'axes': [0, 1]}),
                #('scaling', {'scale_range': (0.85, 1.15)}),
                #('translation', {'trans_range': 0.1}),
                #('axis_rotation', {}),
                #('jitter', {'sigma': 0.01}),
                
                # Color augmentations (recommended to enable)
                #('brightness', {'brightness_range': (0.7, 1.3)}),
                #('contrast', {'contrast_range': (0.8, 1.2)}),
                #('saturation', {'saturation_range': (0.7, 1.3)}),
                #('hue_shift', {'hue_range': (-0.05, 0.05)}),
                #('color_jitter', {'jitter_std': 0.02}),
                #('gamma', {'gamma_range': (0.8, 1.2)}),
            ]

        aug_points = points.copy()
        aug_colors = colors.copy()
        aug_labels = labels.copy()

        for aug_name, aug_params in augmentation_list:
            # Geometric augmentations
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
            
            # Color augmentations
            elif aug_name == 'brightness':
                aug_points, aug_colors, aug_labels = self.random_brightness(
                    aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'contrast':
                aug_points, aug_colors, aug_labels = self.random_contrast(
                    aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'saturation':
                aug_points, aug_colors, aug_labels = self.random_saturation(
                    aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'hue_shift':
                aug_points, aug_colors, aug_labels = self.random_hue_shift(
                    aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'color_jitter':
                aug_points, aug_colors, aug_labels = self.random_color_jitter(
                    aug_points, aug_colors, aug_labels, **aug_params)
            elif aug_name == 'gamma':
                aug_points, aug_colors, aug_labels = self.random_gamma_correction(
                    aug_points, aug_colors, aug_labels, **aug_params)

        return aug_points, aug_colors, aug_labels

class Augmenter:
    def __init__(self):
        self.color_part_augmenter = ColorPartAugmenter()
        self.object_augmenter = ObjectAugmenter()
    
    def augment(self, points, colors, labels):
        # object aug
        points, colors, labels = self.object_augmenter.augment(points, colors, labels)
        
        # color aug
        colors = self.color_part_augmenter.augment(colors, labels)
        
        return points, colors, labels