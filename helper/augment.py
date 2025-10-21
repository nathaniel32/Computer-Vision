import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import config

class ColorPartAugmenter: #tidak merusak norm
    """Augmentasi warna point cloud"""
    
    def __init__(self, target_label=config.TARGET_CLASS_ID, p_aug=0.7):
        self.target_label = target_label
        self.p_aug = p_aug
        self.color_augment_list = self.get_augment_list(hex_colors_list=["#FFFFFF", "#EBD5D5", "#B8B3B3", "#AA9D9D"])
    
    def _get_mask(self, labels):
        return labels == self.target_label
    
    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
    
    @staticmethod
    def _get_hue_from_hex(hex_color):
        rgb = ColorPartAugmenter.hex_to_rgb(hex_color)
        rgb = rgb.reshape(1, -1)
        hsv = rgb_to_hsv(rgb)
        return hsv[0, 0]
    
    def _apply_negative(self, colors, labels):
        colors_aug = colors.copy()
        mask = self._get_mask(labels)
        colors_aug[mask] = 1.0 - colors_aug[mask]
        return colors_aug
    
    def _apply_hue_shift(self, colors, labels, hue_value, use_negative=False):
        colors_aug = colors.copy()
        mask = self._get_mask(labels)
        
        if use_negative:
            colors_aug[mask] = 1.0 - colors_aug[mask]
        
        hsv = rgb_to_hsv(colors_aug[mask])
        hsv[:, 0] = hue_value % 1.0
        hsv[:, 1] = np.clip(hsv[:, 1] * 1.2, 0, 1)
        colors_aug[mask] = hsv_to_rgb(hsv)
        
        return colors_aug
        
    def original_to_color(self, colors, labels, hex_color):
        hue = self._get_hue_from_hex(hex_color)
        return self._apply_hue_shift(colors, labels, hue, use_negative=False)
    
    def negative(self, colors, labels):
        return self._apply_negative(colors, labels)
    
    def negative_to_color(self, colors, labels, hex_color):
        hue = self._get_hue_from_hex(hex_color)
        return self._apply_hue_shift(colors, labels, hue, use_negative=True)
    
    def get_augment_list(self, hex_colors_list):
        """Return list augmentasi dengan custom colors"""
        aug_list = [self.negative]
        
        for hex_color in hex_colors_list:
            aug_list.append(lambda c, l, hc=hex_color: self.negative_to_color(c, l, hc))
        
        for hex_color in hex_colors_list:
            aug_list.append(lambda c, l, hc=hex_color: self.original_to_color(c, l, hc))
        
        return aug_list
    
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
    """Augmentasi geometri point cloud"""
    
    def __init__(self, p_aug=0.7):
        self.p_aug = p_aug
    
    def _should_augment(self):
        return np.random.random() < self.p_aug

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

    def augment(self, points, colors, labels, augmentation_list=None):
        """Apply augmentasi geometri sesuai list"""
        if augmentation_list is None:
            augmentation_list = [
                ('rotation', {}),
                ('flip', {'axes': [0, 1]}),
                #('scaling', {'scale_range': (0.85, 1.15)}),
                #('translation', {'trans_range': 0.1}),
                #('axis_rotation', {}),
                #('jitter', {'sigma': 0.01})
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

class Augmenter:    
    def __init__(self):
        self.color_part_augmenter = ColorPartAugmenter()
        self.object_augmenter = ObjectAugmenter()
    
    def augment(self, points, colors, labels):
        # geometric aug
        #points, colors, labels = self.object_augmenter.augment(points, colors, labels)
        
        # color aug
        colors = self.color_part_augmenter.augment(colors, labels)
        
        return points, colors, labels