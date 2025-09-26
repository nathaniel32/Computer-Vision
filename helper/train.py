import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from utils import logger

class DatasetManager(Dataset):
    def __init__(self, images_data, labels_data, categories_data, transform=None, augment=False):
        self.images_data = images_data
        self.labels_data = labels_data
        self.categories_data = categories_data
        self.transform = transform
        self.augment = augment
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        try:
            image = np.array(self.images_data[idx])
            mask = np.array(self.labels_data[idx])
            category = self.categories_data[idx]
            
            # augmentation (Horizontal flip to maintain mask correspondence)
            if self.augment and np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            # Normalize image
            image = image.astype(np.float32) / 255.0
            image = (image - self.mean) / self.std
            
            # Convert to tensors
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            category = torch.tensor(category, dtype=torch.long)

            return image, mask, category
        
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {e}")
            raise

class SegmentationLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=1.0, 
                 smooth=1e-6, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.class_weights = class_weights
        
    def focal_loss(self, pred, target):
        """Improved Focal Loss"""
        # Clamp predictions to prevent log(0)
        pred = torch.clamp(pred, min=self.smooth, max=1.0 - self.smooth)
        
        # Calculate BCE manually for better numerical stability
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        # Calculate pt
        pt = torch.where(target == 1, pred, 1 - pred)
        
        # Apply class weighting if provided
        alpha_t = self.alpha
        if self.class_weights is not None:
            alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        
        return focal_loss.mean()
    
    def dice_loss(self, pred, target):
        """Improved Dice Loss with better numerical stability"""
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        pred_sum = pred_flat.sum()
        target_sum = target_flat.sum()
        
        # Dice coefficient with smoothing
        dice_coeff = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        return 1 - dice_coeff
    
    def tversky_loss(self, pred, target, alpha_t=0.7, beta_t=0.3):
        """Tversky Loss - generalizes Dice loss, good for imbalanced data"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        true_pos = (pred_flat * target_flat).sum()
        false_neg = (target_flat * (1 - pred_flat)).sum()
        false_pos = ((1 - target_flat) * pred_flat).sum()
        
        tversky_coeff = (true_pos + self.smooth) / (true_pos + alpha_t * false_neg + beta_t * false_pos + self.smooth)
        
        return 1 - tversky_coeff
    
    def boundary_loss(self, pred, target):
        """Boundary-aware loss for better edge detection"""
        # Calculate gradients for boundary detection
        def gradient(x):
            # Sobel operators
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            
            if x.dim() == 4:
                grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
                grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
            else:
                grad_x = F.conv2d(x.unsqueeze(0), sobel_x, padding=1)
                grad_y = F.conv2d(x.unsqueeze(0), sobel_y, padding=1)
                grad_x = grad_x.squeeze(0)
                grad_y = grad_y.squeeze(0)
            
            return torch.sqrt(grad_x**2 + grad_y**2 + self.smooth)
        
        pred_grad = gradient(pred)
        target_grad = gradient(target)
        
        boundary_loss = F.mse_loss(pred_grad, target_grad)
        return boundary_loss
    
    def forward(self, pred, target, loss_type='combined'):
        """
        Forward pass with multiple loss options
        
        Args:
            pred: Predicted segmentation masks [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]
            loss_type: 'bce', 'focal', 'dice', 'tversky', 'combined', 'boundary_enhanced'
        """
        # Ensure same shape
        if pred.shape != target.shape:
            target = target.float()
        
        if loss_type == 'bce':
            return F.binary_cross_entropy(pred, target)
        
        elif loss_type == 'focal':
            return self.focal_loss(pred, target)
        
        elif loss_type == 'dice':
            return self.dice_loss(pred, target)
        
        elif loss_type == 'tversky':
            return self.tversky_loss(pred, target)
        
        elif loss_type == 'combined':
            bce_loss = F.binary_cross_entropy(pred, target, reduction='mean')
            focal_loss = self.focal_loss(pred, target)
            dice_loss = self.dice_loss(pred, target)
            
            return bce_loss + self.focal_weight * focal_loss + self.dice_weight * dice_loss
        
        elif loss_type == 'boundary_enhanced':
            # Enhanced version with boundary awareness
            bce_loss = F.binary_cross_entropy(pred, target, reduction='mean')
            focal_loss = self.focal_loss(pred, target)
            dice_loss = self.dice_loss(pred, target)
            boundary_loss = self.boundary_loss(pred, target)
            
            return (bce_loss + self.focal_weight * focal_loss + self.dice_weight * dice_loss + 0.1 * boundary_loss)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")