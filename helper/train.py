import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from utils import logger

class DatasetManager(Dataset):
    def __init__(self, images_data, labels_data, transform=None, augment=False):
        self.images_data = images_data
        self.labels_data = labels_data
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

            return image, mask
        
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {e}")
            raise

class SegmentationLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=1.0, 
                 smooth=1e-6, class_weights=None, ignore_index=-100):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        
        # Register class weights as buffer if provided
        if class_weights is not None:
            self.register_buffer('weight', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None
    
    def focal_loss(self, pred, target):
        """Multi-class Focal Loss"""
        # pred: [B, C, H, W] - logits
        # target: [B, H, W] - class indices
        
        # Fix target shape if needed
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        target = target.long()
        
        # Convert to probabilities
        pred_probs = F.softmax(pred, dim=1)
        
        # Create one-hot encoding for target
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Clamp predictions for numerical stability
        pred_probs = torch.clamp(pred_probs, min=self.smooth, max=1.0 - self.smooth)
        
        # Calculate cross entropy manually
        ce_loss = -target_one_hot * torch.log(pred_probs)
        
        # Calculate pt (probability of true class)
        pt = (pred_probs * target_one_hot).sum(dim=1)
        
        # Apply alpha weighting
        if self.weight is not None:
            alpha_t = self.weight[target]
        else:
            alpha_t = self.alpha
        
        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss.sum(dim=1)
        
        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            focal_loss = focal_loss * mask
            return focal_loss.sum() / mask.sum().clamp(min=1)
        
        return focal_loss.mean()
    
    def dice_loss(self, pred, target):
        """Multi-class Dice Loss"""
        # Fix target shape if needed
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        target = target.long()
        
        # Convert logits to probabilities
        pred_probs = F.softmax(pred, dim=1)
        
        # Create one-hot encoding for target
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        dice_losses = []
        
        for class_idx in range(self.num_classes):
            pred_class = pred_probs[:, class_idx].contiguous().view(-1)
            target_class = target_one_hot[:, class_idx].contiguous().view(-1)
            
            # Skip background class (index 0) optionally
            # if class_idx == 0:
            #     continue
            
            intersection = (pred_class * target_class).sum()
            pred_sum = pred_class.sum()
            target_sum = target_class.sum()
            
            # Skip if no target pixels for this class
            if target_sum == 0:
                #print("Empty: ", class_idx) # class yg tidak ada di foto ini
                if pred_sum == 0:
                    #dice_coeff = 1.0  # Perfect prediction
                    dice_coeff = torch.tensor(1.0, device=pred_probs.device)
                else:
                    #dice_coeff = 0.0  # False positives
                    dice_coeff = torch.tensor(0.0, device=pred_probs.device)
            else:
                dice_coeff = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            
            dice_losses.append(1 - dice_coeff)
        
        return torch.stack(dice_losses).mean()
    
    def tversky_loss(self, pred, target, alpha_t=0.7, beta_t=0.3):
        """Multi-class Tversky Loss"""
        # Fix target shape if needed
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        target = target.long()
        
        pred_probs = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        tversky_losses = []
        
        for class_idx in range(self.num_classes):
            pred_class = pred_probs[:, class_idx].contiguous().view(-1)
            target_class = target_one_hot[:, class_idx].contiguous().view(-1)
            
            true_pos = (pred_class * target_class).sum()
            false_neg = (target_class * (1 - pred_class)).sum()
            false_pos = ((1 - target_class) * pred_class).sum()
            
            tversky_coeff = (true_pos + self.smooth) / (true_pos + alpha_t * false_neg + beta_t * false_pos + self.smooth)
            tversky_losses.append(1 - tversky_coeff)
        
        return torch.stack(tversky_losses).mean()
    
    def boundary_loss(self, pred, target):
        """Multi-class Boundary-aware loss"""
        # Fix target shape if needed
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        target = target.long()
        
        pred_probs = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        def gradient(x):
            # Sobel operators
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            
            # Handle multi-channel input
            if x.size(1) > 1:
                grad_x = F.conv2d(x, sobel_x.repeat(x.size(1), 1, 1, 1), 
                                padding=1, groups=x.size(1))
                grad_y = F.conv2d(x, sobel_y.repeat(x.size(1), 1, 1, 1), 
                                padding=1, groups=x.size(1))
            else:
                grad_x = F.conv2d(x, sobel_x, padding=1)
                grad_y = F.conv2d(x, sobel_y, padding=1)
            
            return torch.sqrt(grad_x**2 + grad_y**2 + self.smooth)
        
        pred_grad = gradient(pred_probs)
        target_grad = gradient(target_one_hot)
        
        return F.mse_loss(pred_grad, target_grad)
    
    def lovasz_softmax_loss(self, pred, target):
        """Lovász-Softmax loss for multi-class segmentation"""
        # Fix target shape if needed
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        target = target.long()
        
        def lovasz_grad(gt_sorted):
            """Compute gradient of the Lovász extension w.r.t the sorted errors"""
            p = len(gt_sorted)
            gts = gt_sorted.sum()
            if p == 0:
                return torch.tensor(0.0, device=gt_sorted.device)
            intersection = gts.float() - gt_sorted.float().cumsum(0)
            union = gts.float() + (1 - gt_sorted).float().cumsum(0)
            jaccard = 1. - intersection / union
            if p > 1:  # cover 1-pixel case
                jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
            return jaccard
        
        pred_probs = F.softmax(pred, dim=1)
        losses = []
        
        for class_idx in range(self.num_classes):
            fg = (target == class_idx).float()
            if fg.sum() == 0:
                continue
                
            errors = (fg - pred_probs[:, class_idx]).abs()
            errors_sorted, perm = torch.sort(errors.view(-1), descending=True)
            fg_sorted = fg.view(-1)[perm]
            grad = lovasz_grad(fg_sorted)
            loss = torch.dot(errors_sorted, grad)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return torch.stack(losses).mean()
    
    def forward(self, pred, target, loss_type='combined'):
        """
        Forward pass with multiple loss options
        
        Args:
            pred: Predicted segmentation logits [B, C, H, W]
            target: Ground truth class indices [B, H, W] or [B, 1, H, W]
            loss_type: 'ce', 'focal', 'dice', 'tversky', 'combined', 'boundary_enhanced', 'lovasz'
        """
        # Fix target shape - remove channel dimension if present
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        
        # Ensure target is long type for cross-entropy
        target = target.long()
        
        # Handle ignore_index in target
        if self.ignore_index >= 0:
            valid_mask = target != self.ignore_index
            if not valid_mask.any():
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        if loss_type == 'ce':
            return F.cross_entropy(pred, target, weight=self.weight, ignore_index=self.ignore_index)
        
        elif loss_type == 'focal':
            return self.focal_loss(pred, target)
        
        elif loss_type == 'dice':
            return self.dice_loss(pred, target)
        
        elif loss_type == 'tversky':
            return self.tversky_loss(pred, target)
        
        elif loss_type == 'lovasz':
            return self.lovasz_softmax_loss(pred, target)
        
        elif loss_type == 'combined':
            ce_loss = F.cross_entropy(pred, target, weight=self.weight, ignore_index=self.ignore_index)
            focal_loss = self.focal_loss(pred, target)
            dice_loss = self.dice_loss(pred, target)
            
            return ce_loss + self.focal_weight * focal_loss + self.dice_weight * dice_loss
        
        elif loss_type == 'boundary_enhanced':
            ce_loss = F.cross_entropy(pred, target, weight=self.weight, ignore_index=self.ignore_index)
            focal_loss = self.focal_loss(pred, target)
            dice_loss = self.dice_loss(pred, target)
            boundary_loss = self.boundary_loss(pred, target)
            
            return (ce_loss + self.focal_weight * focal_loss + 
                   self.dice_weight * dice_loss + 0.1 * boundary_loss)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")