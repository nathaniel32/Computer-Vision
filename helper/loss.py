import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class FocalLoss(nn.Module):
    """
    Focal Loss untuk mengatasi class imbalance
    Lebih fokus pada sampel yang sulit (hard examples)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B*N, C) logits dari model
            targets: (B*N,) ground truth labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss untuk semantic segmentation
    Bagus untuk menangani class imbalance
    """
    def __init__(self, num_classes, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B*N, C) logits dari model
            targets: (B*N,) ground truth labels
        """
        inputs = F.softmax(inputs, dim=1)
        
        dice_loss = 0.0
        for c in range(self.num_classes):
            pred = inputs[:, c]
            target = (targets == c).float()
            
            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice_coeff
        
        dice_loss /= self.num_classes
        return dice_loss if self.reduction == 'mean' else dice_loss


class CombinedLoss(nn.Module):
    """
    Kombinasi CE Loss + Dice Loss + Focal Loss
    Menggabungkan kekuatan dari berbagai loss function
    """
    def __init__(self, num_classes, alpha_ce=0.5, alpha_dice=0.3, alpha_focal=0.2, 
                 focal_gamma=2.0, dice_smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.alpha_ce = alpha_ce
        self.alpha_dice = alpha_dice
        self.alpha_focal = alpha_focal
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes, smooth=dice_smooth)
        self.focal_loss = FocalLoss(gamma=focal_gamma)

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        
        total_loss = (self.alpha_ce * ce + 
                     self.alpha_dice * dice + 
                     self.alpha_focal * focal)
        return total_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss dengan class weights
    Untuk menangani class imbalance
    """
    def __init__(self, weights=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction=self.reduction)
        return ce_loss


class OhemCrossEntropyLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) Loss
    Fokus pada sampel yang paling sulit (hard negatives)
    """
    def __init__(self, num_classes, thresh=0.6, min_keep=100000):
        super(OhemCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.thresh = thresh
        self.min_keep = min_keep
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        loss = self.ce_loss(inputs, targets)
        
        # Sort losses
        sorted_loss, idx = torch.sort(loss, descending=True)
        
        # Tentukan threshold
        if sorted_loss[self.min_keep] > self.thresh:
            threshold = self.thresh
        else:
            threshold = sorted_loss[self.min_keep]
        
        # Keep hanya hard examples
        hard_mask = loss > threshold
        return loss[hard_mask].mean() if hard_mask.sum() > 0 else loss.mean()


class SmoothCrossEntropyLoss(nn.Module):
    """
    Label Smoothing + Cross Entropy Loss
    Mencegah overconfidence dan overfitting
    """
    def __init__(self, num_classes, smoothing=0.1):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

def compute_alpha(train_labels, num_classes, normalize=True):
    """
    Hitung alpha untuk Focal Loss dari dataset point cloud.

    Args:
        train_labels (list of list/array): 
            List dari label tiap point cloud, misal [[0,1,1,...], [0,2,1,...], ...]
        num_classes (int): Jumlah kelas (misal hand, body, head)
        normalize (bool): Apakah alpha dinormalisasi supaya sum = 1

    Returns:
        torch.Tensor: Alpha tensor untuk Focal Loss
    """
    # Gabungkan semua label jadi satu list
    all_labels = [label for pc_labels in train_labels for label in pc_labels]

    # Hitung frekuensi tiap kelas
    counts = Counter(all_labels)
    total_points = sum(counts.values())
    freq = {cls: counts.get(cls, 0)/total_points for cls in range(num_classes)}

    # Hitung alpha = inverse frequency
    alpha = {cls: 1/f if f > 0 else 0.0 for cls, f in freq.items()}

    # Normalisasi
    if normalize:
        sum_alpha = sum(alpha.values())
        if sum_alpha > 0:
            alpha = {cls: a/sum_alpha for cls, a in alpha.items()}

    # Ubah jadi tensor urut sesuai index kelas
    alpha_tensor = torch.tensor([alpha[i] for i in range(num_classes)], dtype=torch.float32)
    return alpha_tensor

# ===== REKOMENDASI PENGGUNAAN =====
"""
Pilih loss function berdasarkan kasus:

1. **Balanced Dataset** → CrossEntropyLoss (default)
   criterion = nn.CrossEntropyLoss()

2. **Imbalanced Dataset** → FocalLoss atau WeightedCrossEntropyLoss
   criterion = FocalLoss(gamma=2.0)
   # atau
   class_weights = torch.tensor([0.5, 1.0, 2.0])  # adjust sesuai distribusi
   criterion = WeightedCrossEntropyLoss(weights=class_weights)

3. **Best Performance** → CombinedLoss
   criterion = CombinedLoss(num_classes=len(config.CLASSES))

4. **Hard Negatives Mining** → OhemCrossEntropyLoss
   criterion = OhemCrossEntropyLoss(num_classes=len(config.CLASSES))

5. **Prevent Overfitting** → SmoothCrossEntropyLoss
   criterion = SmoothCrossEntropyLoss(num_classes=len(config.CLASSES), smoothing=0.1)
"""