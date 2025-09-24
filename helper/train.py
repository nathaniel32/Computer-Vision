import json
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from utils import logger
import os
from torch.utils.data import DataLoader
import json
import joblib

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

def _load_coco_data(dir_root, dir_name, file_name='_annotations.coco.json'):
    """Safely load COCO data with error handling"""
    try:
        full_path = os.path.join(dir_root, dir_name, file_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {full_path}")
        
        with open(full_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
        
        # Validate required keys
        required_keys = ['categories', 'annotations', 'images']
        for key in required_keys:
            if key not in coco_data:
                raise ValueError(f"Missing required key in COCO data: {key}")
        
        logger.info(f"Successfully loaded COCO data with {len(coco_data['images'])} images")
        return coco_data
    
    except Exception as e:
        logger.error(f"Error loading COCO data: {e}")
        raise

def _create_category_mapping(coco_data):
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Get all category IDs actually used in annotations
    annotation_category_ids = [ann['category_id'] for ann in coco_data['annotations']]
    unique_category_ids = sorted(list(set(annotation_category_ids)))
    
    # Validate all annotation categories exist in categories dict
    missing_categories = [cat_id for cat_id in unique_category_ids if cat_id not in categories]
    if missing_categories:
        logger.warning(f"Found annotations with missing category definitions: {missing_categories}")
    
    # Create continuous mapping starting from 0
    category_id_to_index = {cat_id: idx for idx, cat_id in enumerate(unique_category_ids)}
    index_to_category_id = {idx: cat_id for cat_id, idx in category_id_to_index.items()}
    
    # Check for class imbalance
    category_counts = defaultdict(int)
    for cat_id in annotation_category_ids:
        category_counts[cat_id] += 1
    
    logger.info(f"Category distribution: {dict(category_counts)}")
    
    # Warn about severe class imbalance
    max_count = max(category_counts.values())
    min_count = min(category_counts.values())
    if max_count / min_count > 10:
        logger.warning(f"Severe class imbalance detected. Ratio: {max_count/min_count:.2f}")
    
    return categories, category_id_to_index, index_to_category_id

def _process_segmentation(seg, image_height, image_width):
    """process segmentation data"""
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    try:
        if isinstance(seg, list) and len(seg) > 0:
            if isinstance(seg[0], list):
                # Multiple polygons
                for polygon in seg:
                    if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                        polygon_array = np.array(polygon).reshape((-1, 2)).astype(np.int32)
                        # Clip coordinates to image bounds
                        polygon_array[:, 0] = np.clip(polygon_array[:, 0], 0, image_width-1)
                        polygon_array[:, 1] = np.clip(polygon_array[:, 1], 0, image_height-1)
                        cv2.fillPoly(mask, [polygon_array], color=1)
            else:
                # Single polygon
                if len(seg) >= 6:  # At least 3 points
                    seg_array = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    # Clip coordinates to image bounds
                    seg_array[:, 0] = np.clip(seg_array[:, 0], 0, image_width-1)
                    seg_array[:, 1] = np.clip(seg_array[:, 1], 0, image_height-1)
                    cv2.fillPoly(mask, [seg_array], color=1)
        
        return mask
    
    except Exception as e:
        logger.warning(f"Error processing segmentation: {e}")
        return mask  # Return empty mask on error

def _create_dataset(coco_data, dir_root, dir_name, category_id_to_index, target_size):
    # First, reorganize COCO data by image
    coco_dataset = {}
    for ann in coco_data['annotations']:
        image_id = ann["image_id"]
        cat_id = ann["category_id"]
        seg = ann["segmentation"]

        if image_id not in coco_dataset:
            try:
                image_info = next(i for i in coco_data['images'] if i['id'] == image_id)
                coco_dataset[image_id] = {
                    "image_id": image_id,
                    "image_name": image_info['file_name'],
                    "image_width": image_info['width'],
                    "image_height": image_info['height'],
                    "categories": []
                }
            except StopIteration:
                logger.warning(f"Image info not found for image_id: {image_id}")
                continue

        # Find or create category entry
        found = None
        for cat in coco_dataset[image_id]["categories"]:
            if cat["category_id"] == cat_id:
                found = cat
                break

        if found:
            found["segmentations"].append(seg)
        else:
            coco_dataset[image_id]["categories"].append({
                "category_id": cat_id,
                "segmentations": [seg]
            })

    # Convert to list and process images
    coco_dataset = list(coco_dataset.values())
    
    images_data = []
    categories_data = []
    labels_data = []
    failed_loads = 0

    for image_data in coco_dataset:
        img_path = os.path.join(dir_root, dir_name, image_data['image_name'])
        
        try:
            # Validate image file exists
            if not os.path.exists(img_path):
                logger.warning(f"Image file not found: {img_path}")
                failed_loads += 1
                continue
            
            # Load and validate image
            image_pil = Image.open(img_path).convert('RGB')
            
            # Check if image is corrupted
            if image_pil.size[0] == 0 or image_pil.size[1] == 0:
                logger.warning(f"Invalid image dimensions: {img_path}")
                failed_loads += 1
                continue
            
            image_pil = image_pil.resize(target_size)
            
            for cat in image_data['categories']:
                category_id = cat['category_id']
                
                # Skip if category not in mapping
                if category_id not in category_id_to_index:
                    logger.warning(f"Category ID {category_id} not in mapping")
                    continue
                
                category_index = category_id_to_index[category_id]
                
                # Process mask with error handling
                mask = np.zeros((image_data['image_height'], image_data['image_width']), dtype=np.uint8)
                valid_segmentations = 0
                
                for seg in cat['segmentations']:
                    seg_mask = _process_segmentation(
                        seg, image_data['image_height'], image_data['image_width']
                    )
                    if np.sum(seg_mask) > 0:  # Only add if mask is not empty
                        mask = np.logical_or(mask, seg_mask).astype(np.uint8)
                        valid_segmentations += 1
                
                # Skip if no valid segmentations
                if valid_segmentations == 0:
                    logger.warning(f"No valid segmentations for image {image_data['image_name']}, category {category_id}")
                    continue
                
                # Resize mask
                mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                
                # Validate mask has some positive pixels
                if np.sum(mask) == 0:
                    logger.warning(f"Empty mask after resize for {image_data['image_name']}")
                    continue

                images_data.append(image_pil)
                categories_data.append(category_index)
                labels_data.append(mask)
        
        except Exception as e:
            logger.warning(f"Error processing image {img_path}: {e}")
            failed_loads += 1
            continue

    logger.info(f"Dataset created: {len(images_data)} samples, {failed_loads} failed loads")
    return images_data, labels_data, categories_data

def prepare_datasets(ds_root, image_size, log_dir, save_meta_path):
        ds_name = "train"
        logger.info("Loading COCO data...")
        coco_data_train = _load_coco_data(ds_root, ds_name)
        logger.info("Creating category mappings...")
        categories_classes, category_encoder, category_decoder = _create_category_mapping(coco_data_train)
        logger.info("Creating dataset...")
        X_train, Y_train, Cat_train = _create_dataset(
            coco_data_train, ds_root, ds_name, category_encoder, image_size
        )

        ds_name = "valid"
        logger.info("Loading COCO data...")
        coco_data_valid = _load_coco_data(ds_root, ds_name)
        logger.info("Creating dataset...")
        X_valid, Y_valid, Cat_valid = _create_dataset(
            coco_data_valid, ds_root, ds_name, category_encoder, image_size
        )

        ds_name = "test"
        logger.info("Loading COCO data...")
        coco_data_test = _load_coco_data(ds_root, ds_name)
        logger.info("Creating dataset...")
        X_test, Y_test, Cat_test = _create_dataset(
            coco_data_test, ds_root, ds_name, category_encoder, image_size
        )

        if len(X_train) == 0:
            raise ValueError("No Data!")
        
        # save json
        output_path = os.path.join(log_dir, "annotations_train.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data_train, f, indent=4)

        output_path = os.path.join(log_dir, "annotations_valid.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data_valid, f, indent=4)

        output_path = os.path.join(log_dir, "annotations_test.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data_test, f, indent=4)
        
        # Plot category distribution
        plot_category_distribution(Cat_train, categories_classes, category_decoder)
        
        # Plot sample data
        plot_data_samples(X_train, Y_train, Cat_train, categories_classes, category_decoder, num_samples=1)

        train_dataset = DatasetManager(X_train, Y_train, Cat_train, augment=True)
        val_dataset = DatasetManager(X_valid, Y_valid, Cat_valid, augment=False)
        test_dataset = DatasetManager(X_test, Y_test, Cat_test, augment=False)

        logger.info("Train: ", len(train_dataset))
        logger.info("Val: ", len(val_dataset))
        logger.info("Test: ", len(test_dataset))
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

        # save meta
        meta_data = {
            "categories_classes": categories_classes,
            "category_encoder": category_encoder,
            'category_decoder': category_decoder
        }
        joblib.dump(meta_data, save_meta_path)

        return train_loader, val_loader, test_loader, categories_classes, category_decoder

def plot_category_distribution(categories_data, categories, index_to_category_id):
    """Plot distribution of categories in dataset"""
    category_counts = defaultdict(int)
    for cat_idx in categories_data:
        original_id = index_to_category_id[cat_idx]
        cat_name = categories.get(original_id, f"ID_{original_id}")
        category_counts[cat_name] += 1
    
    plt.figure(figsize=(12, 6))
    names = list(category_counts.keys())
    counts = list(category_counts.values())
    
    bars = plt.bar(range(len(names)), counts)
    plt.xlabel('Category')
    plt.ylabel('Number of Samples')
    plt.title('Category Distribution in Dataset')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    # statistics
    logger.info(f"Dataset Statistics:")
    logger.info(f"  Total samples: {sum(counts)}")
    logger.info(f"  Number of categories: {len(names)}")
    logger.info(f"  Most frequent: {max(category_counts, key=category_counts.get)} ({max(counts)} samples)")
    logger.info(f"  Least frequent: {min(category_counts, key=category_counts.get)} ({min(counts)} samples)")
    logger.info(f"  Imbalance ratio: {max(counts)/min(counts):.2f}")

def plot_data_samples(images_data, labels_data, categories_data, categories, index_to_category_id, num_samples=15):
    """Plot sample data for inspection"""
    logger.info("Plotting data samples...")
    
    for idx in range(min(num_samples, len(images_data))):
        image = np.array(images_data[idx])
        mask = labels_data[idx]
        category_index = categories_data[idx]
        
        original_category_id = index_to_category_id[category_index]
        category_name = categories.get(original_category_id, "Unknown")

        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Ground Truth Mask")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Overlay mask on image
        overlay = image.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red overlay
        plt.imshow(overlay)
        plt.title(f"Category: {category_name}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Sample {idx+1}: {category_name} (index: {category_index})")
        logger.info(f"  Mask pixels: {np.sum(mask > 0)}/{mask.size} ({100*np.sum(mask > 0)/mask.size:.1f}%)")
        logger.info()

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    if val_losses:
        # Val losses are recorded every 5 epochs
        val_epochs = [(i+1)*5-1 for i in range(len(val_losses))]
        plt.plot(val_epochs, val_losses, label='Validation Loss', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if len(train_losses) > 10:
        # Plot smoothed version for better visualization
        window = min(10, len(train_losses)//5)
        smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(train_losses)), smoothed, 
                label='Smoothed Training Loss', color='darkblue')
    plt.plot(train_losses, alpha=0.3, color='lightblue', label='Raw Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def evaluate_and_plot_predictions(model, val_loader, device, categories, index_to_category_id, num_samples=8):
    """Evaluate model and plot predictions with metrics"""
    model.eval()
    
    # Metrics tracking
    total_samples = 0
    total_iou = 0
    total_dice = 0
    category_metrics = defaultdict(list)
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks, categories_tensor) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            categories_tensor = categories_tensor.to(device)
            
            outputs = model(images, categories_tensor)
            
            # Process each sample in batch
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break
                
                # Get data for this sample
                image = images[i].cpu()
                mask_true = masks[i].squeeze().cpu().numpy()
                mask_pred = outputs[i].squeeze().cpu().numpy()
                category_idx = categories_tensor[i].item()
                
                # Denormalize image for visualization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                image_denorm = (image * std + mean).clamp(0, 1)
                image_np = image_denorm.permute(1, 2, 0).numpy()
                
                # Get category name
                original_category_id = index_to_category_id[category_idx]
                category_name = categories.get(original_category_id, "Unknown")
                
                # Calculate metrics
                pred_binary = (mask_pred > 0.5).astype(np.float32)
                
                # IoU calculation
                intersection = np.sum(mask_true * pred_binary)
                union = np.sum(mask_true) + np.sum(pred_binary) - intersection
                iou = intersection / (union + 1e-8)
                
                # Dice coefficient
                dice = (2 * intersection + 1e-8) / (np.sum(mask_true) + np.sum(pred_binary) + 1e-8)
                
                # Accuracy
                accuracy = np.sum((mask_true > 0.5) == (pred_binary > 0.5)) / mask_true.size
                
                # Track metrics
                total_iou += iou
                total_dice += dice
                total_samples += 1
                category_metrics[category_name].append({
                    'iou': iou,
                    'dice': dice,
                    'accuracy': accuracy
                })
                
                # Plot results
                plt.figure(figsize=(15, 5))
                
                # Original image
                plt.subplot(1, 5, 1)
                plt.imshow(image_np)
                plt.title(f"Input Image\nCategory: {category_name}", fontsize=10)
                plt.axis('off')
                
                # Ground truth
                plt.subplot(1, 5, 2)
                plt.imshow(mask_true, cmap='gray')
                plt.title("Ground Truth", fontsize=10)
                plt.axis('off')
                
                # Prediction (continuous)
                plt.subplot(1, 5, 3)
                plt.imshow(mask_pred, cmap='gray')
                plt.title(f"Prediction\n(continuous)", fontsize=10)
                plt.axis('off')
                
                # Prediction (binary)
                plt.subplot(1, 5, 4)
                plt.imshow(pred_binary, cmap='gray')
                plt.title(f"Prediction\n(binary @ 0.5)", fontsize=10)
                plt.axis('off')
                
                # Overlay comparison
                plt.subplot(1, 5, 5)
                overlay = image_np.copy()
                # True positive: Green, False positive: Red, False negative: Blue
                tp = (mask_true > 0.5) & (pred_binary > 0.5)
                fp = (mask_true <= 0.5) & (pred_binary > 0.5)
                fn = (mask_true > 0.5) & (pred_binary <= 0.5)
                
                overlay[tp] = [0, 1, 0]  # Green
                overlay[fp] = [1, 0, 0]  # Red
                overlay[fn] = [0, 0, 1]  # Blue
                
                plt.imshow(overlay)
                plt.title(f"TP(G) FP(R) FN(B)\nIoU: {iou:.3f}\nDice: {dice:.3f}", fontsize=10)
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
                
                # detailed metrics
                logger.info(f"Sample {sample_count + 1}: {category_name}")
                logger.info(f"  IoU: {iou:.3f}")
                logger.info(f"  Dice: {dice:.3f}")
                logger.info(f"  Accuracy: {accuracy:.3f}")
                logger.info(f"  Pred range: [{mask_pred.min():.3f}, {mask_pred.max():.3f}]")
                logger.info(f"  Pred mean: {mask_pred.mean():.3f}")
                logger.info()
                
                sample_count += 1
                
            if sample_count >= num_samples:
                break
    
    # overall performance summary
    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    
    if total_samples > 0:
        logger.info(f"Overall Performance ({total_samples} samples):")
        logger.info(f"  Average IoU: {total_iou/total_samples:.3f}")
        logger.info(f"  Average Dice: {total_dice/total_samples:.3f}")
        logger.info()
        
        logger.info("Per-Category Performance:")
        for cat_name, metrics_list in category_metrics.items():
            ious = [m['iou'] for m in metrics_list]
            dices = [m['dice'] for m in metrics_list]
            accuracies = [m['accuracy'] for m in metrics_list]
            
            logger.info(f"  {cat_name} (n={len(metrics_list)}):")
            logger.info(f"    IoU: {np.mean(ious):.3f} ± {np.std(ious):.3f}")
            logger.info(f"    Dice: {np.mean(dices):.3f} ± {np.std(dices):.3f}")
            logger.info(f"    Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        logger.info()
    
    return category_metrics