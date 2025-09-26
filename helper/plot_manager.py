from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import logger

class PlotManager:
    def plot_category_distribution(self, categories_data, categories, index_to_category_id):
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

    def plot_data_samples(self, images_data, labels_data, categories_data, categories, index_to_category_id, num_samples=15):
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

    def plot_training_curves(self, train_losses, val_losses, val_interval):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
        if len(val_losses) > 0:
            # Val losses are recorded every n epochs
            val_epochs = [(i+1) * val_interval - 1 for i in range(len(val_losses))]
            plt.plot(val_epochs, val_losses, label='Validation Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if len(train_losses) > 10:
            window = max(1, min(10, len(train_losses)//5))
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

    def evaluate_and_plot_predictions(self, model, val_loader, device, categories, index_to_category_id, num_samples=8):
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

    def plot_predictions(self, image_orig, orig_size, mask_pred, title):
        # resize mask ke ukuran asli
        mask_resized = cv2.resize(mask_pred, orig_size)

        # plot
        plt.figure(figsize=(8,8))
        plt.imshow(image_orig)
        plt.imshow(mask_resized, cmap='jet', alpha=0.5)
        plt.title(title, fontsize=10)
        plt.axis('off')
        plt.show()