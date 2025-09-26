from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import logger
import config
import os

class PlotManager:
    def plot_data_samples(self, images_data, labels_data, num_samples=15):
        """Plot sample data for inspection"""
        logger.info("Plotting data samples...")
        
        for idx in range(min(num_samples, len(images_data))):
            image = np.array(images_data[idx])
            mask = labels_data[idx]

            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(mask)
            plt.title(f"Ground Truth Mask")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            # Overlay mask on image
            plt.imshow(image)
            plt.imshow(mask, cmap="tab20", alpha=0.5)
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            logger.info(f"- Mask pixels: {np.sum(mask > 0)}/{mask.size} ({100*np.sum(mask > 0)/mask.size:.1f}%)")
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

    def evaluate_and_plot_predictions(self, model, val_loader, device, categories_classes, num_samples=8):
        """Evaluate model and plot multi-class predictions with metrics"""
        model.eval()
        
        total_samples = 0
        total_iou_per_class = defaultdict(float)
        total_dice_per_class = defaultdict(float)
        
        sample_count = 0
        num_classes = len(categories_classes)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)  # shape (B, H, W)
                
                outputs = model(images)  # shape (B, C, H, W)
                
                # Process each sample
                for i in range(images.size(0)):
                    if sample_count >= num_samples:
                        break
                    
                    image = images[i].cpu()
                    mask_true = masks[i].squeeze().cpu().numpy()   # (H, W)
                    mask_pred = outputs[i].cpu().numpy()          # (C, H, W)
                    mask_pred_class = np.argmax(mask_pred, axis=0)  # (H, W)

                    np.savetxt(os.path.join(config.LOG_DIR, f"mask_pred_{i}.txt"), mask_pred_class, fmt="%d")
                    
                    # Denormalize image
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                    image_denorm = (image * std + mean).clamp(0, 1)
                    image_np = image_denorm.permute(1, 2, 0).numpy()
                    
                    # Metrics per class
                    ious = []
                    dices = []

                    logger.info("=" * 100)
                    for c in range(num_classes):
                        pred_c = (mask_pred_class == c).astype(np.float32)
                        true_c = (mask_true == c).astype(np.float32)
                        
                        intersection = np.sum(pred_c * true_c)
                        union = np.sum(pred_c) + np.sum(true_c) - intersection
                        iou = intersection / (union + 1e-8)
                        dice = (2 * intersection + 1e-8) / (np.sum(pred_c) + np.sum(true_c) + 1e-8)
                        
                        ious.append(iou)
                        dices.append(dice)
                        total_iou_per_class[c] += iou
                        total_dice_per_class[c] += dice

                        logger.info(f"- Class {c} ({categories_classes.get(c)}):")
                        logger.info(f"- IoU: {iou}")
                        logger.info(f"- Dice: {dice}")
                        logger.info("-" * 50)
                    
                    accuracy = np.mean(mask_pred_class == mask_true)
                    logger.info(f"- Accuracy: {accuracy}")

                    # Plot
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(1, 3, 1)
                    plt.imshow(image_np)
                    plt.title(f"Input Image", fontsize=10)
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.imshow(mask_true, cmap='tab20', vmin=0, vmax=num_classes-1)
                    plt.title("Ground Truth", fontsize=10)
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.imshow(mask_pred_class, cmap='tab20', vmin=0, vmax=num_classes-1)
                    plt.title("Prediction", fontsize=10)
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    sample_count += 1
                    total_samples += 1
        
        # Overall summary
        logger.info("="*60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*60)
        
        if total_samples > 0:
            logger.info(f"Overall Performance ({total_samples} samples):")
            for c in range(num_classes):
                logger.info(f"  Class {c} ({categories_classes.get(c)}):")
                logger.info(f"    Average IoU: {total_iou_per_class[c]/total_samples:.3f}")
                logger.info(f"    Average Dice: {total_dice_per_class[c]/total_samples:.3f}")

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