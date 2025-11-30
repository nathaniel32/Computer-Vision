import torch
import torch.optim as optim
import os
import configs
from helper.utils.log import Logging
from helper.train.dataset import PointCloudSegmentationDataset, load_pcd_with_point_labels
from torch.utils.data import DataLoader
from helper.train.model import PointNetSegmentation, get_predict_model
from helper.utils.plot import plot_training_stats, plot_point_cloud, PlotTool
from helper.train.loss import FocalLoss, compute_alpha
from helper.train.scheduler import WarmupScheduler
from helper.train.augment import Augmenter

class TrainModel:
    def __init__(self, logger:Logging, config:configs.BaseConfig):
        self.logger = logger
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _train(self, model, loader, criterion, optimizer, loop=2):
        model.train()
        
        avg_loss_all = 0
        accuracy_all = 0
        
        for i in range(1, loop + 1):
            total_loss = 0
            correct = 0
            total = 0
            
            for points, colors, labels in loader:
                points, colors, labels = points.to(self.device), colors.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                outputs = model(points, colors)
                
                outputs_flat = outputs.reshape(-1, outputs.shape[-1])
                labels_flat = labels.reshape(-1)
                
                loss = criterion(outputs_flat, labels_flat)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = outputs_flat.argmax(dim=1)
                correct += (predictions == labels_flat).sum().item()
                total += labels_flat.size(0)
            
            epoch_loss = total_loss / len(loader)
            epoch_acc = 100.0 * correct / total
            
            avg_loss_all += epoch_loss
            accuracy_all += epoch_acc
            
            print(f"Epoch {i}/{loop} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
        
        avg_loss = avg_loss_all / loop
        accuracy = accuracy_all / loop
        
        return avg_loss, accuracy
    
    def _eval(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for points, colors, labels in loader:
                points, colors, labels = points.to(self.device), colors.to(self.device), labels.to(self.device)
                outputs = model(points, colors)
                
                # Reshape for loss calculation
                outputs_flat = outputs.reshape(-1, outputs.shape[-1])
                labels_flat = labels.reshape(-1)
                
                loss = criterion(outputs_flat, labels_flat)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = outputs_flat.argmax(dim=1)
                correct += (predictions == labels_flat).sum().item()
                total += labels_flat.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def test(self):
        model, model_num_points, model_classes = get_predict_model(self.config.save_model_path, self.device)

        TEST_DIR = os.path.join(self.config.ds_root, "test")
        test_point_clouds, test_colors, test_labels = load_pcd_with_point_labels(TEST_DIR, rand_sampling_min_num=model_num_points)
        test_dataset = PointCloudSegmentationDataset(test_point_clouds, test_colors, labels=test_labels)

        model.eval()
        with torch.no_grad():
            for (point, color, label) in test_dataset:
                point = point.unsqueeze(0).to(self.device)
                color = color.unsqueeze(0).to(self.device)
                outputs = model(point, color)
                
                point_plot = point.squeeze(0).transpose(0, 1).cpu().numpy()
                color_plot = color.squeeze(0).transpose(0, 1).cpu().numpy()
                pred_label = outputs.squeeze(0).argmax(dim=1).cpu().numpy()
                
                plot_point_cloud(point_plot, color_plot, model_classes, pred_label=pred_label, true_label=label, plot_tool=PlotTool.OPEN3D)
    
    def train(self, val_interval=1, warmup_epochs=5, resume=False):
        if not resume:
            self.logger.clear()
        
        TRAIN_DIR = os.path.join(self.config.ds_root, "train")
        train_point_clouds, train_colors, train_labels = load_pcd_with_point_labels(TRAIN_DIR)
        self.logger.info(f"\nTrain samples: {len(train_point_clouds)}")
        augmenter = Augmenter(config=self.config)
        train_dataset = PointCloudSegmentationDataset(train_point_clouds, train_colors, labels=train_labels, augmenter=augmenter, rand_sampling_min_num=self.config.model_num_points)
        
        VAL_DIR = os.path.join(self.config.ds_root, "val")
        val_point_clouds, val_colors, val_labels = load_pcd_with_point_labels(VAL_DIR, rand_sampling_min_num=self.config.model_num_points)
        self.logger.info(f"Validation samples: {len(val_point_clouds)}")
        val_dataset = PointCloudSegmentationDataset(val_point_clouds, val_colors, labels=val_labels)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)

        self.logger.info(f"\nTrain batches: {len(train_loader)}")
        self.logger.info(f"Validation batches: {len(val_loader)}")

        num_classes = len(self.config.classes)
        model = PointNetSegmentation(num_classes=num_classes).to(self.device)

        alpha = compute_alpha(train_labels=train_labels, num_classes=num_classes).to(self.device)
        criterion = FocalLoss(alpha=alpha)
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        
        # Initialize warm-up scheduler
        warmup_scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            initial_lr=self.config.lr * 0.1,
            target_lr=self.config.lr
        )
        
        # Main scheduler
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
        )
        
        # Initialize training state
        start_epoch = 0
        patience_counter = 0
        best_val_acc = 0.0
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        
        # Resume from checkpoint
        if resume:
            if not os.path.exists(self.config.save_model_path):
                raise FileNotFoundError(f"Checkpoint not found: {self.config.save_model_path}")
            
            self.logger.info(f"\n=== Resuming from checkpoint ===")
            checkpoint = torch.load(self.config.save_model_path, map_location=self.device)
            
            # Load all required states (error if missing)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            main_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['val_acc']
            patience_counter = checkpoint['patience_counter']
            
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            train_accuracies = checkpoint['train_accuracies']
            val_accuracies = checkpoint['val_accuracies']
            
            self.logger.info(f"Resumed from epoch {start_epoch}")
            self.logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
            self.logger.info(f"Patience counter: {patience_counter}")
        else:
            self.logger.info(f"\n=== Training with {warmup_epochs} epochs warm-up ===")
            self.logger.info(f"Initial LR: {self.config.lr * 0.1:.2e} -> Target LR: {self.config.lr:.2e}")
        
        for epoch in range(start_epoch, self.config.epochs):
            self.logger.info(f'\nEpoch {epoch+1}/{self.config.epochs}')
            self.logger.info('-' * 60)
            
            # Apply warm-up
            if not resume and epoch < warmup_epochs:
                warmup_scheduler.step()
                current_lr = warmup_scheduler.get_lr()
                self.logger.info(f'Warm-up LR: {current_lr:.2e}')
            
            train_loss, train_acc = self._train(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._eval(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            self.logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            self.logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # After warm-up period
            if resume or epoch >= warmup_epochs:
                # Save best model
                if val_acc > best_val_acc:
                    patience_counter = 0
                    best_val_acc = val_acc
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': main_scheduler.state_dict(),
                        'epoch': epoch,
                        'val_acc': val_acc,
                        'classes': self.config.classes,
                        'model_num_points': self.config.model_num_points,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_accuracies': train_accuracies,
                        'val_accuracies': val_accuracies,
                        'patience_counter': patience_counter,
                        'best_val_acc': best_val_acc
                    }, self.config.save_model_path)
                    self.logger.info(f'Saved best model with validation accuracy: {val_acc:.2f}%')
                else:
                    patience_counter += 1
                    self.logger.info(f"- Patience: {patience_counter}/{self.config.patience}")

                    if patience_counter >= self.config.patience:
                        self.logger.info("= Early stopping triggered!")
                        break
            
                # Apply main scheduler
                main_scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f'Current LR: {current_lr:.2e}')

        self.logger.info(f"\n=== Training Complete ===")
        self.logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")

        plot_training_stats(train_losses, val_losses, train_accuracies, val_accuracies)