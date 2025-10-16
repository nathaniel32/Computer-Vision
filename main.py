import torch
import torch.optim as optim
import torch.nn as nn
import os
import config
from helper.log import logger
from helper.dataset import load_pcd_with_point_labels, PointCloudSegmentationDataset
from torch.utils.data import DataLoader
from model import PointNetSegmentation
from helper.plot import plot_training_stats, plot_test_prediction
from helper.loss import FocalLoss

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Main:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_model_path = os.path.join(config.RES_DIR, "best_model.pth")
        os.makedirs(config.RES_DIR, exist_ok=True)

    def predict(self):
        pass

    def _train(self, model, loader, criterion, optimizer):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for points, labels in loader:
            points, labels = points.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(points)
            
            # Reshape for loss calculation
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
        
        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def _eval(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for points, labels in loader:
                points, labels = points.to(self.device), labels.to(self.device)
                outputs = model(points)
                
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
    
    def _test(self, model, test_dataset):
        checkpoint = torch.load(self.save_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        with torch.no_grad():
            for (points, label) in test_dataset:
                points= points.unsqueeze(0).to(self.device)
                outputs = model(points)
                
                pc_plot = points.squeeze(0).transpose(0, 1).cpu().numpy()
                pred_labels = outputs.squeeze(0).argmax(dim=1).cpu().numpy()
                             
                plot_test_prediction(pc_plot, label, pred_labels)

    def train(self, val_interval=1):
        TRAIN_DIR = os.path.join(config.DS_ROOT, "train")
        VAL_DIR = os.path.join(config.DS_ROOT, "val")
        TEST_DIR = os.path.join(config.DS_ROOT, "test")

        train_point_clouds, train_labels = load_pcd_with_point_labels(TRAIN_DIR, augment=True)
        val_point_clouds, val_labels = load_pcd_with_point_labels(VAL_DIR)
        test_point_clouds, test_labels = load_pcd_with_point_labels(TEST_DIR)
        
        logger.info(f"\nTrain samples: {len(train_point_clouds)}")
        logger.info(f"Validation samples: {len(val_point_clouds)}")

        train_dataset = PointCloudSegmentationDataset(train_point_clouds, train_labels, augment=True)
        val_dataset = PointCloudSegmentationDataset(val_point_clouds, val_labels, augment=False)
        test_dataset = PointCloudSegmentationDataset(test_point_clouds, test_labels, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

        logger.info(f"\nTrain batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")

        num_classes = len(config.CLASSES)
        model = PointNetSegmentation(num_classes=num_classes).to(self.device)

        #"""
        criterion = FocalLoss() #nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        best_val_acc = 0.0
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        for epoch in range(config.EPOCHS):
            logger.info(f'\nEpoch {epoch+1}/{config.EPOCHS}')
            logger.info('-' * 60)
            
            train_loss, train_acc = self._train(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._eval(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'num_classes': num_classes
                }, self.save_model_path)
                logger.info(f'Saved best model with validation accuracy: {val_acc:.2f}%')
            
            scheduler.step()

        logger.info(f"\n=== Training Complete ===")
        logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")

        plot_training_stats(train_losses, val_losses, train_accuracies, val_accuracies)
        #"""
        
        self._test(model, test_dataset)

    def main(self):
        while True:
            logger.print("\n=== Menu ===")
            logger.print("1. Train model")
            logger.print("2. Predict")

            choice = input("Nr: ").strip()

            if not choice:
                break
            elif choice == "1":
                logger.clear()
                self.train()
            elif choice == "2":
                self.predict()

Main().main()