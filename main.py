import torch
import torch.optim as optim
import torch.nn as nn
import os
import config
from helper.log import logger
from helper.dataset import load_pcd_with_point_labels, PointCloudSegmentationDataset, transform_data
from torch.utils.data import DataLoader
from model import PointNetSegmentation
from helper.plot import plot_training_stats, plot_prediction
from helper.loss import FocalLoss
import helper.preds

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

    def predict_object(self):
        mesh_file_path = r"C:\Users\natha\Downloads\test_preds\texturedMesh.obj"
        texture_file_path = r"C:\Users\natha\Downloads\test_preds\texture_1001.png"
        save_pcd_path = r"C:\Users\natha\Downloads\test_preds\point_cloud.pcd"
        save_obj_trim_path = r"C:\Users\natha\Downloads\test_preds\trim_mesh.obj"

        # obj to point cloud
        points, rgb_ints, colors_rgb = helper.preds.mesh_to_point_cloud(mesh_file_path, texture_file_path, save_pcd_path, num_points=config.NUM_SAMPLE_POINTS)

        # Visualize
        #helper.preds.visualize_pointcloud(points, colors_rgb)

        sampled_points, sampled_colors = transform_data(points, rgb_ints)

        pred_dataset = PointCloudSegmentationDataset([sampled_points], [sampled_colors])

        num_classes = len(config.CLASSES)
        model = PointNetSegmentation(num_classes=num_classes).to(self.device)

        checkpoint = torch.load(self.save_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        with torch.no_grad():
            for (point, color) in pred_dataset:
                point = point.unsqueeze(0).to(self.device)
                color = color.unsqueeze(0).to(self.device)
                outputs = model(point, color)

                point_plot = point.squeeze(0).transpose(0, 1).cpu().numpy()
                color_plot = color.squeeze(0).transpose(0, 1).cpu().numpy()
                pred_label = outputs.squeeze(0).argmax(dim=1).cpu().numpy()
                
                #plot_prediction(point_plot, color_plot, pred_label, plot_tool="open3d")
                
                remove_item_id = 0
                helper.preds.remove_object_part(points, pred_label, mesh_file_path, save_obj_trim_path, remove_item_id)
                plot_prediction(points[pred_label == remove_item_id], color_plot[pred_label == remove_item_id], pred_label[pred_label == remove_item_id], plot_tool="open3d")
                plot_prediction(points[pred_label != remove_item_id], color_plot[pred_label != remove_item_id], pred_label[pred_label != remove_item_id], plot_tool="open3d")

    def _train(self, model, loader, criterion, optimizer):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for points, colors, labels in loader:
            points, colors, labels = points.to(self.device), colors.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(points, colors)
            
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
        TEST_DIR = os.path.join(config.DS_ROOT, "test")
        test_point_clouds, test_colors, test_labels = load_pcd_with_point_labels(TEST_DIR)
        test_dataset = PointCloudSegmentationDataset(test_point_clouds, test_colors, labels=test_labels)

        num_classes = len(config.CLASSES)
        model = PointNetSegmentation(num_classes=num_classes).to(self.device)

        checkpoint = torch.load(self.save_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        with torch.no_grad():
            for (point, color, label) in test_dataset:
                point = point.unsqueeze(0).to(self.device)
                color = color.unsqueeze(0).to(self.device)
                outputs = model(point, color)
                
                point_plot = point.squeeze(0).transpose(0, 1).cpu().numpy()
                color_plot = color.squeeze(0).transpose(0, 1).cpu().numpy()
                pred_label = outputs.squeeze(0).argmax(dim=1).cpu().numpy()
                
                plot_prediction(point_plot, color_plot, pred_label, true_label=label)

    def train(self, val_interval=1):
        TRAIN_DIR = os.path.join(config.DS_ROOT, "train")
        VAL_DIR = os.path.join(config.DS_ROOT, "val")

        train_point_clouds, train_colors, train_labels = load_pcd_with_point_labels(TRAIN_DIR, augment=True)
        val_point_clouds, val_colors, val_labels = load_pcd_with_point_labels(VAL_DIR)
        
        
        logger.info(f"\nTrain samples: {len(train_point_clouds)}")
        logger.info(f"Validation samples: {len(val_point_clouds)}")

        train_dataset = PointCloudSegmentationDataset(train_point_clouds, train_colors, labels=train_labels)
        val_dataset = PointCloudSegmentationDataset(val_point_clouds, val_colors, labels=val_labels)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

        logger.info(f"\nTrain batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")

        num_classes = len(config.CLASSES)
        model = PointNetSegmentation(num_classes=num_classes).to(self.device)

        #"""
        criterion = FocalLoss() #nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LR)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)
        
        patience_counter = 0
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
                patience_counter = 0
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'num_classes': num_classes
                }, self.save_model_path)
                logger.info(f'Saved best model with validation accuracy: {val_acc:.2f}%')
            else:
                patience_counter += 1
                logger.info(f"- Patience: {patience_counter}/{config.PATIENCE}")

                if patience_counter >= config.PATIENCE:
                    logger.info("= Early stopping triggered!")
                    break
            
            #scheduler.step()
            scheduler.step(val_loss)

        logger.info(f"\n=== Training Complete ===")
        logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")

        plot_training_stats(train_losses, val_losses, train_accuracies, val_accuracies)
        #"""
        
        #self.test()

    def main(self):
        while True:
            logger.print("\n=== Menu ===")
            logger.print("1. Train Model")
            logger.print("2. Test Model")
            logger.print("3. Predict Object")

            choice = input("Nr: ").strip()

            if not choice:
                break
            elif choice == "1":
                logger.clear()
                self.train()
            elif choice == "2":
                self.test()
            elif choice == "3":
                self.predict_object()

Main().main()