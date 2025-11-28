import torch
import torch.optim as optim
import os
import config
from helper.utils.log import logger
from helper.train.dataset import PointCloudSegmentationDataset, load_pcd_with_point_labels, get_chunks_indices
from torch.utils.data import DataLoader
from model import PointNetSegmentation
from helper.utils.plot import plot_training_stats, plot_point_cloud
from helper.train.loss import FocalLoss, compute_alpha
from helper.train.scheduler import WarmupScheduler
import helper.mesh.mesh_remover
from helper.mesh.mesh_to_pointclouds import convert_mesh_folder_to_pcd, mesh_to_point_cloud
import numpy as np
import random

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Main:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_model_path = os.path.join(config.RES_DIR, "best_model.pth")
        os.makedirs(config.RES_DIR, exist_ok=True)

    def make_dataset(self, input_dir, out_dir) -> None:
        os.makedirs(out_dir, exist_ok=True)

        processed_count = 0
        
        for dir_path, subfolders, filenames in os.walk(input_dir):
            if not filenames:
                continue

            clean_name = os.path.basename(dir_path).replace(" ", "_").replace("(", "").replace(")", "")

            pcd_out_path = os.path.join(out_dir, clean_name + ".pcd")

            try:
                convert_mesh_folder_to_pcd(dir_path, pcd_out_path, num_points=100000)
                processed_count += 1

            except Exception as e:
                print(f"✗ Failed to process {dir_path}: {e}")

        print(f"\n{'='*60}")
        print(f"- ALL DONE! Processed {processed_count} meshes")
        print(f"{'='*60}")

    def predict_object(self, input_dir_path, output_dir_path, plot=False, target_num_points=100000):
        os.makedirs(output_dir_path, exist_ok=True)
        
        checkpoint = torch.load(self.save_model_path, weights_only=True, map_location=torch.device(self.device))
        c_model_state_dict = checkpoint['model_state_dict']
        c_num_classes = checkpoint['num_classes']
        c_num_points = checkpoint['num_points']
        c_val_acc = checkpoint['val_acc']
        c_epoch = checkpoint['epoch']
        model = PointNetSegmentation(num_classes=c_num_classes).to(self.device)
        model.load_state_dict(c_model_state_dict)
        print(f"Checkpoint loaded: epoch={c_epoch}, val_acc={c_val_acc}")

        model.eval()
        with torch.no_grad():
            # divide into chunks
            chunks_indices = get_chunks_indices(target_num_points, c_num_points)

            # obj to point cloud
            pcd_out_path = os.path.join(output_dir_path, "point_cloud.pcd")
            points, colors_int, mesh_file_path = convert_mesh_folder_to_pcd(input_dir_path, pcd_out_path, num_points=100000)

            comb_points = []
            comb_color = []
            comb_pred_label = []

            for indices in chunks_indices:
                points_chunk = points[indices]
                colors_chunk = colors_int[indices]
                pred_dataset = PointCloudSegmentationDataset([points_chunk], [colors_chunk])
                
                for (t_point, t_color) in pred_dataset:
                    t_point = t_point.unsqueeze(0).to(self.device)
                    t_color = t_color.unsqueeze(0).to(self.device)
                    outputs = model(t_point, t_color)

                    point_plot = t_point.squeeze(0).transpose(0, 1).cpu().numpy()
                    color_plot = t_color.squeeze(0).transpose(0, 1).cpu().numpy()
                    pred_label = outputs.squeeze(0).argmax(dim=1).cpu().numpy()
                    pred_label = helper.mesh.mesh_remover.smooth_labels(points=point_plot, pred_label=pred_label)

                    comb_points.extend(points_chunk)
                    comb_color.extend(color_plot)
                    comb_pred_label.extend(pred_label)
                    
            comb_points = np.array(comb_points)
            comb_color = np.array(comb_color)
            comb_pred_label = np.array(comb_pred_label)
            
            keep_label_id = 1
            trim_out_path = os.path.join(output_dir_path, "trim_mesh.obj")
            helper.mesh.mesh_remover.remove_object_part_v2(comb_points, comb_pred_label, mesh_file_path, trim_out_path, keep_label_id)
            
            # keep
            keep_indecies = comb_pred_label == keep_label_id
            
            # remove
            remove_indecies = comb_pred_label != keep_label_id

            if plot:
                plot_point_cloud(comb_points, comb_color, pred_label=comb_pred_label, plot_tool="open3d")
                plot_point_cloud(comb_points[keep_indecies], comb_color[keep_indecies], pred_label=comb_pred_label[keep_indecies], plot_tool="open3d")
                plot_point_cloud(comb_points[remove_indecies], comb_color[remove_indecies], pred_label=comb_pred_label[remove_indecies], plot_tool="open3d")
    
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
        TEST_DIR = os.path.join(config.DS_ROOT, "test")
        test_point_clouds, test_colors, test_labels = load_pcd_with_point_labels(TEST_DIR, sampling=True)
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
                
                plot_point_cloud(point_plot, color_plot, pred_label=pred_label, true_label=label)
    
    def train(self, val_interval=1, warmup_epochs=5, resume=False):
        if not resume:
            logger.clear()
        
        TRAIN_DIR = os.path.join(config.DS_ROOT, "train")
        VAL_DIR = os.path.join(config.DS_ROOT, "val")

        train_point_clouds, train_colors, train_labels = load_pcd_with_point_labels(TRAIN_DIR)
        val_point_clouds, val_colors, val_labels = load_pcd_with_point_labels(VAL_DIR, sampling=True)
        
        logger.info(f"\nTrain samples: {len(train_point_clouds)}")
        logger.info(f"Validation samples: {len(val_point_clouds)}")

        train_dataset = PointCloudSegmentationDataset(train_point_clouds, train_colors, labels=train_labels, augment=True, sampling=True)
        val_dataset = PointCloudSegmentationDataset(val_point_clouds, val_colors, labels=val_labels)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

        logger.info(f"\nTrain batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")

        num_classes = len(config.CLASSES)
        model = PointNetSegmentation(num_classes=num_classes).to(self.device)

        alpha = compute_alpha(train_labels=train_labels, num_classes=num_classes).to(self.device)
        criterion = FocalLoss(alpha=alpha)
        optimizer = optim.Adam(model.parameters(), lr=config.LR)
        
        # Initialize warm-up scheduler
        warmup_scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            initial_lr=config.LR * 0.1,
            target_lr=config.LR
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
            if not os.path.exists(self.save_model_path):
                raise FileNotFoundError(f"Checkpoint not found: {self.save_model_path}")
            
            logger.info(f"\n=== Resuming from checkpoint ===")
            checkpoint = torch.load(self.save_model_path, map_location=self.device)
            
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
            
            logger.info(f"Resumed from epoch {start_epoch}")
            logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
            logger.info(f"Patience counter: {patience_counter}")
        else:
            logger.info(f"\n=== Training with {warmup_epochs} epochs warm-up ===")
            logger.info(f"Initial LR: {config.LR * 0.1:.2e} -> Target LR: {config.LR:.2e}")
        
        for epoch in range(start_epoch, config.EPOCHS):
            logger.info(f'\nEpoch {epoch+1}/{config.EPOCHS}')
            logger.info('-' * 60)
            
            # Apply warm-up
            if not resume and epoch < warmup_epochs:
                warmup_scheduler.step()
                current_lr = warmup_scheduler.get_lr()
                logger.info(f'Warm-up LR: {current_lr:.2e}')
            
            train_loss, train_acc = self._train(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._eval(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
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
                        'num_classes': num_classes,
                        'num_points': config.NUM_SAMPLE_POINTS,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_accuracies': train_accuracies,
                        'val_accuracies': val_accuracies,
                        'patience_counter': patience_counter,
                        'best_val_acc': best_val_acc
                    }, self.save_model_path)
                    logger.info(f'Saved best model with validation accuracy: {val_acc:.2f}%')
                else:
                    patience_counter += 1
                    logger.info(f"- Patience: {patience_counter}/{config.PATIENCE}")

                    if patience_counter >= config.PATIENCE:
                        logger.info("= Early stopping triggered!")
                        break
            
                # Apply main scheduler
                main_scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Current LR: {current_lr:.2e}')

        logger.info(f"\n=== Training Complete ===")
        logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")

        plot_training_stats(train_losses, val_losses, train_accuracies, val_accuracies)

    def main(self):
        while True:
            logger.print("\n=== Menu ===")
            logger.print("1. Train Model")
            logger.print("2. Test Model")
            logger.print("3. Predict Object")
            logger.print("4. Mesh to pointcloud")

            choice = input("Nr: ").strip()

            if not choice:
                break
            elif choice == "1":
                resume = input("Resume training? (y/n): ").strip().lower() == 'y'
                self.train(resume=resume)
            elif choice == "2":
                self.test()
            elif choice == "3":
                input_dir_path = input("Input Dir Path: ").strip('"').strip()
                output_dir_path = input("Output Dir Path: ").strip('"').strip()
                self.predict_object(input_dir_path, output_dir_path, plot=True)
            elif choice == "4":
                print("""Expected folder structure:
                input_dir/
                    data_1/
                        - mesh.obj
                        - mesh.mtl
                        - mesh_tex0.png
                    data_2/
                        - mesh.obj
                        - mesh.mtl
                        - mesh_tex0.png
                    ...
                """)
                input_dir = input('Input directory: ')
                out_dir = input('Output directory: ')
                self.make_dataset(input_dir, out_dir)

if __name__ == "__main__":
    Main().main()