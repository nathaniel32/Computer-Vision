import torch
import torch.optim as optim
import os
import configs
from helper.utils.log import Logging
from helper.train.dataset import PointCloudSegmentationDataset, load_pcd_with_point_labels, get_chunks_indices
from torch.utils.data import DataLoader
from helper.train.model import PointNetSegmentation
from helper.utils.plot import plot_training_stats, plot_point_cloud
from helper.train.loss import FocalLoss, compute_alpha
from helper.train.scheduler import WarmupScheduler
import helper.mesh.mesh_remover
from helper.mesh.mesh_converter import convert_mesh_to_point_cloud_folder, save_point_cloud_in_pcd
import numpy as np
import random
from helper.train.augment import Augmenter
from helper.mesh.mesh_scaler import measure_marker_all_axes, calculate_scale_factor, plot_marker_all_axes, scale_mesh, filter_largest_cluster

random.seed(configs.SEED)
np.random.seed(configs.SEED)
torch.manual_seed(configs.SEED)
torch.cuda.manual_seed(configs.SEED)
torch.cuda.manual_seed_all(configs.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Main:
    def __init__(self, config:configs.BaseConfig):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.out_root_dir = os.path.join(configs.RES_ROOT_DIR, config.name)
        self.logger = Logging(log_dir=self.out_root_dir)
        self.save_model_path = os.path.join(self.out_root_dir, "best_model.pth")
        os.makedirs(self.out_root_dir, exist_ok=True)

    def make_dataset(self, input_dir, out_dir, total_num_points=500000) -> None:
        os.makedirs(out_dir, exist_ok=True)

        processed_count = 0
        
        for dir_path, subfolders, filenames in os.walk(input_dir):
            if not filenames:
                continue

            clean_name = os.path.basename(dir_path).replace(" ", "_").replace("(", "").replace(")", "")

            filename = clean_name + ".pcd"

            try:
                points, colors_int, obj_path = convert_mesh_to_point_cloud_folder(dir_path, total_num_points=total_num_points, visualize=True)
                save_point_cloud_in_pcd(points, colors_int, out_dir, filename=filename)
                processed_count += 1

            except Exception as e:
                print(f"✗ Failed to process {dir_path}: {e}")

        print(f"\n{'='*60}")
        print(f"- ALL DONE! Processed {processed_count} meshes")
        print(f"{'='*60}")

    def _get_and_load_model(self):
        # load model
        checkpoint = torch.load(self.save_model_path, weights_only=True, map_location=torch.device(self.device))
        model_state_dict = checkpoint['model_state_dict']
        classes = checkpoint['classes']
        model_num_points = checkpoint['model_num_points']
        val_acc = checkpoint['val_acc']
        epoch = checkpoint['epoch']
        model = PointNetSegmentation(num_classes=len(classes)).to(self.device)
        model.load_state_dict(model_state_dict)
        print(f"Checkpoint loaded: epoch={epoch}, val_acc={val_acc}, chunk_size={model_num_points}")
        return model, model_num_points, classes
    
    def _predicting(self, input_dir_path, output_dir_path, total_num_points=500000, smoothing=False):
        os.makedirs(output_dir_path, exist_ok=True)
        
        model, model_num_points, model_classes = self._get_and_load_model()

        model.eval()
        with torch.no_grad():
            # divide into chunks
            chunks_indices = get_chunks_indices(total_num_points, model_num_points)

            # obj to point cloud
            points, colors_int, mesh_file_path = convert_mesh_to_point_cloud_folder(input_dir_path, total_num_points=total_num_points)

            comb_points = []
            comb_color_norm = []
            comb_pred_label = []
            comb_color_int = []

            for i, chunk_indices in enumerate(chunks_indices, start=1):
                print(f"- Chunk {i}/{len(chunks_indices)}")
                points_chunk = points[chunk_indices]
                colors_int_chunk = colors_int[chunk_indices]
                pred_dataset = PointCloudSegmentationDataset([points_chunk], [colors_int_chunk])
                
                for (t_point, t_color) in pred_dataset:
                    t_point = t_point.unsqueeze(0).to(self.device)
                    t_color = t_color.unsqueeze(0).to(self.device)
                    outputs = model(t_point, t_color)

                    point_plot = t_point.squeeze(0).transpose(0, 1).cpu().numpy()
                    color_plot = t_color.squeeze(0).transpose(0, 1).cpu().numpy()
                    pred_label = outputs.squeeze(0).argmax(dim=1).cpu().numpy()
                    if smoothing:
                        pred_label = helper.mesh.mesh_remover.smooth_labels(points=point_plot, pred_label=pred_label) #extra smoothing

                    comb_points.extend(points_chunk)
                    comb_color_norm.extend(color_plot)
                    comb_pred_label.extend(pred_label)
                    comb_color_int.extend(colors_int_chunk)
                    
            comb_points = np.array(comb_points)
            comb_color_norm = np.array(comb_color_norm)
            comb_pred_label = np.array(comb_pred_label)

            save_point_cloud_in_pcd(comb_points, comb_color_int, output_dir_path, label=comb_pred_label)

            return comb_points, comb_color_norm, comb_pred_label, model_classes, mesh_file_path

    def cleaning_object(self, input_dir_path, output_dir_path, keep_label):
        points, color, pred_label, model_classes, mesh_file_path = self._predicting(input_dir_path, output_dir_path)

        plot_point_cloud(points, color, model_classes, pred_label=pred_label, plot_tool="open3d", save_dir=output_dir_path)
        
        # trim mesh
        trim_out_path = os.path.join(output_dir_path, "trim_mesh.obj")
        helper.mesh.mesh_remover.remove_object_part_v2(points, pred_label, mesh_file_path, trim_out_path, keep_label)
        

        keep_indecies = pred_label == keep_label # keep
        remove_indecies = pred_label != keep_label # remove
        plot_point_cloud(points[keep_indecies], color[keep_indecies], model_classes, pred_label=pred_label[keep_indecies], plot_tool="open3d", save_dir=output_dir_path, file_category="keep")
        plot_point_cloud(points[remove_indecies], color[remove_indecies], model_classes, pred_label=pred_label[remove_indecies], plot_tool="open3d", save_dir=output_dir_path, file_category="remove")

    def scaling_object(self, input_dir_path, output_dir_path, real_marker_diameter_cm):
        points, color, pred_label, model_classes, mesh_file_path = self._predicting(input_dir_path, output_dir_path)

        plot_point_cloud(points, color, model_classes, pred_label=pred_label, plot_tool="open3d", save_dir=output_dir_path)

        all_markers_metrics = []
        for label in self.config.scale_labels:
            try:
                marker_indecies = pred_label == label
                marker_points = points[marker_indecies]
                filtered_marker_points = filter_largest_cluster(marker_points)
                marker_axes_metrics, center = measure_marker_all_axes(filtered_marker_points)
                all_markers_metrics.append(marker_axes_metrics)
                # Plot
                #plot_marker_all_axes(points, center, results)
                plot_marker_all_axes(marker_points, center, marker_axes_metrics)
                plot_marker_all_axes(filtered_marker_points, center, marker_axes_metrics)
            except Exception as e:
                print(e)

        scale_factor = calculate_scale_factor(all_markers_metrics, real_marker_diameter_cm)
        scale_mesh(scale_factor, mesh_file_path, output_dir_path)
    
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
        model, model_num_points, model_classes = self._get_and_load_model()

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
                
                plot_point_cloud(point_plot, color_plot, model_classes, pred_label=pred_label, true_label=label, plot_tool="open3d")
    
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
            if not os.path.exists(self.save_model_path):
                raise FileNotFoundError(f"Checkpoint not found: {self.save_model_path}")
            
            self.logger.info(f"\n=== Resuming from checkpoint ===")
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
                    }, self.save_model_path)
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

    def main(self):
        while True:
            self.logger.print("\n=== Menu ===")
            self.logger.print("1. Train Model")
            self.logger.print("2. Test Model")
            self.logger.print("3. Cleaning Object")
            self.logger.print("4. Scaling Object")
            self.logger.print("5. Mesh to point cloud")

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
                keep_label = int(input("Keep Label ID: "))
                self.cleaning_object(input_dir_path, output_dir_path, keep_label)
            elif choice == "4":
                input_dir_path = input("Input Dir Path: ").strip('"').strip()
                output_dir_path = input("Output Dir Path: ").strip('"').strip()
                real_marker_diameter_cm = float(input("Marker Diameter (cm): "))
                self.scaling_object(input_dir_path, output_dir_path, real_marker_diameter_cm)
            elif choice == "5":
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
    for i, conf in enumerate(configs.CONFIG_LIST):
        print(f"{i}: {conf.name}")

    selected_config = int(input("ID: "))
    Main(configs.CONFIG_LIST[selected_config]).main()