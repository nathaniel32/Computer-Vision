import numpy as np
import torch
import torch.optim as optim
from utils import logger
import helper.train
import helper.predict
from model import ConditionalSegmentationModel
import os
from PIL import Image
import joblib
import config

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Main:
    def __init__(self):
        self.ds_root = config.DS_ROOT
        self.res_dir = config.RES_DIR
        self.log_dir = config.LOG_DIR
        self.n_epochs = config.N_EPOCHS
        self.patience = config.PATIENCE
        self.lr = config.LR
        self.weight_decay = config.WEIGHT_DECAY
        self.batch_size = config.BATCH_SIZE
        self.emb_dim = config.EMB_DIM
        self.dropout_rate = config.DROPOUT_RATE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_model_path = os.path.join(config.RES_DIR, "best_model.pth")
        self.save_meta_path = os.path.join(config.RES_DIR, "meta.bin")
        self.image_size = (224, 224)
    
    def predict(self):
        meta_data = joblib.load(self.save_meta_path)
        categories_classes = meta_data["categories_classes"]
        category_encoder = meta_data["category_encoder"]
        category_decoder = meta_data["category_decoder"]
        emb_dim = meta_data["emb_dim"]
        dropout_rate = meta_data["dropout_rate"]
        
        n_classes = len(category_decoder) # categories_classes tidak akurat
        logger.info("Num Classes:", n_classes)
        model = ConditionalSegmentationModel(n_classes=n_classes, emb_dim=emb_dim, dropout_rate=dropout_rate).to(self.device)
        logger.info("Loading best model for evaluation...")
        checkpoint = torch.load(self.save_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        with torch.no_grad():
            while True:
                img_path = input("Img Path: ").strip('"').strip()
                if not img_path:
                    break

                while True:
                    logger.info("\nID\tCategory")
                    for key, value in categories_classes.items():
                        logger.info(f"{key}\t {value}")

                    cat_id = input("\nCategory ID: ")
                    if not cat_id:
                        break
                    
                    cat_id = int(cat_id)
                    cat_index = category_encoder[cat_id]

                    logger.info("Category Index:", category_decoder[cat_index])

                    # load gambar asli
                    image_orig = Image.open(img_path).convert('RGB')
                    orig_size = image_orig.size  # (width, height)

                    # resize untuk model
                    image_pil = image_orig.resize(self.image_size)

                    # prepare data
                    empty_mask = np.zeros(self.image_size)
                    dataset = helper.train.DatasetManager([image_pil], [empty_mask], [cat_index])
                    image_tensor, _, category_tensor = dataset[0]
                    image_tensor = image_tensor.unsqueeze(0).to(self.device)
                    category_tensor = category_tensor.unsqueeze(0).to(self.device)

                    # prediksi
                    outputs = model(image_tensor, category_tensor)
                    mask_pred = outputs.squeeze().cpu().numpy()

                    title = categories_classes.get(category_decoder[category_tensor.item()])
                    helper.predict.plot_predictions(image_orig, orig_size, mask_pred, title)
                    
    def train(self, val_interval=1):
        train_loader, val_loader, test_loader, categories_classes, category_encoder, category_decoder = helper.train.prepare_datasets(self.batch_size, self.ds_root, self.image_size, self.log_dir)
        
        # save meta
        meta_data = {
            "categories_classes": categories_classes,
            "category_encoder": category_encoder,
            'category_decoder': category_decoder,
            'emb_dim': self.emb_dim,
            'dropout_rate': self.dropout_rate
        }
        joblib.dump(meta_data, self.save_meta_path)

        logger.info(f"Using device: {self.device}")

        n_classes = len(category_decoder)
        model = ConditionalSegmentationModel(n_classes=n_classes, emb_dim=self.emb_dim, dropout_rate=self.dropout_rate).to(self.device)

        # Optimizer with gradient clipping
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
        )
        criterion = helper.train.SegmentationLoss(
            alpha=0.25, gamma=2.0, dice_weight=0.7, 
            focal_weight=1.2, smooth=1e-6
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.n_epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            
            logger.info(f"Epoch {epoch+1}/{self.n_epochs}")
            for batch_idx, (images, masks, categories) in enumerate(train_loader):
                try:
                    images = images.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)
                    categories = categories.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    outputs = model(images, categories)
                    
                    loss = criterion(outputs, masks, loss_type='boundary_enhanced')
                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"!! NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 20 == 0 or batch_idx+1==len(train_loader):
                        logger.info(f"- Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    
                except Exception as e:
                    logger.error(f"!!! Error in training batch {batch_idx}: {e}")
                    continue
            
            if num_batches > 0:
                avg_train_loss = epoch_train_loss / num_batches
                train_losses.append(avg_train_loss)
                logger.info(f"- Average Train Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            if (epoch + 1) % val_interval == 0:
                model.eval()
                epoch_val_loss = 0.0
                val_num_batches = 0
                
                with torch.no_grad():
                    for batch_idx, (images, masks, categories) in enumerate(val_loader):
                        try:
                            images = images.to(self.device, non_blocking=True)
                            masks = masks.to(self.device, non_blocking=True)
                            categories = categories.to(self.device, non_blocking=True)
                            
                            outputs = model(images, categories)
                            loss = criterion(outputs, masks, loss_type='boundary_enhanced')
                            
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                epoch_val_loss += loss.item()
                                val_num_batches += 1
                        
                        except Exception as e:
                            logger.error(f"!!! Error in validation batch {batch_idx}: {e}")
                            continue
                
                if val_num_batches > 0:
                    avg_val_loss = epoch_val_loss / val_num_batches
                    val_losses.append(avg_val_loss)
                    logger.info(f"- Average Val Loss: {avg_val_loss:.4f}")
                    
                    scheduler.step(avg_val_loss)
                    
                    # Early stopping and model saving
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'val_loss': avg_val_loss,
                        }, self.save_model_path)
                        logger.info(f"- New best model saved with val loss: {avg_val_loss:.4f}")
                    else:
                        patience_counter += 1
                        logger.info(f"- Patience: {patience_counter}/{self.patience}")
                    
                    if patience_counter >= self.patience:
                        logger.info("= Early stopping triggered!")
                        break
        
        # plot graph
        helper.train.plot_training_curves(train_losses, val_losses)

        # Load best model and evaluate
        logger.info("Loading best model for evaluation...")
        checkpoint = torch.load(self.save_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # test
        logger.info("Evaluating model and plotting predictions...")
        category_metrics = helper.train.evaluate_and_plot_predictions(
            model, test_loader, self.device, categories_classes, category_decoder, num_samples=100
        )
        
        logger.info("Training and evaluation completed successfully!")

    def main(self):
        while True:
            logger.info("\n=== Menu ===")
            logger.info("1. Train model")
            logger.info("2. Predict")

            choice = input("Nr: ").strip()

            if not choice:
                break
            elif choice == "1":
                logger.clear()
                self.train()
            elif choice == "2":
                self.predict()

Main().main()