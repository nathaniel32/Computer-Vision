import torch
import config
import os
from sklearn import preprocessing
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import logger, to_yhat, show_conf_matrix
from model import Natnet
import joblib
import torch.nn as nn
from torchvision import transforms

""" 
class DatasetManager():
    def __init__(self, x_data_path, y_label, size):
        self.x_data_path = x_data_path
        self.y_label = y_label
        self.size = size

    def __len__(self):
        return len(self.x_data_path)
    
    def __getitem__(self, idx):
        img_path = self.x_data_path[idx]
        image = Image.open(img_path).resize((self.size, self.size)).convert('RGB')
        image_array = np.array(image) / 255.0
        image_array = np.moveaxis(image_array, -1, 0) # (H, W, C) -> (C, H, W)

        return {
            'image': torch.tensor(image_array, dtype=torch.float32),
            'label': torch.tensor(self.y_label[idx], dtype=torch.int64)
        }
"""

class DatasetManager():
    def __init__(self, x_data_path, y_label, size):
        self.x_data_path = x_data_path
        self.y_label = y_label
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),  # scale [0,1], (H,W,C) -> (C,H,W)
        ])

    def __len__(self):
        return len(self.x_data_path)
    
    def __getitem__(self, idx):
        img_path = self.x_data_path[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(self.y_label[idx], dtype=torch.int64)
        }

class Trainer:
    def _get_datasets(self):
        DATASET_PATH = "dataset"
        data_x = []
        data_y = []
        
        # read
        for label_name in os.listdir(DATASET_PATH): # baca dir
            label_path = os.path.join(DATASET_PATH, label_name)
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if os.path.isfile(file_path):
                    data_x.append(file_path)
                    data_y.append(label_name)

        # encode
        labels_encoder = preprocessing.LabelEncoder()
        data_y_encoded = labels_encoder.fit_transform(data_y)

        # split
        train_data, val_data, train_labels, val_labels = train_test_split(data_x, data_y_encoded, test_size=0.2, random_state=42, stratify=data_y_encoded)

        train_dataset = DatasetManager(x_data_path=train_data, y_label=train_labels, size=config.IMG_SIZE)
        val_dataset = DatasetManager(x_data_path=val_data, y_label=val_labels, size=config.IMG_SIZE)

        train_data_loader = DataLoader( dataset=train_dataset,
                                        batch_size=config.TRAIN_BATCH_SIZE,
                                        shuffle=True,
                                        #num_workers=4,
                                        pin_memory=True)
        
        val_data_loader = DataLoader(dataset=val_dataset,
                                    batch_size=config.VALIDATION_BATCH_SIZE,
                                    shuffle=False,
                                    #num_workers=4,
                                    pin_memory=True)
        
        logger("Train data:", len(train_data))
        logger("Val data:", len(val_data))
        logger("Train labels:", len(train_labels))
        logger("Val labels:", len(val_labels))

        # save conf
        meta_data = {
            'encoder_labels': labels_encoder
        }
        os.makedirs(os.path.dirname(config.META_PATH), exist_ok=True)
        joblib.dump(meta_data, config.META_PATH)

        return train_data_loader, val_data_loader, labels_encoder

    def _train(self, data_loader, model, optimizer, criterion):
        model.train()
        final_loss = 0

        for batch in data_loader:
            image = batch['image'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            final_loss += loss.item()

        return final_loss/len(data_loader)

    def _val(self, data_loader, model, criterion):
        model.eval()
        final_loss = 0
        preds_array = []
        solution_array = []

        with torch.no_grad():
            for batch in data_loader:
                image = batch['image'].to(config.DEVICE)
                labels = batch['label'].to(config.DEVICE)

                output =  model(image)
                loss =  criterion(output, labels)
    
                final_loss += loss.item()

                _, preds_labels = to_yhat(output)

                preds_array.extend(preds_labels)
                solution_array.extend(labels)

        return final_loss/len(data_loader), preds_array, solution_array

    def main(self):
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        train_data_loader, val_data_loader, labels_encoder = self._get_datasets()

        model = Natnet(3, 32, len(labels_encoder.classes_), size=config.IMG_SIZE)
        model.to(config.DEVICE)
        model.info()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5, mode="min")

        for epoch in range(config.EPOCHS):
            train_loss = self._train(train_data_loader, model, optimizer, criterion)
            val_loss, preds_array, solution_array = self._val(val_data_loader, model, criterion)
            
            scheduler.step(val_loss)

            print(f'\n== Epoch {epoch + 1}/{config.EPOCHS}')
            print(f'Train Loss: {train_loss}')
            print(f'Validation Loss: {val_loss}')

            best_loss = float('inf')
            if val_loss < best_loss and config.SAVE_MODEL:
                best_preds_array = preds_array
                best_solution_array = solution_array
                best_loss = val_loss

                os.makedirs(os.path.dirname(config.TRAINED_PATH), exist_ok=True)
                torch.save(model.state_dict(), config.TRAINED_PATH)
                print('new Model')

            show_conf_matrix(preds_array=preds_array, solution_array=solution_array, label=labels_encoder.classes_, title=f"{epoch + 1} Epochs | Total: {len(preds_array)}", plot=best_loss==val_loss)

if __name__ == '__main__':
    t = Trainer().main()