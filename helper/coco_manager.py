import json
import cv2
import numpy as np
from PIL import Image, ImageOps
from utils import logger
import os
from torch.utils.data import DataLoader
import json
from helper.train import DatasetManager
from helper.plot_manager import PlotManager

class CocoManager:
    def __init__(self):
        self.plot_manager = PlotManager()
    
    def _load_coco_data(self, dir_root, dir_name, file_name='_annotations.coco.json'):
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

    def transform_image(self, image_pil, target_size, mask=None):
        # Normalisasi EXIF orientation pada gambar
        image_pil = ImageOps.exif_transpose(image_pil)  # hapus Orientation setelah diterapkan

        # Tentukan orientasi berdasarkan dimensi gambar pasca-EXIF
        width, height = image_pil.size
        rotate90 = (width >= height)  # landscape atau square -> portrait

        if rotate90:
            # Gambar: 90° CCW
            image_pil = image_pil.transpose(Image.ROTATE_90)
            # Mask (jika ada): 90° CCW tanpa interpolasi
            if mask is not None:
                mask = np.rot90(mask, k=1)  # CCW, konservasi label

        # Resize
        image_pil = image_pil.resize(target_size)
        if mask is not None:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)  # jaga nilai kelas

        return image_pil, mask, rotate90

    def _create_dataset(self, coco_data, dir_root, dir_name, target_size):
        images_data = []
        labels_data = []
        failed_loads = 0

        # Lookup annotation per image
        ann_by_image = {}
        for ann in coco_data['annotations']:
            ann_by_image.setdefault(ann['image_id'], []).append(ann)

        for image_data in coco_data['images']:
            file_name = image_data['file_name']
            img_path = os.path.join(dir_root, dir_name, file_name)

            # Bangun mask pada resolusi asli COCO
            mask = np.zeros((image_data['height'], image_data['width']), dtype=np.uint8)

            for annotation in ann_by_image.get(image_data['id'], []):
                category_index = annotation['category_id']
                for poly in annotation['segmentation']:
                    poly = np.array(poly).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [poly], color=category_index)

            try:
                image_pil = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Load error: {img_path}: {e}")  # fail-fast
                failed_loads += 1
                continue

            if image_pil.size[0] == 0 or image_pil.size[1] == 0:
                logger.warning(f"Invalid image dimensions: {img_path}")  # fail-fast
                failed_loads += 1
                continue
            
            image_pil, mask, rotate90 = self.transform_image(image_pil, target_size, mask=mask)

            images_data.append(image_pil)
            labels_data.append(mask)

        logger.info(f"Dataset created: {len(images_data)} samples, {failed_loads} failed loads")
        return images_data, labels_data

    def prepare_datasets(self, batch_size, ds_root, image_size, log_dir):
        ds_name = "train"
        logger.info("Loading COCO data...")
        coco_data_train = self._load_coco_data(ds_root, ds_name)
        logger.info("Creating category mappings...")
        categories_classes = {cat['id']: cat['name'] for cat in coco_data_train['categories']}
        logger.info("Creating dataset...")
        X_train, Y_train = self._create_dataset(
            coco_data_train, ds_root, ds_name, image_size
        )

        ds_name = "valid"
        logger.info("Loading COCO data...")
        coco_data_valid = self._load_coco_data(ds_root, ds_name)
        logger.info("Creating dataset...")
        X_valid, Y_valid = self._create_dataset(
            coco_data_valid, ds_root, ds_name, image_size
        )

        ds_name = "test"
        logger.info("Loading COCO data...")
        coco_data_test = self._load_coco_data(ds_root, ds_name)
        logger.info("Creating dataset...")
        X_test, Y_test = self._create_dataset(
            coco_data_test, ds_root, ds_name, image_size
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
        
        # Plot sample data
        self.plot_manager.plot_data_samples(X_train, Y_train, num_samples=10)

        train_dataset = DatasetManager(X_train, Y_train, augment=True)
        val_dataset = DatasetManager(X_valid, Y_valid, augment=False)
        test_dataset = DatasetManager(X_test, Y_test, augment=False)

        logger.info("Train: ", len(train_dataset))
        logger.info("Val: ", len(val_dataset))
        logger.info("Test: ", len(test_dataset))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        return train_loader, val_loader, test_loader, categories_classes