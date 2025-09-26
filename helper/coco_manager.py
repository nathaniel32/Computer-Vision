import json
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
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

    def _create_category_mapping(self, coco_data):
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Get all category IDs actually used in annotations
        annotation_category_ids = [ann['category_id'] for ann in coco_data['annotations']]
        unique_category_ids = sorted(list(set(annotation_category_ids)))
        
        # Validate all annotation categories exist in categories dict
        missing_categories = [cat_id for cat_id in unique_category_ids if cat_id not in categories]
        if missing_categories:
            logger.warning(f"Found annotations with missing category definitions: {missing_categories}")
        
        # Create continuous mapping starting from 0
        category_id_to_index = {cat_id: idx for idx, cat_id in enumerate(unique_category_ids)}
        index_to_category_id = {idx: cat_id for cat_id, idx in category_id_to_index.items()}
        
        # Check for class imbalance
        category_counts = defaultdict(int)
        for cat_id in annotation_category_ids:
            category_counts[cat_id] += 1
        
        logger.info(f"Category distribution: {dict(category_counts)}")
        
        # Warn about severe class imbalance
        max_count = max(category_counts.values())
        min_count = min(category_counts.values())
        if max_count / min_count > 10:
            logger.warning(f"Severe class imbalance detected. Ratio: {max_count/min_count:.2f}")
        
        return categories, category_id_to_index, index_to_category_id

    def _process_segmentation(self, seg, image_height, image_width):
        """process segmentation data"""
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        try:
            if isinstance(seg, list) and len(seg) > 0:
                if isinstance(seg[0], list):
                    # Multiple polygons
                    for polygon in seg:
                        if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                            polygon_array = np.array(polygon).reshape((-1, 2)).astype(np.int32)
                            # Clip coordinates to image bounds
                            polygon_array[:, 0] = np.clip(polygon_array[:, 0], 0, image_width-1)
                            polygon_array[:, 1] = np.clip(polygon_array[:, 1], 0, image_height-1)
                            cv2.fillPoly(mask, [polygon_array], color=1)
                else:
                    # Single polygon
                    if len(seg) >= 6:  # At least 3 points
                        seg_array = np.array(seg).reshape((-1, 2)).astype(np.int32)
                        # Clip coordinates to image bounds
                        seg_array[:, 0] = np.clip(seg_array[:, 0], 0, image_width-1)
                        seg_array[:, 1] = np.clip(seg_array[:, 1], 0, image_height-1)
                        cv2.fillPoly(mask, [seg_array], color=1)
            
            return mask
        
        except Exception as e:
            logger.warning(f"Error processing segmentation: {e}")
            return mask  # Return empty mask on error

    def _create_dataset(self, coco_data, dir_root, dir_name, category_id_to_index, target_size):
        # First, reorganize COCO data by image
        coco_dataset = {}
        for ann in coco_data['annotations']:
            image_id = ann["image_id"]
            cat_id = ann["category_id"]
            seg = ann["segmentation"]

            if image_id not in coco_dataset:
                try:
                    image_info = next(i for i in coco_data['images'] if i['id'] == image_id)
                    coco_dataset[image_id] = {
                        "image_id": image_id,
                        "image_name": image_info['file_name'],
                        "image_width": image_info['width'],
                        "image_height": image_info['height'],
                        "categories": []
                    }
                except StopIteration:
                    logger.warning(f"Image info not found for image_id: {image_id}")
                    continue

            # Find or create category entry
            found = None
            for cat in coco_dataset[image_id]["categories"]:
                if cat["category_id"] == cat_id:
                    found = cat
                    break

            if found:
                found["segmentations"].append(seg)
            else:
                coco_dataset[image_id]["categories"].append({
                    "category_id": cat_id,
                    "segmentations": [seg]
                })

        # Convert to list and process images
        coco_dataset = list(coco_dataset.values())
        
        images_data = []
        categories_data = []
        labels_data = []
        failed_loads = 0

        for image_data in coco_dataset:
            img_path = os.path.join(dir_root, dir_name, image_data['image_name'])
            
            try:
                # Validate image file exists
                if not os.path.exists(img_path):
                    logger.warning(f"Image file not found: {img_path}")
                    failed_loads += 1
                    continue
                
                # Load and validate image
                image_pil = Image.open(img_path).convert('RGB')
                
                # Check if image is corrupted
                if image_pil.size[0] == 0 or image_pil.size[1] == 0:
                    logger.warning(f"Invalid image dimensions: {img_path}")
                    failed_loads += 1
                    continue
                
                image_pil = image_pil.resize(target_size)
                
                for cat in image_data['categories']:
                    category_id = cat['category_id']
                    
                    # Skip if category not in mapping
                    if category_id not in category_id_to_index:
                        logger.warning(f"Category ID {category_id} not in mapping")
                        continue
                    
                    category_index = category_id_to_index[category_id]
                    
                    # Process mask with error handling
                    mask = np.zeros((image_data['image_height'], image_data['image_width']), dtype=np.uint8)
                    valid_segmentations = 0
                    
                    for seg in cat['segmentations']:
                        seg_mask = self._process_segmentation(
                            seg, image_data['image_height'], image_data['image_width']
                        )
                        if np.sum(seg_mask) > 0:  # Only add if mask is not empty
                            mask = np.logical_or(mask, seg_mask).astype(np.uint8)
                            valid_segmentations += 1
                    
                    # Skip if no valid segmentations
                    if valid_segmentations == 0:
                        logger.warning(f"No valid segmentations for image {image_data['image_name']}, category {category_id}")
                        continue
                    
                    # Resize mask
                    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                    
                    # Validate mask has some positive pixels
                    if np.sum(mask) == 0:
                        logger.warning(f"Empty mask after resize for {image_data['image_name']}")
                        continue

                    images_data.append(image_pil)
                    categories_data.append(category_index)
                    labels_data.append(mask)
            
            except Exception as e:
                logger.warning(f"Error processing image {img_path}: {e}")
                failed_loads += 1
                continue

        logger.info(f"Dataset created: {len(images_data)} samples, {failed_loads} failed loads")
        return images_data, labels_data, categories_data

    def prepare_datasets(self, batch_size, ds_root, image_size, log_dir):
        ds_name = "train"
        logger.info("Loading COCO data...")
        coco_data_train = self._load_coco_data(ds_root, ds_name)
        logger.info("Creating category mappings...")
        categories_classes, category_encoder, category_decoder = self._create_category_mapping(coco_data_train)
        logger.info("Creating dataset...")
        X_train, Y_train, Cat_train = self._create_dataset(
            coco_data_train, ds_root, ds_name, category_encoder, image_size
        )

        ds_name = "valid"
        logger.info("Loading COCO data...")
        coco_data_valid = self._load_coco_data(ds_root, ds_name)
        logger.info("Creating dataset...")
        X_valid, Y_valid, Cat_valid = self._create_dataset(
            coco_data_valid, ds_root, ds_name, category_encoder, image_size
        )

        ds_name = "test"
        logger.info("Loading COCO data...")
        coco_data_test = self._load_coco_data(ds_root, ds_name)
        logger.info("Creating dataset...")
        X_test, Y_test, Cat_test = self._create_dataset(
            coco_data_test, ds_root, ds_name, category_encoder, image_size
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
        
        # Plot category distribution
        self.plot_manager.plot_category_distribution(Cat_train, categories_classes, category_decoder)
        
        # Plot sample data
        self.plot_manager.plot_data_samples(X_train, Y_train, Cat_train, categories_classes, category_decoder, num_samples=1)

        train_dataset = DatasetManager(X_train, Y_train, Cat_train, augment=True)
        val_dataset = DatasetManager(X_valid, Y_valid, Cat_valid, augment=False)
        test_dataset = DatasetManager(X_test, Y_test, Cat_test, augment=False)

        logger.info("Train: ", len(train_dataset))
        logger.info("Val: ", len(val_dataset))
        logger.info("Test: ", len(test_dataset))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        return train_loader, val_loader, test_loader, categories_classes, category_encoder, category_decoder