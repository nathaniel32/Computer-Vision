import config
from helper.coco_manager import CocoManager

batch_size = 10
ds_root = r"D:\Documents\debug\face_object"
res_dir = config.RES_DIR
log_dir = config.LOG_DIR
image_size = (224, 224)

coco_manager = CocoManager()
train_loader, val_loader, test_loader, categories_classes = coco_manager.prepare_datasets(batch_size, ds_root, image_size, log_dir)