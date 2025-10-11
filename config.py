import os

DS_ROOT = r"C:\Users\natha\Downloads\medico 2.v3i.coco-segmentation" #"D:\Datasets\fcn_datasets\hand"
RES_DIR = "result"
LOG_DIR = "result/logs"

N_EPOCHS = 1000
PATIENCE = 15
LR = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 15
DROPOUT_RATE = 0.1
IMAGE_SIZE = (150, 300) # w,h

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)