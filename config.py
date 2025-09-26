import os

DS_ROOT = r"D:\Datasets\fcn_datasets\hand"
RES_DIR = "result"
LOG_DIR = "result/logs"

N_EPOCHS = 1
PATIENCE = 15
LR = 0.001
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 15
DROPOUT_RATE = 0.1

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)