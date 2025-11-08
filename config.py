import os

DS_ROOT = r"D:\Datasets\medico_part_color\datasets\mix"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

NUM_SAMPLE_POINTS = 10000
CLASSES = [
    {"label": "background", "color": "#FFFFFF"},
    {"label": "hand", "color": "#00FF00"}
]

SEED = 100

TARGET_CLASS_ID = 1

EPOCHS = 2000
BATCH_SIZE = 12
LR = 1e-3
PATIENCE = 20