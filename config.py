import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

NUM_SAMPLE_POINTS = 10000
SEED = 42
EPOCHS = 2000
BATCH_SIZE = 12
LR = 1e-3
PATIENCE = 20

DS_ROOT = r"D:\Datasets\3d\medico_part_color\datasets\latest"
TARGET_COLOR_AUGMENT_LABELS = [0, 1]
CLASSES = [
    {"label": "background", "color": "#FFFFFF"},
    {"label": "hand", "color": "#00FF00"}
]