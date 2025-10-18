DS_ROOT = r"D:\Datasets\hand_part_color\datasets"
LOG_DIR = "result"
RES_DIR = "result"

NUM_SAMPLE_POINTS = 30000
CLASSES = [
    {"label": "background", "color": "#FFFFFF"},
    {"label": "hand", "color": "#00FF00"}
]

EPOCHS = 200
BATCH_SIZE = 3
LR = 1e-3
PATIENCE = 15