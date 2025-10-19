DS_ROOT = r"D:\Datasets\hand_part_color\datasets" #"C:\Users\natha\Downloads\d"
LOG_DIR = "result"
RES_DIR = "result"

NUM_SAMPLE_POINTS = 10000
CLASSES = [
    {"label": "background", "color": "#FFFFFF"},
    {"label": "hand", "color": "#00FF00"}
]

EPOCHS = 2000
BATCH_SIZE = 8
LR = 1e-3
PATIENCE = 20