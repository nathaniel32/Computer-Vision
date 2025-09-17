import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAINED_PATH = "data/pretrained.pth"
META_PATH = "data/meta.bin"
RETRAIN_MODEL = False
SAVE_MODEL = True
SEED = 42

EPOCHS = 100
TRAIN_BATCH_SIZE = 120
VALIDATION_BATCH_SIZE = 120
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4

SEQUENCE_CANDLE_LENGTH = 100
HORIZON_LABEL = 5
THRESHOLD_LABEL = 0.00000

IMG_SIZE = 224

PRETRAINED = True
DATASET_PATH = "datasets/finetune"