import os
from dataclasses import dataclass
from typing import List

@dataclass
class BaseConfig:
    name: str
    save_model_path: str
    save_log_path: str
    total_num_points: int
    model_num_points: int
    epochs: int
    batch_size: int
    lr: float
    patience: int
    ds_root: str
    target_color_augment_labels: list
    scale_labels: list
    classes: list

    def __post_init__(self):
        CONFIG_LIST.append(self)

RES_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
SEED = 42
CONFIG_LIST:List[BaseConfig] = []

hand_config = BaseConfig(
    name = "hand",
    save_model_path = os.path.join(RES_ROOT_DIR, "hand", "best_model.pth"),
    save_log_path = os.path.join(RES_ROOT_DIR, "hand", "log.txt"),
    total_num_points=200000,
    model_num_points=10000,
    epochs=2000,
    batch_size=12,
    lr=1e-3,
    patience=20,
    ds_root=r"D:\Datasets\3d\medico_part_color\datasets\latest",
    target_color_augment_labels=[0, 1],
    scale_labels=[1],
    classes=[
        {"label": "background", "color": "#FFFFFF"},
        {"label": "hand", "color": "#00FF00"}
    ]
)

marker_config = BaseConfig(
    name = "marker",
    save_model_path = os.path.join(RES_ROOT_DIR, "marker", "best_model.pth"),
    save_log_path = os.path.join(RES_ROOT_DIR, "marker", "log.txt"),
    total_num_points=500000,
    model_num_points=100000,
    epochs=2000,
    batch_size=2,
    lr=1e-3,
    patience=20,
    ds_root=r"D:\Datasets\3d\medico_marker\latest",
    target_color_augment_labels=[],
    scale_labels=[1, 2, 3, 4],
    classes=[
        {"label": "background", "color": "#FFFFFF"},
        {"label": "Red", "color": "#FF0000"},
        {"label": "Green", "color": "#00FF00"},
        {"label": "Blue", "color": "#0000FF"},
        {"label": "Yellow", "color": "#FFFF00"}
    ]
)