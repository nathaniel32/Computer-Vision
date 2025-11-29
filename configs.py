import os
from dataclasses import dataclass

RES_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
SEED = 42

@dataclass
class BaseConfig:
    name: str
    model_num_points: int
    epochs: int
    batch_size: int
    lr: float
    patience: int
    ds_root: str
    target_color_augment_labels: list
    classes: list

hand_config = BaseConfig(
    name = "hand",
    model_num_points=10000,
    epochs=2000,
    batch_size=12,
    lr=1e-3,
    patience=20,

    ds_root=r"D:\Datasets\3d\medico_part_color\datasets\latest",
    target_color_augment_labels=[0, 1],
    classes=[
        {"label": "background", "color": "#FFFFFF"},
        {"label": "hand", "color": "#00FF00"}
    ]
)

marker_config = BaseConfig(
    name = "marker",
    model_num_points=100000,
    epochs=2000,
    batch_size=12,
    lr=1e-3,
    patience=20,

    ds_root=r"D:\Datasets\3d\medico_marker\latest",
    target_color_augment_labels=[0, 1, 2, 3, 4],
    classes=[
        {"label": "background", "color": "#FFFFFF"},
        {"label": "Red", "color": "#FF0000"},
        {"label": "Green", "color": "#00FF00"},
        {"label": "Blue", "color": "#0000FF"},
        {"label": "Yellow", "color": "#FFFF00"}
    ]
)

selected_config = marker_config