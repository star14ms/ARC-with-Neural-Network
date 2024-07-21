from dataclasses import dataclass
from typing import List


def get_model_class(model_name: str):
    from arc_prize.model.fill.lightning import (
        FillerKeepInputL,
        FillerKeepInputIgnoreColorL,
    )

    if model_name in ['FillerKeepInput', 'FillerKeepInputL']:
        model_class = FillerKeepInputL
    elif model_name in ['FillerKeepInputIgnoreColor', 'FillerKeepInputIgnoreColorL']:
        model_class = FillerKeepInputIgnoreColorL
    else:
        raise ValueError(f"Model name {model_name} not found")

    return model_class

@dataclass
class DataConfig:
    name: str = 'default'
    batch_size: int = 1
    cold_value: int = -1
    augment_data: bool = False
    ignore_color: bool = False

@dataclass
class TrainConfig:
    name: str = 'default'
    epoch: int = 1
    
@dataclass
class TestConfig:
    model_path: str = "./output/model_FillerKeepInput.pth"
    verbose_single: bool = False

@dataclass
class ModelConfig:
    name: str = 'FillerKeepInput'
    pass

@dataclass
class FillerKeepInputConfig:
    reduced_channels_encoder: List[int] = (512, 32)
    reduced_channels_decoder: List[int] = (128, 32)

@dataclass
class FillerKeepInputIgnoreColorConfig:
    reduced_channels_encoder: List[int] = (512, 32)
    reduced_channels_decoder: List[int] = (128, 32)