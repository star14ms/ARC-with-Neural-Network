from dataclasses import dataclass
from typing import List


def get_model_class(model_name: str):
    from arc_prize.model_lightning import (
        ShapeStableSolverL,
    )

    if model_name == 'ShapeStableSolver':
        model_class = ShapeStableSolverL
    else:
        raise ValueError(f"Model name {model_name} not found")

    return model_class

@dataclass
class DataConfig:
    name: str = 'default'
    batch_size: int = 1
    augment_data: bool = False

@dataclass
class TrainConfig:
    name: str = 'default'
    epoch: int = 1

@dataclass
class ModelConfig:
    name: str = 'ShapeStableSolver'
    pass

@dataclass
class ShapeStableSolverConfig:
    reduced_channels_encoder: List[int] = (512, 128, 32)
    reduced_channels_decoder: List[int] = (128, 32)
