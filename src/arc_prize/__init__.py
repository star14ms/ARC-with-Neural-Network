from dataclasses import dataclass
from typing import List


def get_model_class(model_name: str):
    from arc_prize.model_lightning import (
        ARCSameShapeConvL,
    )

    if model_name == 'ARCSameShapeConv':
        model_class = ARCSameShapeConvL
    else:
        raise ValueError(f"Model name {model_name} not found")

    return model_class

@dataclass
class DataConfig:
    name: str = 'default'
    batch_size: int = 1

@dataclass
class TrainConfig:
    name: str = 'default'
    epoch: int = 1

@dataclass
class ModelConfig:
    name: str = 'ARCSameShapeConvL'
    pass

@dataclass
class ARCSameShapeConvConfig:
    input_size: List[int] = (5, 5)
