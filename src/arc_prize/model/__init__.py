from dataclasses import dataclass
from typing import List


def get_model_class(model_name: str):
    from arc_prize.model.fill.lightning import (
        FillerKeepInputL,
        FillerKeepInputIgnoreColorL,
    )
    from arc_prize.model.substitute.lightning import (
        PixelEachSubstitutorL,
    )

    model_classes = [
        FillerKeepInputL,
        FillerKeepInputIgnoreColorL,
        PixelEachSubstitutorL,
    ]

    for model_class in model_classes:
        model_class_name = model_class.__name__
        if model_name == model_class_name or model_name + 'L' == model_class_name:
            return model_class
    else:
        raise ValueError(f"Model name {model_name} not found")


@dataclass
class DataConfig:
    name: str = 'default'
    cold_value: int = -1
    ignore_color: bool = False

@dataclass
class TrainConfig:
    max_epochs: int = 1
    augment_data: bool = True
    batch_size_max: int = 8
    lr: float = 0.01
    save_dir: str = "./output/"
    ckpt_path: str | None = None

@dataclass
class TestConfig:
    model_path: str = "./output/model_FillerKeepInput.ckpt"
    verbose_single: bool = False
    augment_data: bool = False

@dataclass
class ModelConfig:
    name: str = 'FillerKeepInput'
    pass

@dataclass
class FillerKeepInputConfig:
    reduced_channels_encoder: List[int] = (512, 32)
    reduced_channels_decoder: List[int] = (128, 32)
    pad_value: int = -1
    d_conv_feature: int = 16
    d_class_feature: int = 32

@dataclass
class FillerKeepInputIgnoreColorConfig:
    reduced_channels_encoder: List[int] = (512, 32)
    reduced_channels_decoder: List[int] = (128, 32)

@dataclass
class PixelEachSubstitutorConfig:
    pass